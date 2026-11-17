"""Compare parameter reuse, internal features, and black-box functional reuse."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import shutil
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import torch

from .core import DynamicsNet, coefficients, features, parameter_vector, rollout, sample
from .deployment import rms

FAMILIES=("duffing","pendulum","contact","vanderpol")
METHODS=("source","warm_full","fresh_full","learned_head","linear_residual",
         "cubic_residual","rbf_residual","ard_rbf_residual","neural_residual")
RIDGES=(1e-5,1e-4,1e-3,1e-2,.1)
LENGTHS=(.25,.5,1.,2.)
POWERS=[()] + [p for degree in (1,2,3) for p in itertools.combinations_with_replacement(range(3),degree)]


def source_acceleration(x,c,family):
    q,v,u=x.unbind(-1)
    if family in ("duffing","pendulum"):
        return features(x,family)@c.to(x)
    linear=-c[0]*q-c[1]*v+c[3]*u
    if family=="contact":
        return linear-c[2]*torch.sign(q)*torch.relu(q.abs()-.7)
    if family=="vanderpol":
        return linear+c[2]*(1-q.square())*v
    raise ValueError(family)


def changed_acceleration(x,c,family,shift):
    y=source_acceleration(x,c,family)
    q,v,_=x.unbind(-1)
    if shift=="parameter":
        return y-.5*c[0]*q-c[1]*v
    if shift=="nonlinear":
        return y-.2*torch.sin(3*q)-.3*torch.tanh(4*v)
    raise ValueError(shift)


def train_source(seed,family):
    torch.manual_seed(1000+seed)
    model=DynamicsNet(32).float()
    x=sample(4096,10000+seed).float()
    y=source_acceleration(x,coefficients(seed,family).float(),family)
    generator=torch.Generator().manual_seed(20000+seed)
    optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    for step in range(2400):
        idx=torch.randint(len(x),(256,),generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss=(model(x[idx])-y[idx]).square().mean()
        loss.backward();optimizer.step()
        if step==1800:
            for group in optimizer.param_groups:
                group["lr"]*=.2
    return model.double().eval()


def polynomial(x,degree):
    normalized=x/1.5
    columns=[]
    for p in POWERS:
        if len(p)<=degree:
            columns.append(normalized[:,list(p)].prod(1) if p else torch.ones(len(x),dtype=x.dtype))
    return torch.stack(columns,1)


def loo_solve(kernel,y):
    values,vectors=torch.linalg.eigh(kernel)
    values=values.clamp_min(0)
    projected=vectors.T@y
    best=None
    for ridge in RIDGES:
        inverse=1/(values+len(y)*ridge)
        alpha=vectors@(projected*inverse)
        inv_diagonal=(vectors.square()*inverse[None,:]).sum(1)
        score=float((alpha/inv_diagonal).square().mean())
        if best is None or score<best[0]:
            best=(score,ridge,alpha)
    return best


def feature_residual(feature_fn,x,residual):
    with torch.no_grad():
        phi=feature_fn(x)
        scale=phi.square().mean(0).sqrt().clamp_min(1e-12)
        normalized=phi/scale
        score,ridge,alpha=loo_solve(normalized@normalized.T,residual)
        beta=(normalized.T@alpha)/scale
    def predict(z):
        return feature_fn(z)@beta
    return predict,dict(selected_ridge=ridge,selected_length=0.,cv_mse=score,
                        extra_scalars=len(beta)),beta


def rbf_residual(source_predict,x,y,ard=False):
    """Only a callable source predictor is needed, not parameters/activations."""
    with torch.no_grad():
        residual=y-source_predict(x)
        normalized=x/1.5
        best=None
        candidates=itertools.product(LENGTHS,repeat=3) if ard else ((s,s,s) for s in LENGTHS)
        for lengths in candidates:
            scale=torch.tensor(lengths,dtype=x.dtype)
            distances=torch.cdist(normalized/scale,normalized/scale).square()
            kernel=torch.exp(-distances/2)
            score,ridge,alpha=loo_solve(kernel,residual)
            if best is None or score<best[0]:
                best=(score,ridge,lengths,alpha)
        score,ridge,lengths,alpha=best
        scale=torch.tensor(lengths,dtype=x.dtype)
    def predict(z):
        kernel=torch.exp(-torch.cdist((z/1.5)/scale,(x/1.5)/scale).square()/2)
        return source_predict(z)+kernel@alpha
    return predict,dict(selected_ridge=ridge,selected_length=float(np.mean(lengths)),cv_mse=score,
                        length_q=lengths[0],length_v=lengths[1],length_u=lengths[2],
                        extra_scalars=4*len(x)+4)


def fresh_network(seed,residual=False):
    with torch.random.fork_rng():
        torch.manual_seed(900000+seed)
        model=DynamicsNet(32).double().eval()
    if residual:
        with torch.no_grad():
            model.net[4].weight.zero_();model.net[4].bias.zero_()
    return model


def optimize_lbfgs(initial,x,y,budgets,validation=None):
    model=copy.deepcopy(initial)
    optimizer=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=30,max_eval=60,
        history_size=20,line_search_fn="strong_wolfe",tolerance_grad=1e-9,tolerance_change=1e-12)
    calls=0
    scores={}
    for step in range(30,max(budgets)+1,30):
        def closure():
            nonlocal calls
            optimizer.zero_grad(set_to_none=True)
            loss=(model(x)-y).square().mean()
            loss.backward();calls+=1
            return loss
        optimizer.step(closure)
        if step in budgets and validation is not None:
            vx,vy=validation
            with torch.no_grad():
                scores[step]=float((model(vx)-vy).square().mean())
    return model,scores,calls


def fit_neural(base,source_predict,x,y,seed,method):
    residual_mode=method=="neural_residual"
    initial=copy.deepcopy(base) if method=="warm_full" else fresh_network(seed,residual_mode)
    with torch.no_grad():
        target=y-source_predict(x) if residual_mode else y
    order=np.random.default_rng(480000+seed).permutation(len(x))
    split=max(1,int(.75*len(x)))
    train_idx,val_idx=order[:split],order[split:]
    _,scores,cv_calls=optimize_lbfgs(initial,x[train_idx],target[train_idx],(30,120,480),
                                   validation=(x[val_idx],target[val_idx]))
    selected=min(scores,key=scores.get)
    student,_,refit_calls=optimize_lbfgs(initial,x,target,(selected,))
    if residual_mode:
        def predict(z):
            return source_predict(z)+student(z)
    else:
        predict=student
    return predict,dict(selected_iterations=selected,cv_mse=scores[selected],
        gradient_points=cv_calls*len(train_idx)+refit_calls*len(x),
        extra_scalars=1217 if residual_mode else 0)


def adapt(base,x,y,seed,method):
    source_predict=lambda z:base(z)
    if method=="source":
        return source_predict,dict(extra_scalars=0)
    if method in ("warm_full","fresh_full","neural_residual"):
        return fit_neural(base,source_predict,x,y,seed,method)
    if method in ("rbf_residual","ard_rbf_residual"):
        return rbf_residual(source_predict,x,y,ard=method=="ard_rbf_residual")
    with torch.no_grad():
        residual=y-source_predict(x)
    if method=="learned_head":
        def phi(z):
            h=base.hidden(z)
            return torch.cat((h,torch.ones(len(z),1,dtype=z.dtype)),1)
        _,meta,beta=feature_residual(phi,x,residual)
        out=copy.deepcopy(base)
        with torch.no_grad():
            out.net[4].weight.add_(beta[:-1][None,:]);out.net[4].bias.add_(beta[-1:])
        meta["extra_scalars"]=0
        return out,meta
    degree=1 if method=="linear_residual" else 3
    correction,meta,_=feature_residual(lambda z:polynomial(z,degree),x,residual)
    return lambda z:source_predict(z)+correction(z),meta


def run(args):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    out=Path(args.output)
    if (out/"manifest.json").exists():
        raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True);(out/"source").mkdir(exist_ok=True);(out/"checkpoints").mkdir(exist_ok=True)
    h=hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        shutil.copy2(path,out/"source"/path.name);h.update(path.name.encode());h.update(path.read_bytes())
    manifest=dict(config=vars(args),status="running",started_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=h.hexdigest(),git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        torch=torch.__version__,noise_std=.01,physics_library_supplied=False,
        task="few-shot adaptation from actual changed-environment acceleration observations",
        primary="nonlinear shift, 64 labels: RBF residual minus warm full; RBF residual minus learned head")
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    start=time.perf_counter();count=0;models=[]
    with (out/"raw.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=None
        for seed in range(args.seed_start,args.seed_start+args.seeds):
            for family in args.families:
                base=train_source(seed,family)
                c=coefficients(seed,family)
                torch.save(dict(state_dict=base.state_dict(),width=32,seed=seed,family=family),
                           out/"checkpoints"/f"{family}_{seed}.pt")
                test=sample(1024,570000+seed,1.25)
                shell=sample(3072,571000+seed,2.)
                shell=shell[shell.abs().amax(1)>1.5][:1024]
                with torch.no_grad():
                    base_error=rms(base(test)-source_acceleration(test,c,family))
                models.append(dict(seed=seed,family=family,source_rmse=base_error))
                probes=sample(max(args.budgets),560000+seed)
                noise=torch.tensor(np.random.default_rng(561000+seed).normal(0,.01,max(args.budgets)),dtype=torch.float64)
                initial=sample(24,572000+seed,.8)[:,:2]
                t=torch.arange(100,dtype=torch.float64)[:,None]
                actions=.5*torch.sin(t*.07+torch.arange(24,dtype=torch.float64)[None,:]*.4)
                for shift in ("parameter","nonlinear"):
                    with torch.no_grad():
                        actual=changed_acceleration(test,c,family,shift)
                        actual_shell=changed_acceleration(shell,c,family,shift)
                        reference=rollout(lambda z:changed_acceleration(z,c,family,shift),initial,actions)
                    for budget in args.budgets:
                        x=probes[:budget]
                        y=changed_acceleration(x,c,family,shift)+noise[:budget]
                        for method in args.methods:
                            begin=time.perf_counter()
                            prediction,meta=adapt(base,x,y,seed,method)
                            seconds=time.perf_counter()-begin
                            with torch.no_grad():
                                pred=prediction(test)
                                trajectory=rollout(prediction,initial,actions)
                                row=dict(seed=seed,family=family,shift=shift,budget=budget,method=method,
                                    id_rmse=rms(pred-actual),ood_rmse=rms(prediction(shell)-actual_shell),
                                    rollout_rmse=rms(trajectory-reference),
                                    rollout_nonfinite=int(not torch.isfinite(trajectory).all()),
                                    calibration_noisy_rmse=rms(prediction(x)-y),source_rmse=base_error,
                                    adaptation_seconds=seconds,extra_scalars=meta.get("extra_scalars",0),
                                    selected_ridge=meta.get("selected_ridge",0.),selected_length=meta.get("selected_length",0.),
                                    length_q=meta.get("length_q",0.),length_v=meta.get("length_v",0.),length_u=meta.get("length_u",0.),
                                    selected_iterations=meta.get("selected_iterations",0),cv_mse=meta.get("cv_mse",0.),
                                    gradient_points=meta.get("gradient_points",0))
                            if writer is None:
                                writer=csv.DictWriter(handle,fieldnames=list(row));writer.writeheader()
                            writer.writerow(row);count+=1
                    handle.flush()
                (out/"models.json").write_text(json.dumps(models,indent=2))
                print(json.dumps(dict(seed=seed,family=family,rows=count,source_rmse=base_error,
                    elapsed_seconds=round(time.perf_counter()-start,1))),flush=True)
    manifest.update(status="complete",rows=count,elapsed_seconds=time.perf_counter()-start,
                    completed_utc=datetime.now(timezone.utc).isoformat())
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True);p.add_argument("--seed-start",type=int,default=60)
    p.add_argument("--seeds",type=int,default=12)
    p.add_argument("--families",nargs="+",choices=FAMILIES,default=list(FAMILIES))
    p.add_argument("--budgets",type=int,nargs="+",default=[16,64])
    p.add_argument("--methods",nargs="+",choices=METHODS,default=list(METHODS))
    run(p.parse_args())
