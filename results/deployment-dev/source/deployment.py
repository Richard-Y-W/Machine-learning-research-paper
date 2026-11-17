"""Same-architecture factorial study of targets, initialization, and edit geometry."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .core import (DynamicsNet, acceleration, coefficients, features, fit_coefficients,
                   intervention, jacobian, parameter_vector, reparameterize, rollout,
                   sample, set_vector, train)


METHODS = ["warm_adam", "cold_adam", "warm_lbfgs", "cold_lbfgs",
           "euclidean_jacobian", "column_jacobian", "relative_jacobian"]
SCHEDULES = {"short": dict(adam=150, lbfgs=30, jacobian=3),
             "long": dict(adam=600, lbfgs=120, jacobian=12)}


def relative_step(j, residual, theta, ridge=1e-4):
    """Diagonal-coordinate equivariant step; zero coordinates remain frozen."""
    scale = theta.abs()
    a = j * scale
    kernel = a @ a.T
    lam = ridge * kernel.diag().mean().clamp_min(1e-15)
    alpha = torch.linalg.solve(kernel + lam * torch.eye(len(j), dtype=j.dtype), residual)
    return scale * (a.T @ alpha)


def initialize(base, seed, method, gauge):
    if method.startswith("cold_"):
        # Cold initialization has no access to the pretrained parameters or gauge.
        with torch.random.fork_rng():
            torch.manual_seed(900000 + seed)
            return DynamicsNet(base.net[0].out_features).double().eval()
    return reparameterize(base, 80000 + seed, gauge)


def fit_student(base, x, target, seed, method, gauge, schedule, ridge=1e-4):
    out = initialize(base, seed, method, gauge)
    counts = dict(forward_points=0, gradient_points=0, jacobian_output_points=0,
                  accepted_updates=0)
    config = SCHEDULES[schedule]
    if method.endswith("adam"):
        optimizer = torch.optim.Adam(out.parameters(), lr=.003)
        for _ in range(config["adam"]):
            optimizer.zero_grad(set_to_none=True)
            loss = (out(x)-target).square().mean()
            loss.backward()
            optimizer.step()
            counts["forward_points"] += len(x)
            counts["gradient_points"] += len(x)
            counts["accepted_updates"] += 1
    elif method.endswith("lbfgs"):
        optimizer = torch.optim.LBFGS(out.parameters(), lr=1., max_iter=config["lbfgs"],
            max_eval=config["lbfgs"]*2, history_size=20, line_search_fn="strong_wolfe",
            tolerance_grad=1e-9, tolerance_change=1e-12)
        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = (out(x)-target).square().mean()
            loss.backward()
            counts["forward_points"] += len(x)
            counts["gradient_points"] += len(x)
            return loss
        optimizer.step(closure)
        state=next(iter(optimizer.state.values()))
        counts["accepted_updates"] = int(state.get("n_iter",0))
    else:
        from .core import ridge_step
        for _ in range(config["jacobian"]):
            with torch.no_grad():
                residual=target-out(x)
                old_loss=float(residual.square().mean())
            counts["forward_points"] += len(x)
            j=jacobian(out,x)
            counts["jacobian_output_points"] += len(x)
            before=parameter_vector(out)
            if method=="relative_jacobian":
                delta=relative_step(j,residual,before,ridge)
            else:
                delta=ridge_step(j,residual,method=="column_jacobian",ridge)
            accepted=False
            for alpha in (1.,.5,.25,.125,.0625,.03125,.015625):
                set_vector(out,before+alpha*delta)
                with torch.no_grad():
                    loss=float((out(x)-target).square().mean())
                counts["forward_points"] += len(x)
                if np.isfinite(loss) and loss<old_loss:
                    accepted=True
                    counts["accepted_updates"] += 1
                    break
            if not accepted:
                set_vector(out,before)
                break
    return out,counts


def rms(x):
    result=float(x.square().mean().sqrt())
    return result if np.isfinite(result) else None


def latency(predict,x,repeats=20):
    # CPU execution is synchronous. Timing is descriptive, not a compute budget.
    with torch.no_grad():
        for _ in range(3):
            predict(x)
        t=time.perf_counter()
        for _ in range(repeats):
            predict(x)
    return (time.perf_counter()-t)*1000/repeats


def run(args):
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    output=Path(args.output)
    if (output/"manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True,exist_ok=True)
    (output/"checkpoints").mkdir(exist_ok=True)
    (output/"source").mkdir(exist_ok=True)
    source_hash=hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        shutil.copy2(path,output/"source"/path.name)
        source_hash.update(path.name.encode())
        source_hash.update(path.read_bytes())
    manifest=dict(config=vars(args),status="running",started_utc=datetime.now(timezone.utc).isoformat(),
                  python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
                  device="cpu",source_sha256=source_hash.hexdigest(),schedules=SCHEDULES,
                  semantic_prior="shared known four-term mechanism library",
                  primary_contrast="warm_lbfgs minus cold_lbfgs; symbolic; 128 queries; long; original")
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    started=time.perf_counter()
    count=0
    models=[]
    with (output/"raw.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=None
        for seed in range(args.seed_start,args.seed_start+args.seeds):
            for family in args.families:
                base,diagnostics=train(seed,family,args.width,args.train_steps)
                c=coefficients(seed,family)
                mult=intervention("combined")
                models.append(dict(seed=seed,family=family,**diagnostics))
                torch.save(dict(state_dict=base.state_dict(),seed=seed,family=family,width=args.width),
                           output/"checkpoints"/f"{family}_{seed}.pt")
                xall=sample(max(args.budgets),60000+seed)
                test=sample(args.test_points,170000+seed,1.25)
                shell=sample(args.test_points*3,171000+seed,2.)
                shell=shell[shell.abs().amax(1)>1.5][:args.test_points]
                initial=sample(24,172000+seed,.8)[:,:2]
                tt=torch.arange(100,dtype=torch.float64)[:,None]
                actions=.5*torch.sin(tt*.07+torch.arange(24,dtype=torch.float64)[None,:]*.4)
                with torch.no_grad():
                    physical_id=acceleration(test,c*(1+mult),family)
                    physical_ood=acceleration(shell,c*(1+mult),family)
                    reference=rollout(lambda z:acceleration(z,c*(1+mult),family),initial,actions)
                for budget in args.budgets:
                    x=xall[:budget]
                    with torch.no_grad():
                        base_y=base(x)
                        fitted=fit_coefficients(x,base_y,family)
                    for teacher in args.teachers:
                        def predict_teacher(z):
                            if teacher=="residual":
                                return base(z)+features(z,family)@(fitted*mult)
                            return features(z,family)@(fitted*(1+mult))
                        with torch.no_grad():
                            target=predict_teacher(x)
                            teacher_id=predict_teacher(test)
                            teacher_ood=predict_teacher(shell)
                        teacher_latency=latency(predict_teacher,test[:256])
                        for schedule in args.schedules:
                            originals={}
                            for gauge in args.gauges:
                                for method in args.methods:
                                    t0=time.perf_counter()
                                    student,counts=fit_student(base,x,target,seed,method,gauge,
                                                              schedule,args.ridge)
                                    seconds=time.perf_counter()-t0
                                    with torch.no_grad():
                                        prediction=student(test)
                                        extrapolated=student(shell)
                                        trajectory=rollout(student,initial,actions)
                                        if gauge=="original":
                                            originals[method]=prediction.clone()
                                        disagreement=rms(prediction-originals[method])
                                        student_latency=latency(student,test[:256])
                                        row=dict(seed=seed,family=family,budget=budget,teacher=teacher,
                                            schedule=schedule,gauge=gauge,method=method,
                                            id_rmse=rms(prediction-physical_id),
                                            ood_rmse=rms(extrapolated-physical_ood),
                                            teacher_fit_rmse=rms(prediction-teacher_id),
                                            teacher_ood_fit_rmse=rms(extrapolated-teacher_ood),
                                            calibration_fit_rmse=rms(student(x)-target),
                                            teacher_id_rmse=rms(teacher_id-physical_id),
                                            teacher_ood_rmse=rms(teacher_ood-physical_ood),
                                            rollout_rmse=rms(trajectory-reference),
                                            rollout_nonfinite=int(not torch.isfinite(trajectory).all()),
                                            gauge_disagreement=disagreement,
                                            edit_seconds=seconds,student_batch256_ms=student_latency,
                                            teacher_batch256_ms=teacher_latency,
                                            parameters=len(parameter_vector(student)),
                                            coefficient_rmse=rms(fitted-c),**counts)
                                    if writer is None:
                                        writer=csv.DictWriter(handle,fieldnames=list(row))
                                        writer.writeheader()
                                    writer.writerow(row)
                                    count+=1
                    handle.flush()
                (output/"models.json").write_text(json.dumps(models,indent=2))
                print(json.dumps(dict(seed=seed,family=family,rows=count,
                    elapsed_seconds=round(time.perf_counter()-started,1),**diagnostics)),flush=True)
    manifest.update(status="complete",rows=count,elapsed_seconds=time.perf_counter()-started,
                    completed_utc=datetime.now(timezone.utc).isoformat())
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(status="complete",rows=count,seconds=manifest["elapsed_seconds"])),flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    p.add_argument("--seed-start",type=int,default=20)
    p.add_argument("--seeds",type=int,default=12)
    p.add_argument("--width",type=int,default=32)
    p.add_argument("--train-steps",type=int,default=2400)
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--budgets",type=int,nargs="+",default=[32,128])
    p.add_argument("--families",nargs="+",choices=["duffing","pendulum"],default=["duffing","pendulum"])
    p.add_argument("--teachers",nargs="+",choices=["residual","symbolic"],default=["residual","symbolic"])
    p.add_argument("--schedules",nargs="+",choices=list(SCHEDULES),default=list(SCHEDULES))
    p.add_argument("--gauges",nargs="+",choices=["original","rescaled"],default=["original","rescaled"])
    p.add_argument("--methods",nargs="+",choices=METHODS,default=METHODS)
    p.add_argument("--test-points",type=int,default=1024)
    p.add_argument("--ridge",type=float,default=1e-4)
    return p


if __name__=="__main__":
    args=parser().parse_args()
    if args.gauges[0]!="original":
        raise ValueError("Original representation must be evaluated first")
    run(args)
