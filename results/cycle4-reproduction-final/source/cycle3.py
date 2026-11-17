"""Longer optimization and hidden-family query identification on fresh models."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import torch

from .core import (DynamicsNet, acceleration, coefficients, features, fit_coefficients,
                   intervention, parameter_vector, sample, train)
from .deployment import initialize, rms

FAMILIES=("duffing","pendulum")


def query_design(budget,seed,design):
    if design=="narrow":
        return sample(budget,260000+seed,.25)
    if design=="broad":
        return sample(budget,261000+seed,1.5)
    if design=="mixed":
        if budget<8:
            raise ValueError("Mixed design needs at least eight queries")
        return torch.cat((sample(8,260000+seed,.25),sample(budget-8,261000+seed,1.5)))
    raise ValueError(design)


def select_family(x,y,seed):
    """No physical family label or true coefficients are passed to this routine."""
    if len(x)<8:
        raise ValueError("Four-fold identification requires >=8 points")
    permutation=np.random.default_rng(270000+seed).permutation(len(x))
    folds=np.array_split(permutation,4)
    scores={}
    for family in FAMILIES:
        errors=[]
        for fold in folds:
            mask=np.ones(len(x),dtype=bool);mask[fold]=False
            fitted=fit_coefficients(x[mask],y[mask],family)
            errors.append((features(x[fold],family)@fitted-y[fold]).square().mean())
        scores[family]=float(torch.stack(errors).mean())
    selected=min(FAMILIES,key=lambda family:scores[family])
    return selected,fit_coefficients(x,y,selected),scores


def load_or_train(output,seed,family,steps):
    path=output/"checkpoints"/f"{family}_{seed}.pt"
    if path.exists():
        saved=torch.load(path,weights_only=True)
        model=DynamicsNet(saved["width"]).double().eval()
        model.load_state_dict(saved["state_dict"])
        return model,saved["diagnostics"]
    model,diagnostics=train(seed,family,32,steps)
    torch.save(dict(state_dict=model.state_dict(),width=32,seed=seed,family=family,
                    diagnostics=diagnostics),path)
    return model,diagnostics


def long_fit(base,x,target,seed,warm,test,teacher_test,physical_test,shell,physical_shell,
             *,cap=1920,threshold=.002):
    student=initialize(base,seed,"warm_lbfgs" if warm else "cold_lbfgs","original")
    optimizer=torch.optim.LBFGS(student.parameters(),lr=1.,max_iter=30,max_eval=60,
        history_size=20,line_search_fn="strong_wolfe",tolerance_grad=1e-9,tolerance_change=1e-12)
    closures=0
    start=time.perf_counter()
    records=[]
    trace=[]
    crossed=False
    for budget in range(30,cap+1,30):
        def closure():
            nonlocal closures
            optimizer.zero_grad(set_to_none=True)
            loss=(student(x)-target).square().mean()
            loss.backward()
            closures+=1
            return loss
        optimizer.step(closure)
        state=next(iter(optimizer.state.values()))
        with torch.no_grad():
            calibration=rms(student(x)-target)
        trace.append(dict(iteration_budget=budget,accepted_iterations=int(state.get("n_iter",0)),
                          closure_calls=closures,calibration_fit_rmse=calibration))
        events=[]
        if not crossed and calibration is not None and calibration<=threshold:
            crossed=True
            events.append("threshold")
        if budget in (120,480,cap):
            events.append("fixed")
        if events:
            with torch.no_grad():
                prediction=student(test)
                metrics=dict(id_rmse=rms(prediction-physical_test),
                    teacher_fit_rmse=rms(prediction-teacher_test),
                    ood_rmse=rms(student(shell)-physical_shell),
                    calibration_fit_rmse=calibration,iteration_budget=budget,
                    accepted_iterations=int(state.get("n_iter",0)),closure_calls=closures,
                    elapsed_seconds=time.perf_counter()-start,threshold_attained=int(crossed),
                    parameters=len(parameter_vector(student)))
            for event in events:
                records.append(dict(event=event,**metrics))
    return records,trace,crossed


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    out=Path(args.output)
    if (out/"manifest.json").exists():
        raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    (out/"checkpoints").mkdir(exist_ok=True)
    (out/"source").mkdir(exist_ok=True)
    h=hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        shutil.copy2(path,out/"source"/path.name)
        h.update(path.name.encode());h.update(path.read_bytes())
    manifest=dict(config=vars(args),status="running",started_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        source_sha256=h.hexdigest(),python=platform.python_version(),torch=torch.__version__,
        numpy=np.__version__,device="cpu",threshold=.002,cap=1920)
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    started=time.perf_counter()
    fitting=[];identification=[];traces=[];models=[]
    for seed in range(args.seed_start,args.seed_start+args.seeds):
        for family in FAMILIES:
            base,diagnostics=load_or_train(out,seed,family,args.train_steps)
            models.append(dict(seed=seed,family=family,**diagnostics))
            c=coefficients(seed,family)
            mult=intervention("combined")
            test=sample(1024,370000+seed,1.25)
            shell=sample(3072,371000+seed,2.)
            shell=shell[shell.abs().amax(1)>1.5][:1024]
            with torch.no_grad():
                physical=acceleration(test,c*(1+mult),family)
                physical_shell=acceleration(shell,c*(1+mult),family)
            x=sample(128,60000+seed)
            with torch.no_grad():
                fitted=fit_coefficients(x,base(x),family)
                target=features(x,family)@(fitted*(1+mult))
                teacher_test=features(test,family)@(fitted*(1+mult))
            for warm in (False,True):
                records,trace,crossed=long_fit(base,x,target,seed,warm,test,teacher_test,physical,
                                              shell,physical_shell,cap=args.cap)
                method="warm" if warm else "fresh"
                for record in records:
                    fitting.append(dict(seed=seed,family=family,method=method,**record))
                for record in trace:
                    traces.append(dict(seed=seed,family=family,method=method,**record))
            for budget in (16,64,256):
                for design in ("narrow","broad","mixed"):
                    probes=query_design(budget,seed,design)
                    with torch.no_grad():
                        y=base(probes)
                    chosen,chosen_c,scores=select_family(probes,y,seed)
                    for access in ("selected","known"):
                        library=chosen if access=="selected" else family
                        fitted_c=chosen_c if access=="selected" else fit_coefficients(probes,y,family)
                        with torch.no_grad():
                            pred=features(test,library)@(fitted_c*(1+mult))
                            extrap=features(shell,library)@(fitted_c*(1+mult))
                        identification.append(dict(seed=seed,family=family,budget=budget,
                            design=design,access=access,selected_family=library,
                            correct_family=int(library==family),id_rmse=rms(pred-physical),
                            ood_rmse=rms(extrap-physical_shell),
                            condition_number=float(torch.linalg.cond(features(probes,library))),
                            source_query_rmse=rms(y-acceleration(probes,c,family)),
                            duffing_cv_mse=scores["duffing"],pendulum_cv_mse=scores["pendulum"]))
            # Preserve completed work after each independent source model.
            for name,rows in (("fitting.csv",fitting),("identification.csv",identification),("trace.csv",traces)):
                with (out/name).open("w",newline="",encoding="utf-8") as handle:
                    writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            (out/"models.json").write_text(json.dumps(models,indent=2))
            print(json.dumps(dict(seed=seed,family=family,fitting_rows=len(fitting),
                identification_rows=len(identification),elapsed_seconds=round(time.perf_counter()-started,1))),flush=True)
    manifest.update(status="complete",fitting_rows=len(fitting),identification_rows=len(identification),
        elapsed_seconds=time.perf_counter()-started,completed_utc=datetime.now(timezone.utc).isoformat())
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    p.add_argument("--seed-start",type=int,default=40)
    p.add_argument("--seeds",type=int,default=12)
    p.add_argument("--train-steps",type=int,default=2400)
    p.add_argument("--cap",type=int,default=1920)
    run(p.parse_args())
