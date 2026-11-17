"""Strengthen query distillation with free synthetic labels from its fitted library."""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path

import torch

from .core import (DynamicsNet, acceleration, coefficients, features, fit_coefficients,
                   intervention, parameter_vector, rollout, sample)
from .deployment import initialize, rms


def train_student(base,x,y,seed,warm,steps):
    student=initialize(base,seed,"warm_adam" if warm else "cold_adam","original")
    optimizer=torch.optim.Adam(student.parameters(),lr=.003)
    generator=torch.Generator().manual_seed(950000+seed)
    for step in range(steps):
        idx=torch.randint(len(x),(256,),generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss=(student(x[idx])-y[idx]).square().mean()
        loss.backward()
        optimizer.step()
        if step==int(steps*.75):
            for group in optimizer.param_groups:
                group["lr"]*=.2
    return student


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    source=Path(args.source)
    out=Path(args.output)
    if (out/"manifest.json").exists():
        raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    (out/"source").mkdir(exist_ok=True)
    h=hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        shutil.copy2(path,out/"source"/path.name)
        h.update(path.name.encode());h.update(path.read_bytes())
    manifest=dict(config=vars(args),status="running",started_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        source_sha256=h.hexdigest(),torch=torch.__version__,calibration_queries=128,synthetic_labels=4096,
        note="Synthetic labels use fitted coefficients only; original physical labels withheld.")
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    started=time.perf_counter()
    count=0
    with (out/"raw.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=None
        for seed in range(args.seed_start,args.seed_start+args.seeds):
            for family in ("duffing","pendulum"):
                saved=torch.load(source/"checkpoints"/f"{family}_{seed}.pt",weights_only=True)
                base=DynamicsNet(saved["width"]).double().eval()
                base.load_state_dict(saved["state_dict"])
                c=coefficients(seed,family)
                mult=intervention("combined")
                probes=sample(128,60000+seed)
                with torch.no_grad():
                    fitted=fit_coefficients(probes,base(probes),family)
                    x=sample(4096,960000+seed)
                    y=features(x,family)@(fitted*(1+mult))
                    test=sample(1024,170000+seed,1.25)
                    shell=sample(3072,171000+seed,2.)
                    shell=shell[shell.abs().amax(1)>1.5][:1024]
                    physical=acceleration(test,c*(1+mult),family)
                    physical_ood=acceleration(shell,c*(1+mult),family)
                    teacher=features(test,family)@(fitted*(1+mult))
                    initial=sample(24,172000+seed,.8)[:,:2]
                    tt=torch.arange(100,dtype=torch.float64)[:,None]
                    actions=.5*torch.sin(tt*.07+torch.arange(24,dtype=torch.float64)[None,:]*.4)
                    reference=rollout(lambda z:acceleration(z,c*(1+mult),family),initial,actions)
                for steps in args.steps:
                    for warm in (False,True):
                        t0=time.perf_counter()
                        student=train_student(base,x,y,seed,warm,steps)
                        seconds=time.perf_counter()-t0
                        with torch.no_grad():
                            prediction=student(test)
                            trajectory=rollout(student,initial,actions)
                            row=dict(seed=seed,family=family,method="warm_dense" if warm else "cold_dense",
                                steps=steps,id_rmse=rms(prediction-physical),ood_rmse=rms(student(shell)-physical_ood),
                                teacher_fit_rmse=rms(prediction-teacher),teacher_id_rmse=rms(teacher-physical),
                                rollout_rmse=rms(trajectory-reference),
                                rollout_nonfinite=int(not torch.isfinite(trajectory).all()),
                                edit_seconds=seconds,parameters=len(parameter_vector(student)),
                                base_queries=128,synthetic_labels=4096,gradient_points=steps*256)
                        if writer is None:
                            writer=csv.DictWriter(handle,fieldnames=list(row));writer.writeheader()
                        writer.writerow(row);count+=1
                handle.flush()
                print(json.dumps(dict(seed=seed,family=family,rows=count,
                                      elapsed_seconds=round(time.perf_counter()-started,1))),flush=True)
    manifest.update(status="complete",rows=count,elapsed_seconds=time.perf_counter()-started,
                    completed_utc=datetime.now(timezone.utc).isoformat())
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--seed-start",type=int,default=20)
    p.add_argument("--seeds",type=int,default=12)
    p.add_argument("--steps",type=int,nargs="+",default=[600,2400])
    run(p.parse_args())
