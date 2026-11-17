"""Prepare public raw-weight prompts and separately stored private evaluation metadata."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .core import coefficients,features,sample
from .wsp_common import TASKS,forward,gauge,make_prompt,quantize


def seal(out):
    path=out/"manifest.json"
    manifest=json.loads(path.read_text())
    hashes={}
    for case in manifest["cases"]:
        hashes[case]={}
        for access in ("public","private"):
            value=json.loads((out/access/f"{case}.json").read_text(encoding="utf-8"))
            canonical=json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()
            hashes[case][access]=hashlib.sha256(canonical).hexdigest()
    manifest["canonical_json_sha256"]=hashes
    path.write_text(json.dumps(manifest,indent=2))


def prepare(out,seeds,capacities=("dense","spare"),representations=("original","transformed"),tasks=TASKS):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    out.mkdir(parents=True,exist_ok=True)
    if (out/"manifest.json").exists():
        raise FileExistsError(out)
    (out/"public").mkdir();(out/"private").mkdir()
    cases=[]
    for seed in seeds:
        torch.manual_seed(810000+seed)
        network=nn.Sequential(nn.Linear(3,12),nn.ReLU(),nn.Linear(12,1)).double()
        x=sample(4096,820000+seed,1.5)
        c=coefficients(seed,"pendulum")
        y=features(x,"pendulum")@c
        optimizer=torch.optim.Adam(network.parameters(),lr=.005)
        generator=torch.Generator().manual_seed(830000+seed)
        for step in range(6000):
            idx=torch.randint(len(x),(256,),generator=generator)
            optimizer.zero_grad(set_to_none=True)
            loss=(network(x[idx]).squeeze(-1)-y[idx]).square().mean()
            loss.backward();optimizer.step()
            if step==4500:
                for group in optimizer.param_groups:group["lr"]*=.2
        weights=quantize(dict(W1=network[0].weight.detach().numpy(),b1=network[0].bias.detach().numpy(),
            W2=network[2].weight.detach().numpy()[0],b2=network[2].bias.detach().numpy()))
        for capacity in capacities:
            w=json.loads(json.dumps(weights))
            if capacity=="spare":
                w["W1"] += [[0.,0.,0.],[0.,0.,0.]];w["b1"] += [0.,0.];w["W2"] += [0.,0.]
            for representation in representations:
                represented=gauge(w,seed,representation=="transformed")
                for task in tasks:
                    label=f"{seed}/{capacity}/{representation}/{task}"
                    case=hashlib.sha256(label.encode()).hexdigest()[:16]
                    prompt=make_prompt(represented,task)
                    public=dict(case_id=case,prompt=prompt,weights=represented)
                    private=dict(seed=seed,capacity=capacity,representation=representation,task=task,
                        coefficients=c.tolist(),true_c=float(c[1]),source_training_steps=6000)
                    (out/"public"/f"{case}.json").write_text(json.dumps(public,indent=2),encoding="utf-8")
                    (out/"private"/f"{case}.json").write_text(json.dumps(private,indent=2),encoding="utf-8")
                    cases.append(case)
    manifest=dict(cases=cases,source_seeds=seeds,case_count=len(cases),
        description="API runner may read public files only; evaluator reads private metadata after calls")
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    seal(out)
    print(json.dumps(dict(fixtures=len(cases),sources=len(seeds),output=str(out))))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--seeds",nargs="+",type=int,default=[90,91])
    p.add_argument("--capacities",nargs="+",choices=["dense","spare"],default=["dense","spare"])
    p.add_argument("--representations",nargs="+",choices=["original","transformed"],default=["original","transformed"])
    p.add_argument("--tasks",nargs="+",choices=TASKS,default=list(TASKS))
    p.add_argument("--seal-only",action="store_true")
    a=p.parse_args()
    if a.seal_only:seal(a.output)
    else:prepare(a.output,a.seeds,a.capacities,a.representations,a.tasks)
