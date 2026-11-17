"""Analyze adaptation access contracts using seed-clustered paired comparisons."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np

from .analyze import estimate

LABELS=dict(source="Source only",warm_full="Warm full",fresh_full="Fresh full",
    learned_head="Learned head",linear_residual="Linear residual",cubic_residual="Cubic residual",
    rbf_residual="RBF residual",ard_rbf_residual="ARD RBF residual",neural_residual="Neural residual")
COLORS=dict(zip(LABELS,["#999999","#0072B2","#222222","#E69F00","#56B4E9","#009E73","#D55E00","#CC79A7","#882255"]))


def read(path):
    rows=list(csv.DictReader(path.open(encoding="utf-8")))
    for r in rows:
        for k,v in r.items():
            if k not in ("family","shift","method"):
                r[k]=float(v) if v else np.nan
    return rows


def subset(rows,**criteria):
    return [r for r in rows if all(r[k]==v for k,v in criteria.items())]


def cluster(rows,metric):
    groups=defaultdict(list)
    for r in rows:
        groups[r["seed"]].append(r[metric])
    return estimate([np.mean(v) for _,v in sorted(groups.items())])


def contrast(rows,first,second,metric):
    groups=defaultdict(dict)
    for r in rows:
        groups[(r["seed"],r["family"],r["shift"],r["budget"])][r["method"]]=r
    by_seed=defaultdict(list)
    for key,group in groups.items():
        if first in group and second in group:
            by_seed[key[0]].append(group[first][metric]-group[second][metric])
    return estimate([np.mean(v) for _,v in sorted(by_seed.items())])


def fmt(e):
    if e["mean"] is None:
        return "NONFINITE"
    return f"{e['mean']:.5g} [{e['low']:.5g}, {e['high']:.5g}]"


def analyze(folder):
    manifest=json.loads((folder/"manifest.json").read_text())
    if manifest["status"]!="complete":
        raise RuntimeError("Incomplete run")
    h=hashlib.sha256()
    for path in sorted((folder/"source").glob("*.py")):
        h.update(path.name.encode());h.update(path.read_bytes())
    if h.hexdigest()!=manifest["source_sha256"]:
        raise AssertionError("Source snapshot mismatch")
    rows=read(folder/"raw.csv");c=manifest["config"]
    expected=c["seeds"]*len(c["families"])*2*len(c["budgets"])*len(c["methods"])
    if len(rows)!=expected:
        raise AssertionError((len(rows),expected))
    keys=[tuple(r[k] for k in ("seed","family","shift","budget","method")) for r in rows]
    if len(keys)!=len(set(keys)):
        raise AssertionError("Duplicate result rows")
    primary=subset(rows,shift="nonlinear",budget=64)
    summary=dict(rows=len(rows),source_models=c["seeds"]*len(c["families"]),
        nonfinite_rollout_rows=int(sum(r["rollout_nonfinite"] for r in rows)),source_hash="passed",
        nonfinite_metrics={m:int(sum(not np.isfinite(r[m]) for r in rows))
                           for m in ("id_rmse","ood_rmse","rollout_rmse")},
        contrasts={},strata=[],family_strata=[])
    for first in ("rbf_residual","ard_rbf_residual","neural_residual","linear_residual"):
        for second in ("warm_full","learned_head","fresh_full"):
            for metric in ("id_rmse","ood_rmse","rollout_rmse"):
                summary["contrasts"][f"{first}_minus_{second}_{metric}"]=contrast(primary,first,second,metric)
    metrics=("id_rmse","ood_rmse","rollout_rmse","calibration_noisy_rmse","adaptation_seconds","extra_scalars","gradient_points")
    for shift in ("parameter","nonlinear"):
        for budget in c["budgets"]:
            for method in c["methods"]:
                sub=subset(rows,shift=shift,budget=budget,method=method)
                summary["strata"].append(dict(shift=shift,budget=budget,method=method,
                                             metrics={m:cluster(sub,m) for m in metrics}))
    for family in c["families"]:
        for method in c["methods"]:
            sub=subset(primary,family=family,method=method)
            summary["family_strata"].append(dict(family=family,method=method,id_rmse=cluster(sub,"id_rmse")))
    lines=["# Cycle 4: predictions versus internal features","",
        f"{summary['source_models']} source models; {len(rows)} condition-method rows; "
        f"{manifest['elapsed_seconds']/60:.2f} minutes on CPU.","",
        "This is adaptation from noisy changed-environment observations, not specification-only "
        "editing. No physics library or intervention formula is supplied to the adapters. Source "
        "predictions are callable during adaptation and deployment; residual wrappers have a "
        "different architecture/storage contract from in-place edits.","",
        "## Primary setting: added mechanisms, 64 observations","",
        "| Method | Physical RMSE | Extrapolation RMSE | Rollout RMSE | Adaptation seconds |",
        "|---|---:|---:|---:|---:|"]
    for method in c["methods"]:
        sub=subset(primary,method=method)
        lines.append("| "+LABELS[method]+" | "+" | ".join(fmt(cluster(sub,m)) for m in
            ("id_rmse","ood_rmse","rollout_rmse","adaptation_seconds"))+" |")
    lines += ["","## Paired access contrasts","",
        "Negative differences favor the prediction-only residual method. Intervals bootstrap "
        "independent seeds, averaging all four systems within a seed; no multiplicity adjustment.","",
        "| Contrast | Physical RMSE difference | Rollout RMSE difference |",
        "|---|---:|---:|"]
    for first,second in (("rbf_residual","warm_full"),("rbf_residual","learned_head"),
                         ("ard_rbf_residual","warm_full"),("neural_residual","warm_full")):
        lines.append("| "+LABELS[first]+" minus "+LABELS[second]+" | "+" | ".join(
            fmt(summary["contrasts"][f"{first}_minus_{second}_{m}"]) for m in ("id_rmse","rollout_rmse"))+" |")
    lines += ["","## System-specific physical error, nonlinear shift, 64 observations","",
              "| System | Warm full | Learned head | RBF | ARD RBF | Neural residual |",
              "|---|---:|---:|---:|---:|---:|"]
    for family in c["families"]:
        lines.append("| "+family+" | "+" | ".join(fmt(cluster(subset(primary,family=family,method=m),"id_rmse"))
            for m in ("warm_full","learned_head","rbf_residual","ard_rbf_residual","neural_residual"))+" |")
    lines += ["","## Parameter-shift control, 16 observations","",
              "| Method | Physical RMSE |","|---|---:|"]
    for method in c["methods"]:
        lines.append("| "+LABELS[method]+" | "+fmt(cluster(subset(rows,shift="parameter",budget=16,method=method),"id_rmse"))+" |")
    lines += ["","## Integrity and limits","",f"Nonfinite rollout rows: {summary['nonfinite_rollout_rows']}.",
        "No LLM is tested. Generic polynomial and kernel smoothness assumptions replace the earlier "
        "named mechanism library; this is not assumption-free adaptation. Neural early stopping and "
        "kernel hyperparameters use calibration observations only. Inference wrapper costs differ; "
        "timings include validation/refitting work but are descriptive single-machine measurements."]
    (folder/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
    (folder/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    figures(folder,rows,c)
    print("\n".join(lines))


def figures(folder,rows,c):
    out=folder/"figures";out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,shift in zip(axes,("parameter","nonlinear")):
        for method in c["methods"]:
            ys=[cluster(subset(rows,shift=shift,budget=b,method=method),"id_rmse")["mean"] for b in c["budgets"]]
            ax.plot(c["budgets"],ys,"o-",label=LABELS[method],color=COLORS[method],ms=4)
        ax.set(xscale="log",yscale="log",xlabel="Changed-environment observations",ylabel="Physical RMSE",title=shift.capitalize()+" shift")
        ax.set_xticks(c["budgets"],[str(b) for b in c["budgets"]])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=3,frameon=False)
    for ext in ("png","pdf"):
        fig.savefig(out/f"access_comparison.{ext}",dpi=180)
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True,type=Path)
    analyze(p.parse_args().input)
