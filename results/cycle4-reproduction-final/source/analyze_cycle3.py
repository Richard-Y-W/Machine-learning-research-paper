"""Analyze matched fitting and hidden-family identification without dropping failures."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

from .analyze import estimate

STRINGS={"family","method","event","design","access","selected_family"}
COLORS={"narrow":"#D55E00","broad":"#0072B2","mixed":"#009E73"}


def read(path):
    rows=list(csv.DictReader(path.open(encoding="utf-8")))
    for r in rows:
        for k,v in r.items():
            if k not in STRINGS:
                r[k]=float(v) if v else np.nan
    return rows


def select(rows,**criteria):
    return [r for r in rows if all(r[k]==v for k,v in criteria.items())]


def cluster(rows,metric):
    groups=defaultdict(list)
    for r in rows:
        groups[r["seed"]].append(r[metric])
    return estimate([np.mean(v) for _,v in sorted(groups.items())])


def paired(rows,first,second,metric,keys=("seed","family")):
    def lookup(criteria):
        return {tuple(r[k] for k in keys):r for r in select(rows,**criteria)}
    a,b=lookup(first),lookup(second)
    shared=sorted(set(a)&set(b))
    groups=defaultdict(list)
    for key in shared:
        groups[key[0]].append(a[key][metric]-b[key][metric])
    result=estimate([np.mean(v) for _,v in sorted(groups.items())])
    result["paired_model_conditions"]=len(shared)
    return result


def fmt(e):
    if e["mean"] is None:
        return "unavailable/nonfinite"
    return f"{e['mean']:.5g} [{e['low']:.5g}, {e['high']:.5g}]"


def joint_success(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[r["seed"]].append(bool(r["correct_family"]))
    successes=sum(all(values) for values in groups.values())
    n=len(groups)
    interval=binomtest(successes,n).proportion_ci(confidence_level=.95,method="exact")
    return dict(successful_seeds=successes,seeds=n,low=float(interval.low),high=float(interval.high),
                event="Both candidate-system models in a seed are identified correctly")


def analyze(folder):
    manifest=json.loads((folder/"manifest.json").read_text())
    if manifest["status"]!="complete":
        raise RuntimeError("Run is incomplete")
    h=hashlib.sha256()
    for path in sorted((folder/"source").glob("*.py")):
        h.update(path.name.encode());h.update(path.read_bytes())
    if h.hexdigest()!=manifest["source_sha256"]:
        raise AssertionError("Source snapshot hash mismatch")
    fit=read(folder/"fitting.csv")
    ident=read(folder/"identification.csv")
    trace=read(folder/"trace.csv")
    nmodels=manifest["config"]["seeds"]*2
    if len(ident)!=nmodels*18:
        raise AssertionError("Missing identification conditions")
    fixed=select(fit,event="fixed")
    if len(fixed)!=nmodels*2*3:
        raise AssertionError("Missing fixed-budget fitting checkpoints")
    thresholds=select(fit,event="threshold")
    fit_contrast=paired(thresholds,dict(method="warm"),dict(method="fresh"),"id_rmse")
    fixed_contrast=paired(select(fixed,iteration_budget=1920),dict(method="warm"),dict(method="fresh"),"id_rmse")
    query_contrast=paired(select(ident,access="selected"),dict(budget=64,design="mixed"),
                          dict(budget=256,design="narrow"),"id_rmse")
    summary=dict(source_hash="passed",source_models=nmodels,threshold=.002,
        threshold_attainment={m:len(select(thresholds,method=m)) for m in ("warm","fresh")},
        matched_fit_warm_minus_fresh=fit_contrast,final_warm_minus_fresh=fixed_contrast,
        mixed64_minus_narrow256=query_contrast,
        nonfinite_fitting_metrics={m:sum(not np.isfinite(r[m]) for r in fit)
                                  for m in ("id_rmse","teacher_fit_rmse","calibration_fit_rmse","ood_rmse")},
        fitting_summary=[],identification_summary=[])
    matched_scope=("All evaluated model conditions enter this contrast; it does not uniquely "
        "separate optimization path from initialization-dependent solution bias."
        if fit_contrast["paired_model_conditions"]==nmodels else
        "This contrast is conditional on the jointly attained subset and is not an unconditional estimate.")
    lines=["# Cycle 3 findings","",f"{nmodels} new source models; {len(fit)} fitting checkpoints; "
           f"{len(ident)} identification measurements; {manifest['elapsed_seconds']/60:.2f} minutes on CPU.","",
           "Intervals are exploratory 95% seed-cluster bootstrap intervals; families are averaged within "
           "each seed. Threshold crossings are selected using calibration error only.","",
           "## 1. Longer optimization and similar calibration fit","",
           f"Calibration threshold <=0.002 attained by {summary['threshold_attainment']['warm']}/{nmodels} warm "
           f"and {summary['threshold_attainment']['fresh']}/{nmodels} fresh students.","",
           "At first threshold crossing, warm minus fresh physical RMSE on the jointly attained subset: "
           +fmt(fit_contrast)+f"; {fit_contrast['paired_model_conditions']}/{nmodels} paired model conditions. "
           +matched_scope,"",
           "At the full fixed 1,920-iteration budget, warm minus fresh physical RMSE: "+fmt(fixed_contrast)+".","",
           "| Initialization | Checkpoint | Calibration fit | Held-out teacher fit | Physical RMSE | Iterations reported |",
           "|---|---|---:|---:|---:|---:|"]
    for event,budget in (("fixed",120),("fixed",480),("fixed",1920),("threshold",None)):
        for method in ("fresh","warm"):
            sub=select(fit,event=event,method=method)
            if budget is not None:
                sub=select(sub,iteration_budget=budget)
            metrics={m:cluster(sub,m) for m in ("calibration_fit_rmse","teacher_fit_rmse","id_rmse","accepted_iterations")}
            label=str(budget) if budget is not None else "first threshold"
            summary["fitting_summary"].append(dict(event=event,budget=budget,method=method,metrics=metrics))
            lines.append(f"| {method} | {label} | "+" | ".join(fmt(metrics[m]) for m in
                         ("calibration_fit_rmse","teacher_fit_rmse","id_rmse","accepted_iterations"))+" |")
    lines += ["","## 2. Family uncertainty and query placement","",
        "Primary query-design contrast, 64 mixed minus 256 narrow queries, selected family: "+fmt(query_contrast)+
        ". Negative differences favor fewer, broader queries. All queries evaluate the source model, not a changed environment.","",
        "| Budget | Design | Selected-family accuracy | Selected physical RMSE | Known-family physical RMSE |",
        "|---:|---|---:|---:|---:|"]
    for budget in (16,64,256):
        for design in ("narrow","broad","mixed"):
            selected=select(ident,budget=budget,design=design,access="selected")
            known=select(ident,budget=budget,design=design,access="known")
            accuracy=cluster(selected,"correct_family")
            error=cluster(selected,"id_rmse")
            oracle_error=cluster(known,"id_rmse")
            joint=joint_success(selected)
            summary["identification_summary"].append(dict(budget=budget,design=design,
                accuracy=accuracy,selected_id_rmse=error,known_id_rmse=oracle_error,
                joint_seed_success=joint,
                selected_ood_rmse=cluster(selected,"ood_rmse"),
                known_ood_rmse=cluster(known,"ood_rmse"),
                mean_condition_number=cluster(selected,"condition_number")))
            correct=int(sum(r["correct_family"] for r in selected))
            lines.append(f"| {budget} | {design} | {correct}/{len(selected)} ({accuracy['mean']:.1%}) | {fmt(error)} | {fmt(oracle_error)} |")
    broad_joint=joint_success(select(ident,budget=64,design="broad",access="selected"))
    lines += ["","Observed family-accuracy counts are not population guarantees. For the joint event "
        "that both models in a seed are correctly identified at 64 broad queries, "
        f"{broad_joint['successful_seeds']}/{broad_joint['seeds']} seeds succeed; the exact binomial "
        f"95% interval over independent seed events is [{broad_joint['low']:.3f}, {broad_joint['high']:.3f}]. "
        "This supplementary interval avoids interpreting an all-success percentile bootstrap as certainty.",
        "","## Interpretation limits","",
        "The candidate library contains only two supplied families, one of which is correct. This does not "
        "demonstrate arbitrary mechanism discovery. Broader source-model queries are allowed here; their "
        "availability is an application assumption. No semantic weight-reading or LLM claim is tested.","",
        "A residual initialization gap at similar calibration fit can reflect an implicit prior, optimization "
        "path, or solution selection. It does not uniquely identify one mechanism."]
    (folder/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
    (folder/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    figures(folder,fit,ident,trace)
    print("\n".join(lines))


def figures(folder,fit,ident,trace):
    out=folder/"figures";out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    for method,color in (("fresh","#D55E00"),("warm","#0072B2")):
        budgets=sorted({r["iteration_budget"] for r in trace})
        ys=[cluster(select(trace,method=method,iteration_budget=b),"calibration_fit_rmse")["mean"] for b in budgets]
        axes[0].plot(budgets,ys,label=method,color=color)
        xs=[120,480,1920]
        ys=[cluster(select(fit,event="fixed",method=method,iteration_budget=b),"teacher_fit_rmse")["mean"] for b in xs]
        axes[1].plot(xs,ys,"o-",label=method,color=color)
    axes[0].axhline(.002,color="#777777",ls="--",label="Fit threshold")
    for ax,title in zip(axes,["Calibration teacher fit","Held-out teacher fit"]):
        ax.set(xscale="log",yscale="log",xlabel="Iteration budget",ylabel="RMSE",title=title)
        ax.grid(alpha=.2);ax.legend(frameon=False)
    for ext in ("png","pdf"):
        fig.savefig(out/f"long_fitting.{ext}",dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    for design in ("narrow","broad","mixed"):
        xs=[16,64,256]
        ys=[cluster(select(ident,budget=b,design=design,access="selected"),"id_rmse")["mean"] for b in xs]
        acc=[cluster(select(ident,budget=b,design=design,access="selected"),"correct_family")["mean"] for b in xs]
        axes[0].plot(xs,ys,"o-",label=design,color=COLORS[design])
        axes[1].plot(xs,acc,"o-",label=design,color=COLORS[design])
    axes[0].set(xscale="log",yscale="log",xlabel="Source-model queries",ylabel="Physical RMSE",title="Query placement versus count")
    axes[1].set(xscale="log",xlabel="Source-model queries",ylabel="Correct family fraction",ylim=(-.03,1.03),title="Two-family identification")
    for ax in axes:
        ax.grid(alpha=.2);ax.legend(frameon=False)
    for ext in ("png","pdf"):
        fig.savefig(out/f"query_placement.{ext}",dpi=180)
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True,type=Path)
    analyze(p.parse_args().input)
