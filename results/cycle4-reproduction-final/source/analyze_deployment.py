"""Seed-clustered factorial analysis for the same-architecture follow-up."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analyze import estimate
from .deployment import METHODS

LABELS=dict(warm_adam="Warm Adam",cold_adam="Fresh Adam",warm_lbfgs="Warm L-BFGS",
            cold_lbfgs="Fresh L-BFGS",euclidean_jacobian="Euclidean Jacobian",
            column_jacobian="Column Jacobian",relative_jacobian="Relative Jacobian")
COLORS=dict(zip(METHODS,["#56B4E9","#999999","#0072B2","#000000","#D55E00","#CC79A7","#009E73"]))
METRICS=["id_rmse","ood_rmse","teacher_fit_rmse","calibration_fit_rmse","rollout_rmse",
         "gauge_disagreement","edit_seconds","student_batch256_ms","teacher_batch256_ms",
         "teacher_id_rmse","forward_points","gradient_points","jacobian_output_points"]
STRINGS={"family","teacher","schedule","gauge","method"}


def read_rows(path):
    rows=list(csv.DictReader(path.open(encoding="utf-8")))
    for row in rows:
        for k,v in row.items():
            if k not in STRINGS:
                row[k]=float(v) if v else np.nan
    return rows


def subset(rows,**kwargs):
    return [r for r in rows if all(r[k]==v for k,v in kwargs.items())]


def cluster(rows,metric):
    seeds=defaultdict(list)
    for row in rows:
        seeds[row["seed"]].append(row[metric])
    return estimate([np.mean(v) for _,v in sorted(seeds.items())])


def difference(rows,field,first,second,metric):
    if not rows:
        return estimate([])
    keys=[k for k in ("seed","family","budget","teacher","schedule","gauge","method","steps")
          if k in rows[0] and k!=field]
    paired=defaultdict(dict)
    for r in rows:
        paired[tuple(r[k] for k in keys)][r[field]]=r
    seeds=defaultdict(list)
    for key,group in paired.items():
        if first in group and second in group:
            seeds[key[0]].append(group[first][metric]-group[second][metric])
    return estimate([np.mean(v) for _,v in sorted(seeds.items())])


def fmt(e,digits=4):
    if e["mean"] is None:
        return "NONFINITE"
    return f"{e['mean']:.{digits}g} [{e['low']:.{digits}g}, {e['high']:.{digits}g}]"


def analyze(folder):
    manifest=json.loads((folder/"manifest.json").read_text())
    if manifest["status"]!="complete":
        raise RuntimeError("Incomplete run")
    if "synthetic_labels" in manifest:
        return analyze_dense(folder,manifest)
    rows=read_rows(folder/"raw.csv")
    c=manifest["config"]
    expected=c["seeds"]
    for key in ("families","budgets","teachers","schedules","gauges","methods"):
        expected*=len(c[key])
    if len(rows)!=expected:
        raise AssertionError((len(rows),expected))
    keys=[tuple(r[k] for k in ("seed","family","budget","teacher","schedule","gauge","method")) for r in rows]
    if len(set(keys))!=len(keys):
        raise AssertionError("Duplicate condition rows")
    if any(r["parameters"]!=3*c["width"]+c["width"]+c["width"]**2+c["width"]+c["width"]+1 for r in rows):
        raise AssertionError("Mismatched deployment architectures")
    focal=subset(rows,budget=max(c["budgets"]),schedule="long",gauge="original")
    primary=difference(subset(focal,teacher="symbolic"),"method","warm_lbfgs","cold_lbfgs","id_rmse")
    contrasts={"warm_minus_fresh_lbfgs_symbolic_physical":primary}
    for method in c["methods"]:
        for metric in ("id_rmse","teacher_fit_rmse","rollout_rmse"):
            contrasts[f"symbolic_minus_residual_{method}_{metric}"]=difference(
                subset(focal,method=method),"teacher","symbolic","residual",metric)
    summary=dict(rows=len(rows),expected_rows=expected,seeds=c["seeds"],
                 nonfinite_rollout_rows=int(sum(r["rollout_nonfinite"] for r in rows)),
                 nonfinite_metric_cells={m:int(sum(not np.isfinite(r[m]) for r in rows)) for m in METRICS},
                 primary_contrast=primary,contrasts=contrasts,strata=[])
    for teacher in c["teachers"]:
        for budget in c["budgets"]:
            for schedule in c["schedules"]:
                for gauge in c["gauges"]:
                    for method in c["methods"]:
                        sub=subset(rows,teacher=teacher,budget=budget,schedule=schedule,gauge=gauge,method=method)
                        summary["strata"].append(dict(teacher=teacher,budget=budget,schedule=schedule,gauge=gauge,method=method,
                                                     metrics={m:cluster(sub,m) for m in METRICS}))
    (folder/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
    lines=["# Same-architecture follow-up", "",f"{len(rows):,} rows; {c['seeds']} independent "
           f"seed clusters; {manifest['elapsed_seconds']/60:.2f} CPU minutes.","",
           "All deployed students have the same architecture. Warm starts use the pretrained weights; "
           "fresh starts do not. All receive the same calibration queries and physics library. "
           "The two schedules are not matched FLOP budgets.","",
           "## Predeclared primary contrast", "",
           "Warm minus fresh L-BFGS, symbolic targets, 128 queries, long schedule, original representation: "
           +fmt(primary)+" acceleration RMSE. Negative values favor initialization from pretrained weights.","",
           "Exploratory 95% percentile intervals bootstrap independent seed clusters; average families "
           "within each seed. No multiplicity adjustment."]
    for teacher in c["teachers"]:
        lines += ["",f"## {teacher.capitalize()} targets, {max(c['budgets'])} queries, long schedule, original representation","",
                  "| Method | Physical RMSE | Teacher-fit RMSE | Rollout RMSE | Construction seconds |",
                  "|---|---:|---:|---:|---:|"]
        for method in c["methods"]:
            sub=subset(focal,teacher=teacher,method=method)
            lines.append("| "+LABELS[method]+" | "+" | ".join(fmt(cluster(sub,m)) for m in
                ("id_rmse","teacher_fit_rmse","rollout_rmse","edit_seconds"))+" |")
    lines += ["","## Target-objective contrasts","",
              "Symbolic minus residual targets at 128 queries, long schedule, original representation. "
              "Negative physical-error contrasts favor physics reconstruction; teacher fitting measures "
              "a different objective and must be reported separately.","",
              "| Solver | Physical RMSE difference | Rollout RMSE difference |","|---|---:|---:|"]
    for method in c["methods"]:
        lines.append("| "+LABELS[method]+" | "+" | ".join(fmt(contrasts[f"symbolic_minus_residual_{method}_{m}"])
                      for m in ("id_rmse","rollout_rmse"))+" |")
    lines += ["","## Integrity","",f"Nonfinite rollout rows: {summary['nonfinite_rollout_rows']}.",
              "The complete JSON includes all budgets, schedules, representations, and nonfinite counts. "
              "This study does not evaluate semantic weight reading or language models."]
    (folder/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    figures(folder,rows,c)
    print("\n".join(lines))


def analyze_dense(folder,manifest):
    rows=read_rows(folder/"raw.csv")
    c=manifest["config"]
    expected=c["seeds"]*2*2*len(c["steps"])
    if len(rows)!=expected:
        raise AssertionError((len(rows),expected))
    summary=dict(rows=len(rows),seeds=c["seeds"],strata=[],contrasts={},
                 nonfinite_rollout_rows=int(sum(r["rollout_nonfinite"] for r in rows)))
    lines=["# Dense symbolic distillation", "",f"{len(rows)} measurements; {c['seeds']} seed clusters; "
           f"{manifest['elapsed_seconds']/60:.2f} CPU minutes.","",
           "128 base queries determine a symbolic teacher; 4,096 synthetic labels require no "
           "additional base queries. Each deployed student has 1,217 parameters.","",
           "| Initialization | Steps | Physical RMSE | Teacher-fit RMSE | Seconds |",
           "|---|---:|---:|---:|---:|"]
    for steps in c["steps"]:
        for method,label in (("cold_dense","Fresh"),("warm_dense","Warm")):
            sub=subset(rows,steps=steps,method=method)
            metrics={m:cluster(sub,m) for m in ("id_rmse","ood_rmse","teacher_fit_rmse","rollout_rmse","edit_seconds")}
            summary["strata"].append(dict(steps=steps,method=method,metrics=metrics))
            lines.append(f"| {label} | {steps} | "+" | ".join(fmt(metrics[m]) for m in
                         ("id_rmse","teacher_fit_rmse","edit_seconds"))+" |")
        summary["contrasts"][str(steps)]=difference(subset(rows,steps=steps),"method","warm_dense","cold_dense","id_rmse")
    (folder/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
    (folder/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines))


def figures(folder,rows,c):
    out=folder/"figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    budget=max(c["budgets"])
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,teacher in zip(axes,c["teachers"]):
        for s,schedule in enumerate(c["schedules"]):
            means=[]
            low=[]
            high=[]
            for method in c["methods"]:
                e=cluster(subset(rows,budget=budget,teacher=teacher,schedule=schedule,gauge="original",method=method),"id_rmse")
                means.append(e["mean"]);low.append(e["mean"]-e["low"]);high.append(e["high"]-e["mean"])
            x=np.arange(len(means))+(s-.5)*.3
            ax.errorbar(x,means,yerr=[low,high],fmt="o",capsize=3,label=schedule,color=["#888888","#0072B2"][s])
        teacher_ref=cluster(subset(rows,budget=budget,teacher=teacher,gauge="original"),"teacher_id_rmse")["mean"]
        ax.axhline(teacher_ref,color="#009E73",ls="--",label="Teacher reference")
        ax.set_xticks(np.arange(len(c["methods"])),[LABELS[m].replace(" ","\n") for m in c["methods"]],fontsize=8)
        ax.set(yscale="log",ylabel="Physical acceleration RMSE",title=teacher.capitalize()+" targets")
        ax.grid(axis="y",alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=3,frameon=False)
    for ext in ("png","pdf"):
        fig.savefig(out/f"factorial.{ext}",dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,teacher in zip(axes,c["teachers"]):
        for method in c["methods"]:
            xs=[];ys=[]
            for schedule in c["schedules"]:
                sub=subset(rows,budget=budget,teacher=teacher,schedule=schedule,gauge="original",method=method)
                xs.append(cluster(sub,"edit_seconds")["mean"])
                ys.append(cluster(sub,"id_rmse")["mean"])
            ax.plot(xs,ys,"o-",label=LABELS[method],color=COLORS[method],ms=5)
        ax.set(xscale="log",yscale="log",xlabel="Mean construction seconds",ylabel="Physical acceleration RMSE",
               title=teacher.capitalize()+" targets")
        ax.grid(alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=4,frameon=False)
    for ext in ("png","pdf"):
        fig.savefig(out/f"cost_accuracy.{ext}",dpi=180)
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True,type=Path)
    analyze(p.parse_args().input)
