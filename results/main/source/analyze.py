"""Seed-clustered summaries, paired intervals, and standalone research figures."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = ["unedited", "query_residual", "query_symbolic", "head_ridge",
           "jacobian_ridge", "normalized_jacobian", "finetune_adam", "oracle_residual"]
LABELS = dict(unedited="Unedited", query_residual="Query + residual",
              query_symbolic="Query symbolic", head_ridge="Head ridge",
              jacobian_ridge="Jacobian ridge", normalized_jacobian="Normalized Jacobian",
              finetune_adam="Adam fine-tuning", oracle_residual="Oracle residual")
COLORS = dict(zip(METHODS, ["#777777", "#0072B2", "#009E73", "#E69F00",
                           "#D55E00", "#CC79A7", "#56B4E9", "#222222"]))
METRICS = ["id_rmse", "ood_rmse", "edit_nrmse", "rollout_rmse", "collateral_rmse",
           "gauge_disagreement", "edit_seconds", "coefficient_rmse"]


def read_rows(path):
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    for row in rows:
        for key in row:
            if key not in ("family", "intervention", "gauge", "method"):
                row[key] = float(row[key]) if row[key] != "" else np.nan
    return rows


def estimate(values, seed=1729):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        return {"mean": None, "low": None, "high": None, "n_seeds": len(values),
                "nonfinite_seed_values": int((~np.isfinite(values)).sum())}
    rng = np.random.default_rng(seed)
    boots = values[rng.integers(0, len(values), (5000, len(values)))].mean(1)
    return dict(mean=float(values.mean()), low=float(np.quantile(boots, .025)),
                high=float(np.quantile(boots, .975)), n_seeds=len(values),
                nonfinite_seed_values=0)


def clustered(rows, metric):
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[row["seed"]].append(row[metric])
    return estimate([np.mean(by_seed[s]) for s in sorted(by_seed)])


def paired(rows, method, baseline, metric):
    keyed = defaultdict(dict)
    for row in rows:
        key = (row["seed"], row["family"], row["budget"], row["intervention"], row["gauge"])
        keyed[key][row["method"]] = row
    seed_differences = defaultdict(list)
    wins = []
    for key, group in keyed.items():
        if method in group and baseline in group:
            d = group[method][metric] - group[baseline][metric]
            seed_differences[key[0]].append(d)
            wins.append(d < 0)
    result = estimate([np.mean(v) for _, v in sorted(seed_differences.items())])
    result["condition_win_fraction"] = float(np.mean(wins)) if wins else None
    return result


def analyze(folder):
    rows = read_rows(folder / "raw.csv")
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise RuntimeError("Run has not completed")
    config = manifest["config"]
    expected = config["seeds"] * len(config["families"]) * len(config["budgets"]) * len(
        config["interventions"]) * len(config["gauges"]) * len(config["methods"])
    if len(rows) != expected:
        raise AssertionError((len(rows), expected))
    keys = [(r["seed"],r["family"],r["budget"],r["intervention"],r["gauge"],r["method"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise AssertionError("Duplicate result rows")
    summaries = []
    for family in config["families"]:
        for budget in config["budgets"]:
            for gauge in config["gauges"]:
                for method in config["methods"]:
                    subset = [r for r in rows if r["family"] == family and r["budget"] == budget
                              and r["gauge"] == gauge and r["method"] == method]
                    summaries.append(dict(family=family, budget=budget, gauge=gauge, method=method,
                                          metrics={m:clustered(subset,m) for m in METRICS}))
    max_budget = max(config["budgets"])
    matched = [r for r in rows if r["family"] != "duffing_mismatch" and r["budget"] == max_budget
               and r["gauge"] == "original"]
    paired_results = {}
    for baseline in ("query_residual", "query_symbolic"):
        for method in ("head_ridge", "jacobian_ridge", "normalized_jacobian", "finetune_adam"):
            paired_results[f"{method}_minus_{baseline}"] = {
                metric: paired(matched,method,baseline,metric)
                for metric in ("id_rmse", "ood_rmse", "edit_nrmse", "rollout_rmse")}
    report = dict(rows=len(rows), expected_rows=expected, summaries=summaries,
                  paired_matched_original_max_budget=paired_results,
                  nonfinite_rollout_rows=sum(r["rollout_nonfinite"] for r in rows),
                  nonfinite_metric_cells={m:sum(not np.isfinite(r[m]) for r in rows) for m in METRICS},
                  symmetry_max_error=max(r["symmetry_max_error"] for r in rows))
    (folder / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    lines = ["# Pilot results", "", f"{len(rows)} rows; {config['seeds']} independent seeds; "
             f"{manifest['elapsed_seconds']:.1f} seconds. Maximum symmetry error: "
             f"{report['symmetry_max_error']:.3g}.", "",
             "Intervals are exploratory 95% percentile bootstraps over seed clusters. "
             "No multiplicity correction; no claim of confirmatory hypothesis testing.", "",
             f"## Matched libraries, original representation, {max_budget} calibration queries", "",
             "| Method | Acceleration RMSE | Extrapolation RMSE | Edit NRMSE | Rollout RMSE |",
             "|---|---:|---:|---:|---:|"]
    for method in config["methods"]:
        subset = [r for r in matched if r["method"] == method]
        cells=[]
        for metric in ("id_rmse", "ood_rmse", "edit_nrmse", "rollout_rmse"):
            e=clustered(subset,metric)
            cells.append(f"{e['mean']:.4g} [{e['low']:.4g}, {e['high']:.4g}]" if e["mean"] is not None else "NONFINITE")
        lines.append("| " + LABELS[method] + " | " + " | ".join(cells) + " |")
    lines += ["", "## Paired differences against the query residual", "",
              "Positive differences favor the query residual baseline.", "",
              "| Weight editor | Acceleration RMSE difference | Rollout RMSE difference |",
              "|---|---:|---:|"]
    for method in ("head_ridge", "jacobian_ridge", "normalized_jacobian", "finetune_adam"):
        if method not in config["methods"]:
            continue
        cells=[]
        for metric in ("id_rmse", "rollout_rmse"):
            e=paired_results[f"{method}_minus_query_residual"][metric]
            cells.append(f"{e['mean']:.4g} [{e['low']:.4g}, {e['high']:.4g}]" if e["mean"] is not None else "NONFINITE")
        lines.append("| " + LABELS[method] + " | " + " | ".join(cells) + " |")
    lines += ["", "## Integrity", "", f"Nonfinite rollout rows: {report['nonfinite_rollout_rows']}.",
              "", "See summary.json for all family/budget/representation strata and nonfinite counts.",
              "Query symbolic replacement and query residual use different deployment architectures "
              "from weight editors. Their comparison establishes a task baseline, not equal-capacity superiority."]
    (folder / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    figures(folder, rows, config)
    print("\n".join(lines))


def figures(folder, rows, config):
    figdir=folder/"figures"
    figdir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False,
                         "pdf.fonttype":42, "ps.fonttype":42})
    fig, axes = plt.subplots(1,3,figsize=(13,3.7), constrained_layout=True)
    for ax,family in zip(axes, config["families"]):
        for method in config["methods"]:
            ys=[]
            for budget in config["budgets"]:
                sub=[r for r in rows if r["family"]==family and r["budget"]==budget
                     and r["gauge"]=="original" and r["method"]==method]
                ys.append(clustered(sub,"id_rmse")["mean"])
            ax.plot(config["budgets"],ys,"o-",label=LABELS[method],color=COLORS[method],ms=4)
        ax.set(xscale="log",yscale="log",xlabel="Calibration queries",ylabel="Acceleration RMSE",
               title=family.replace("_"," ").capitalize())
        ax.grid(alpha=.2)
    handles, labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=4,frameon=False)
    for ext in ("png","pdf"):
        fig.savefig(figdir/f"query_scaling.{ext}",dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    budget=max(config["budgets"])
    selected=["query_residual","head_ridge","jacobian_ridge","normalized_jacobian"]
    for ax,metric,title in zip(axes,["gauge_disagreement","collateral_rmse"],
                               ["Sensitivity to ReLU rescaling", "Error where requested change is zero"]):
        for i,method in enumerate(selected):
            sub=[r for r in rows if r["family"]!="duffing_mismatch" and r["budget"]==budget
                 and r["gauge"]=="rescaled" and r["method"]==method]
            by_seed=defaultdict(list)
            for r in sub:
                by_seed[r["seed"]].append(r[metric])
            y=np.array([np.mean(v) for v in by_seed.values()])
            ax.scatter(np.full(len(y),i)+np.linspace(-.12,.12,len(y)),np.maximum(y,1e-16),
                       color=COLORS[method],s=24)
            ax.plot([i-.2,i+.2],[max(y.mean(),1e-16)]*2,color="black",lw=1.5)
        ax.set(yscale="log",title=title,ylabel="Function-space RMSE")
        ax.set_xticks(range(len(selected)),[LABELS[m].replace(" ","\n") for m in selected])
        ax.grid(axis="y",alpha=.2)
    for ext in ("png","pdf"):
        fig.savefig(figdir/f"robustness.{ext}",dpi=180)
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True,type=Path)
    analyze(p.parse_args().input)
