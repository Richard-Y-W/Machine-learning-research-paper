"""Post-hoc diagnostics of frozen raw-weight editing trials; no API calls."""
import csv
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def report():
    rows=[]; details={}; public={}
    for cohort in ("pilot","reading"):
        folder=ROOT/f"results/wsp-{cohort}-evaluation"
        cohort_rows=list(csv.DictReader((folder/"raw.csv").open()))
        for r in cohort_rows:
            r["cohort"]=cohort
        rows+=cohort_rows
        details.update(json.loads((folder/"details.json").read_text()))
        for path in (ROOT/f"results/wsp-{cohort}-fixtures/public").glob("*.json"):
            public[path.stem]=json.loads(path.read_text())
    llm=[r for r in rows if r["method"]=="llm"]
    semantic=[r for r in llm if r["task"]=="semantic_damping"]
    # One common condition per independently trained source avoids counting
    # representation/capacity copies as independent coefficient observations.
    unique=[r for r in semantic if r["capacity"]=="spare" and r["representation"]=="transformed"]
    table=[]
    for r in sorted(unique,key=lambda r:int(r["seed"])):
        case=r["case_id"];w=public[case]["weights"];d=details[case]
        b=np.asarray(w["b1"]);wv=np.asarray(w["W1"])[:,1];out=np.asarray(w["W2"])
        local_c=-float(np.sum(out[b>0]*wv[b>0]))
        true=d["private"]["true_c"];estimate=float(r["estimated_c"])
        query=d["query_estimated_c"]
        table.append(dict(seed=int(r["seed"]),true_c=true,llm_c=estimate,
            local_slope_c=local_c,query_c=query,prior_c=.4,
            llm_abs_error=abs(estimate-true),query_abs_error=abs(query-true),
            prior_abs_error=abs(.4-true),local_abs_error=abs(local_c-true),
            llm_local_abs_error=abs(estimate-local_c),
            own_compilation_nrmse=float(r["own_estimate_compilation_nrmse"]),
            semantic_success=int(r["success"]),source_rmse=float(r["source_physical_rmse"])))
    summary=dict(evaluated_calls=len(llm),independent_sources=len(unique),
        valid=sum(int(r["valid_patch"]) for r in llm),
        successes=sum(int(r["success"]) for r in llm),tasks={},coefficient_sources=table,
        coefficient_mae={m:float(np.mean([r[m+"_abs_error"] for r in table]))
                         for m in ("llm","query","prior","local")},
        max_own_compilation_nrmse=max(float(r["own_estimate_compilation_nrmse"]) for r in semantic),
        llm_local_mae=float(np.mean([r["llm_local_abs_error"] for r in table])),
        evaluated_cost_usd=sum(float(r["estimated_cost_usd"]) for r in llm))
    ledger=json.loads((ROOT/"results/wsp-budget.json").read_text())
    summary["total_cost_including_development_usd"]=sum(e["charged_or_reserved_usd"] for e in ledger["entries"])
    assert all(e["status"]=="completed" for e in ledger["entries"])
    assert summary["total_cost_including_development_usd"]<=ledger["cap_usd"]
    for task in ("action_gain","add_drag","semantic_damping"):
        group=[r for r in llm if r["task"]==task]
        summary["tasks"][task]=dict(n=len(group),successes=sum(int(r["success"]) for r in group),
            mean_edit_nrmse=float(np.mean([float(r["edit_nrmse"]) for r in group])),
            max_collateral_rmse=max(float(r["collateral_rmse"]) for r in group),
            mean_ood_nrmse=float(np.mean([float(r["ood_nrmse"]) for r in group])))
    folder=ROOT/"results/wsp-combined";folder.mkdir(exist_ok=True)
    (folder/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
    with (folder/"coefficient_sources.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(8,3.1),layout="constrained")
    xs=[r["true_c"] for r in table]
    for key,label,marker in (("llm_c","LLM estimate","o"),("query_c","128-query fit","s"),
                             ("local_slope_c","Network slope at origin","x")):
        axes[0].scatter(xs,[r[key] for r in table],label=label,marker=marker,s=30)
    axes[0].plot([.1,.8],[.1,.8],"k--",lw=.8);axes[0].set(xlabel="True physical c",ylabel="Estimated c",title="One transformed spare model per source")
    axes[0].legend(fontsize=7)
    names=["Action gain","Known drag","Infer damping"]
    tasks=list(summary["tasks"].values())
    axes[1].bar(names,[v["successes"]/v["n"] for v in tasks],color=["#317785","#317785","#b35943"])
    axes[1].set(ylim=(0,1.18),ylabel="Fraction of prompts passing",title="Descriptive counts; paired prompts")
    for i,v in enumerate(tasks):axes[1].text(i,v["successes"]/v["n"]+.03,f"{v['successes']}/{v['n']}",ha="center")
    figures=ROOT/"paper/figures";figures.mkdir(exist_ok=True)
    fig.savefig(figures/"wsp-diagnostics.png",dpi=220);plt.close(fig)
    lines=["# Direct weight-space programming: combined diagnostic report","",
        "Primary and follow-up outcomes remain separately available. This pooled report is descriptive; "
        "36 prompts represent eight trained sources, not 36 independent networks.","",
        "| Task | Passed | Mean edit NRMSE | Mean OOD NRMSE |",
        "|---|---:|---:|---:|"]
    for task,v in summary["tasks"].items():
        lines.append(f"| {task} | {v['successes']}/{v['n']} | {v['mean_edit_nrmse']:.5g} | {v['mean_ood_nrmse']:.5g} |")
    lines += ["","## Coefficient estimates: one condition per source","",
        "| Seed | True c | LLM | Local slope | Query fit |","|---|---:|---:|---:|---:|"]
    for r in table:lines.append(f"| {r['seed']} | {r['true_c']:.5f} | {r['llm_c']:.5f} | {r['local_slope_c']:.5f} | {r['query_c']:.5f} |")
    lines += ["","Coefficient MAE: "+json.dumps(summary["coefficient_mae"]),"",
        f"Maximum own-estimate compilation NRMSE: {summary['max_own_compilation_nrmse']:.6g}.","",
        "Local-slope analysis is post hoc. It tests an explanation of the failure, not a prespecified superiority claim. "
        "A piecewise-linear approximation's slope at one point need not equal the physical coefficient.","",
        f"Estimated list-price spend including development: ${summary['total_cost_including_development_usd']:.6f}. No additional paid calls used in analysis."]
    (folder/"REPORT.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))
    return summary


if __name__=="__main__":report()
