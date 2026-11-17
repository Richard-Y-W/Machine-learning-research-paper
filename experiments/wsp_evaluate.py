"""Evaluate already committed LLM patches; never supplies feedback to API calls."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .wsp_common import apply_patch,compile_edit,forward


def phi(x):
    q,v,u=x.T
    return np.stack((-np.sin(q),-v,-v*np.abs(v),u),1)


def rms(x):
    value=float(np.sqrt(np.mean(np.asarray(x)**2)))
    return value if np.isfinite(value) else None


def changed_count(a,b):
    return sum(int(np.sum(np.abs(np.asarray(a[k])-np.asarray(b[k]))>1e-12)) for k in a)


def target(weights,x,task,c):
    if task=="action_gain":
        shifted=x.copy();shifted[:,2]*=1.5
        return forward(weights,shifted)
    return forward(weights,x)-(.2 if task=="add_drag" else c)*x[:,1]


def evaluate(fixtures,calls,out):
    manifest=json.loads((calls/"manifest.json").read_text())
    if manifest["status"] not in ("complete","budget_stopped","api_error"):
        raise RuntimeError("Calls still in progress; do not reveal outcomes yet")
    out.mkdir(parents=True,exist_ok=True)
    rows=[];details={}
    fixture_manifest=json.loads((fixtures/"manifest.json").read_text())
    for case in manifest["planned_cases"]:
        public=json.loads((fixtures/"public"/f"{case}.json").read_text())
        private=json.loads((fixtures/"private"/f"{case}.json").read_text())
        for access,value in (("public",public),("private",private)):
            if "canonical_json_sha256" in fixture_manifest:
                actual=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()
                if actual!=fixture_manifest["canonical_json_sha256"][case][access]:
                    raise AssertionError("Fixture modified after sealing")
        w=public["weights"];task=private["task"];c=private["true_c"]
        seed=private["seed"]
        x=np.random.default_rng(910000+seed).uniform(-1,1,(4096,3))
        ood=np.random.default_rng(920000+seed).uniform(-1.5,1.5,(4096,3))
        zero=x.copy();zero[:,2 if task=="action_gain" else 1]=0
        expected=target(w,x,task,c);expected_ood=target(w,ood,task,c)
        base=forward(w,x);delta=expected-base
        source_physical=rms(base-phi(x)@private["coefficients"])
        response_path=calls/"responses"/f"{case}.response.json"
        if not response_path.exists():
            rows.append(dict(case_id=case,**{k:private[k] for k in ("seed","capacity","representation","task")},
                method="llm",attempted=int(case in manifest.get("failed_cases",[])),valid_patch=0,
                status="api_error" if case in manifest.get("failed_cases",[]) else "not_attempted",
                success=0,edit_nrmse=None,ood_nrmse=None,collateral_rmse=None,changed_parameters=0,
                estimated_c=None,coefficient_abs_error=None,own_estimate_compilation_nrmse=None,
                source_physical_rmse=source_physical,estimated_cost_usd=0.,stop_reason=""))
            continue
        response=json.loads(response_path.read_text())
        request=json.loads((calls/"responses"/f"{case}.request.json").read_text())
        if request["messages"][0]["content"]!=public["prompt"] or "tools" in request:
            raise AssertionError("Request does not match the frozen tool-free prompt")
        text="".join(b.get("text","") for b in response.get("content",[]) if b.get("type")=="text")
        parsed=None;error=None
        try:
            if response.get("stop_reason")!="end_turn":
                raise ValueError("Response did not complete normally")
            parsed=json.loads(text)
            modified=apply_patch(w,parsed["patches"])
            status="empty_patch" if not parsed["patches"] else "valid"
        except (ValueError,KeyError,TypeError) as exc:
            error=str(exc);modified=w;status="invalid_response"
        methods=[("llm",modified,parsed.get("estimated_c") if isinstance(parsed,dict) else None)]
        methods.append(("unchanged",w,None))
        oracle=compile_edit(w,task,c)
        if oracle is not None:
            methods.append(("oracle_compiler" if task=="semantic_damping" else "analytic_compiler",oracle,c))
        probes=np.random.default_rng(930000+seed).uniform(-1,1,(128,3))
        fitted=np.linalg.lstsq(phi(probes),forward(w,probes),rcond=None)[0]
        if task=="semantic_damping":
            query_compiled=compile_edit(w,task,float(fitted[1]))
            if query_compiled is not None:
                methods.append(("query_compiler",query_compiled,float(fitted[1])))
            prior=compile_edit(w,task,.4)
            if prior is not None:methods.append(("prior_mean_compiler",prior,.4))
        # A numerical capacity reference. Semantic labels use an oracle c here.
        hidden=np.maximum(probes@np.asarray(w["W1"]).T+w["b1"],0)
        h=np.column_stack((hidden,np.ones(len(probes))))
        correction=target(w,probes,task,c)-forward(w,probes)
        beta=np.linalg.lstsq(h,correction,rcond=None)[0]
        numerical=copy.deepcopy(w)
        numerical["W2"]=(np.asarray(w["W2"])+beta[:-1]).tolist()
        numerical["b2"]=[w["b2"][0]+beta[-1]]
        methods.append(("oracle_head_fit" if task=="semantic_damping" else "numerical_head_fit",numerical,c))
        for method,weights,estimate in methods:
            predicted=forward(weights,x)
            nrmse=rms(predicted-expected)/max(rms(delta),1e-12)
            extrap=rms(forward(weights,ood)-expected_ood)/max(rms(expected_ood-forward(w,ood)),1e-12)
            collateral=rms(forward(weights,zero)-forward(w,zero))
            tolerance=1e-4 if task=="action_gain" else .05 if task=="add_drag" else .1
            collateral_tolerance=1e-6 if task=="action_gain" else .01
            valid=method!="llm" or error is None
            estimate_ok=type(estimate) in (float,int) and np.isfinite(estimate)
            own=None
            if task=="semantic_damping" and estimate_ok and abs(estimate)>1e-12:
                own=rms(predicted-base+estimate*x[:,1])/max(rms(estimate*x[:,1]),1e-12)
            row=dict(case_id=case,**{k:private[k] for k in ("seed","capacity","representation","task")},
                method=method,attempted=1,valid_patch=int(valid),status=status if method=="llm" else "reference",
                success=int(valid and nrmse<=tolerance and collateral<=collateral_tolerance),
                edit_nrmse=nrmse,ood_nrmse=extrap,collateral_rmse=collateral,
                changed_parameters=changed_count(w,weights),estimated_c=float(estimate) if estimate_ok else None,
                coefficient_abs_error=abs(estimate-c) if task=="semantic_damping" and estimate_ok else None,
                own_estimate_compilation_nrmse=own,source_physical_rmse=source_physical,
                estimated_cost_usd=response.get("estimated_list_price_usd",0.) if method=="llm" else 0.,
                stop_reason=response.get("stop_reason","") if method=="llm" else "")
            rows.append(row)
        details[case]=dict(private=private,parsed_final_response=parsed,error=error,
            query_estimated_c=float(fitted[1]),
            provably_inactive_units=int(np.sum(np.asarray(w["b1"])+np.abs(w["W1"]).sum(1)<=0)))
    with (out/"raw.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/"details.json").write_text(json.dumps(details,indent=2))
    llm=[r for r in rows if r["method"]=="llm"]
    summary=dict(planned=len(llm),attempted=sum(r["attempted"] for r in llm),
        valid=sum(r["valid_patch"] for r in llm),successes=sum(r["success"] for r in llm),
        estimated_cost_usd=sum(r["estimated_cost_usd"] for r in llm),
        source_seeds=sorted({r["seed"] for r in llm}),strata=[])
    lines=["# Direct LLM weight-space programming results","",
        f"Model: {manifest['model']}; tool-free, no forward passes, no data, no feedback.","",
        f"{summary['attempted']}/{summary['planned']} calls attempted; {summary['valid']} valid patches; "
        f"{summary['successes']} successful functional edits. Estimated generation cost: ${summary['estimated_cost_usd']:.4f}.","",
        "These prompts derive from only "+str(len(summary["source_seeds"]))+" independently trained source networks. "
        "Capacity and representation variants are paired controls, not independent replications.","",
        "| Task | Capacity | Valid / attempted | Successful | Mean edit NRMSE |",
        "|---|---|---:|---:|---:|"]
    for task in ("action_gain","add_drag","semantic_damping"):
        for capacity in ("dense","spare"):
            group=[r for r in llm if r["task"]==task and r["capacity"]==capacity and r["attempted"]]
            values=[r["edit_nrmse"] for r in group if r["edit_nrmse"] is not None]
            mean=float(np.mean(values)) if values else None
            result=dict(task=task,capacity=capacity,attempted=len(group),valid=sum(r["valid_patch"] for r in group),
                successful=sum(r["success"] for r in group),mean_edit_nrmse=mean)
            summary["strata"].append(result)
            lines.append(f"| {task} | {capacity} | {result['valid']}/{len(group)} | {result['successful']}/{len(group)} | "
                +(f"{mean:.5g}" if mean is not None else "unavailable")+" |")
    lines += ["","## Interpretation","",
        "Action-column scaling is an architectural arithmetic control. Spare-unit drag edits have an "
        "exact elementary ReLU compiler. Matching these controls demonstrates direct numerical "
        "programming, not a new optimization algorithm or general semantic understanding.","",
        "Semantic damping combines reading an approximate physical coefficient with compiling a "
        "correction. The details file separates coefficient error and fidelity to the model's own "
        "estimated change. Oracle references use hidden physical coefficients; query references "
        "use source evaluations unavailable to the tool-free LLM arm.","",
        "Invalid/truncated responses count as failures. Budget-stopped prompts are reported as "
        "unattempted. No prompt tuning or retries are performed on evaluated pilot outcomes."]
    (out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
    (out/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixtures",required=True,type=Path);p.add_argument("--calls",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args();evaluate(a.fixtures,a.calls,a.output)
