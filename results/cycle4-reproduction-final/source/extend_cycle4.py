"""Rebuild the manuscript around the four-stage access comparison."""
import json
from pathlib import Path

from .analyze_cycle4 import LABELS,cluster,fmt,read,subset
from .analyze_cycle3 import read as read3,select as select3,cluster as cluster3


def extend(source,folder):
    root=source.parents[1]
    manifest=json.loads((folder/"manifest.json").read_text())
    summary=json.loads((folder/"summary.json").read_text())
    if manifest["status"]!="complete" or manifest["config"]["seed_start"]!=60:
        raise AssertionError("Expected complete fresh-model cycle 4")
    rows=read(folder/"raw.csv")
    focal=subset(rows,shift="nonlinear",budget=64)
    def mean(method,metric="id_rmse",data=focal):
        return cluster(subset(data,method=method),metric)["mean"]
    warm,fresh=mean("warm_full"),mean("fresh_full")
    rbf,ard=mean("rbf_residual"),mean("ard_rbf_residual")
    neural,head=mean("neural_residual"),mean("learned_head")
    pri=summary["contrasts"]["rbf_residual_minus_warm_full_id_rmse"]
    ard_pri=summary["contrasts"]["ard_rbf_residual_minus_warm_full_id_rmse"]
    paired=("The primary isotropic-RBF-minus-warm-full physical-error contrast was "+fmt(pri)+
        "; isotropic RBF minus learned head was "+
        fmt(summary["contrasts"]["rbf_residual_minus_learned_head_id_rmse"])+
        ". The additional coordinate-scaled RBF control minus warm full was "+fmt(ard_pri)+
        ". These exploratory paired intervals resample independent seed clusters and are "
        "not multiplicity-adjusted hypothesis tests. Coordinate scaling also increases "
        "hyperparameter-search work, so its contrast is not a pure causal test of one geometric factor.")
    models=json.loads((folder/"models.json").read_text())
    errors=[m["source_rmse"] for m in models]
    fidelity=(f"Source physical RMSE ranged from {min(errors):.4f} to {max(errors):.4f}, "
              f"with mean {sum(errors)/len(errors):.4f}. All source models and conditions are retained.")
    table=["| Method | Physical RMSE | Extrapolation RMSE | Rollout RMSE | Extra scalars |",
           "|---|---:|---:|---:|---:|"]
    for method in manifest["config"]["methods"]:
        table.append("| "+LABELS[method]+" | "+" | ".join(f"{mean(method,m):.4f}" for m in
            ("id_rmse","ood_rmse","rollout_rmse"))+f" | {mean(method,'extra_scalars'):.0f} |")
    parameter=subset(rows,shift="parameter",budget=16)
    parameter_result=(f"For the affine parameter-shift control at 16 observations, linear-residual "
        f"physical error was {mean('linear_residual',data=parameter):.4f}, compared with "
        f"{mean('warm_full',data=parameter):.4f} for warm full adaptation and "
        f"{mean('fresh_full',data=parameter):.4f} for a fresh full model. This control favors a "
        "simple correction by construction and should not be interpreted as evidence of "
        "unrestricted adaptation capability. Under added mechanisms at 64 observations, "
        f"fresh full-network and fresh neural-residual mean errors were {fresh:.4f} and "
        f"{neural:.4f}. The latter retains source predictions at deployment without copying "
        "the source weights into its correction model.")
    system_result="Per-system nonlinear-shift errors at 64 observations were "
    system_result+=("; ".join(f"{family}: warm full "
        f"{mean('warm_full',data=subset(focal,family=family)):.4f}, coordinate-scaled residual "
        f"{mean('ard_rbf_residual',data=subset(focal,family=family)):.4f}"
        for family in manifest["config"]["families"])+". These are new source contexts with the same "
        "two fixed shift definitions; they do not constitute unseen intervention-family transfer.")
    prior3=root/"results/cycle3-heldout"
    s3=json.loads((prior3/"summary.json").read_text())
    fit3=read3(prior3/"fitting.csv")
    ident3=read3(prior3/"identification.csv")
    fit_warm=cluster3(select3(fit3,event="threshold",method="warm"),"id_rmse")["mean"]
    fit_fresh=cluster3(select3(fit3,event="threshold",method="fresh"),"id_rmse")["mean"]
    matched=(f"A third frozen evaluation trained 24 fresh source models on seeds 40-51. "
        f"All warm and fresh students reached calibration RMSE <=0.002. At the first crossing, "
        f"physical errors were {fit_warm:.4f} warm versus {fit_fresh:.4f} fresh; the paired "
        "difference was "+fmt(s3["matched_fit_warm_minus_fresh"])+
        ". The gap also persisted at a fixed 1,920-iteration budget. Thus additional fitting "
        "did not eliminate the initialization effect in that architecture-constrained task.")
    mixed=cluster3(select3(ident3,access="selected",design="mixed",budget=64),"id_rmse")["mean"]
    narrow=cluster3(select3(ident3,access="selected",design="narrow",budget=256),"id_rmse")["mean"]
    query_result=(f"When the true family was hidden among two supplied candidate libraries, "
        f"64 mixed-location source queries gave error {mixed:.4f}, versus {narrow:.4f} for "
        "256 narrow queries. The paired difference was "+fmt(s3["mixed64_minus_narrow256"])+
        ". This demonstrates sensitivity to coverage within a favorable candidate set, not "
        "discovery of arbitrary physical laws. All-success family counts are observations "
        "from 12 seed clusters rather than population guarantees.")
    if pri["high"]<0:
        primary_conclusion="The primary isotropic residual comparison favors prediction-only reuse in this setting. "
    elif pri["low"]>0:
        primary_conclusion="Warm parameter adaptation outperforms the primary isotropic residual comparator in this setting. "
    else:
        primary_conclusion="The primary isotropic residual comparison does not establish a clear direction of advantage. "
    conclusion=(primary_conclusion+"The additional controls and earlier studies show why conclusions "
        "must be tied to an access and deployment contract. An initialization advantage over "
        "a fresh student does not by itself establish an advantage over methods that retain "
        "the source function. Internal feature reuse and functional reuse should be evaluated "
        "separately, especially when observed changes differ from source training dynamics.")
    original=source.read_text(encoding="utf-8")
    references=original.split("## References\n\n",1)[1].strip()
    references+="\n\n[13] Rasmussen, C. E., and Williams, C. K. I. (2006). Gaussian Processes for Machine Learning. MIT Press. https://gaussianprocess.org/gpml/\n\n"
    references+="[14] Brudigam, J., Schuck, M., Capone, A., Sosnowski, S., and Hirche, S. (2022). Structure-Preserving Learning Using Gaussian Processes and Variational Integrators. L4DC, PMLR 168, 1150-1162. https://proceedings.mlr.press/v168/brudigam22a.html"
    tokens=dict(ABSTRACT_RESULT=f"At 64 observations under added mechanisms, mean physical "
        f"RMSE was {warm:.4f} for full warm adaptation, {rbf:.4f} for an isotropic prediction-only "
        f"residual, and {ard:.4f} for a coordinate-scaled residual control.",
        MATCHED_FIT_RESULT=matched,QUERY_DESIGN_RESULT=query_result,SOURCE_FIDELITY=fidelity,
        MAIN_RESULT=f"Under added mechanisms and 64 noisy observations, full warm adaptation "
        f"achieved physical RMSE {warm:.4f}, compared with {fresh:.4f} for fresh full adaptation, "
        f"{head:.4f} for learned-head adaptation, {rbf:.4f} for isotropic residuals, and "
        f"{ard:.4f} for coordinate-scaled residuals. No adapter received the changed force formulas.",
        MAIN_TABLE="\n".join(table),PAIRED_RESULT=paired,PARAMETER_RESULT=parameter_result,
        SYSTEM_RESULT=system_result,INTEGRITY_RESULT=f"The run completed in "
        f"{manifest['elapsed_seconds']/60:.2f} wall-clock minutes on CPU. Nonfinite rollout "
        f"rows: {summary['nonfinite_rollout_rows']}. Calibration losses, selection choices, "
        "and extra storage counts are retained for every row.",CONCLUSION=conclusion,REFERENCES=references)
    text=(root/"paper/v4.template.md").read_text(encoding="utf-8")
    for key,value in tokens.items():
        text=text.replace("{{"+key+"}}",value)
    if "{{" in text:
        raise AssertionError("Unfilled manuscript field")
    source.write_text(text,encoding="utf-8")
    memo=("# Cycle 4 discoveries\n\nThe task is now adaptation from actual noisy changed-environment "
        "observations. This differs from specification editing. No mechanism library is supplied.\n\n"+
        tokens["MAIN_RESULT"]+"\n\n"+paired+"\n\n"+parameter_result+"\n\n"+conclusion+
        "\n\nThe kernel and neural residuals deploy around a callable source function. Their "
        "performance is not equal-capacity superiority over in-place edits, and the low-dimensional "
        "state interface is favorable to classical regression. No LLM is tested.\n")
    (root/"docs/cycle4-discoveries.md").write_text(memo,encoding="utf-8")
    (root/"docs/decision.md").write_text("# Updated research decision\n\n"+conclusion+"\n\n"+
        tokens["MAIN_RESULT"]+"\n\n"+paired+"\n\nThe most useful next test is higher-dimensional "
        "or nuisance-rich observations and varied interventions, with prediction-only residual "
        "baselines preserved. Current evidence supports a controlled access-comparison case study; "
        "publication novelty remains unresolved.\n",encoding="utf-8")
    return source
