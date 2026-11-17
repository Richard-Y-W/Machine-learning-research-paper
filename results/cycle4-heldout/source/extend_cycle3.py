"""Integrate optimization and identification diagnostics into manuscript version 3."""
import json
from pathlib import Path

from .analyze_cycle3 import cluster, fmt, read, select


def extend(source,folder):
    root=source.parents[1]
    summary=json.loads((folder/"summary.json").read_text())
    manifest=json.loads((folder/"manifest.json").read_text())
    if manifest["status"]!="complete" or manifest["config"]["seed_start"]!=40:
        raise AssertionError("Expected completed fresh-seed cycle 3")
    fit=read(folder/"fitting.csv")
    ident=read(folder/"identification.csv")
    def fm(method,event="threshold",metric="id_rmse",budget=None):
        sub=select(fit,method=method,event=event)
        if budget is not None:
            sub=select(sub,iteration_budget=budget)
        return cluster(sub,metric)["mean"]
    def im(design,budget,access="selected",metric="id_rmse"):
        return cluster(select(ident,design=design,budget=budget,access=access),metric)["mean"]
    warm,fresh=fm("warm"),fm("fresh")
    wi,fi=fm("warm",metric="accepted_iterations"),fm("fresh",metric="accepted_iterations")
    matched=summary["matched_fit_warm_minus_fresh"]
    counts=summary["threshold_attainment"]
    n=summary["source_models"]
    matching=(f"The calibration RMSE threshold of 0.002 was attained by {counts['warm']}/{n} "
        f"warm and {counts['fresh']}/{n} fresh students. At the first crossing, warm and fresh "
        f"physical errors were {warm:.4f} and {fresh:.4f}; the paired warm-minus-fresh difference "
        f"was {fmt(matched)} across {matched['paired_model_conditions']} jointly attained "
        "model conditions. Thresholds are evaluated only every 30 optimizer iterations, so "
        "this is approximately matched fitting rather than exact equality of training losses. "
        "If attainment is incomplete, this contrast is conditional on the jointly attained subset.")
    iterations=(f"Mean L-BFGS iteration counters at first crossing were {wi:.1f} for warm and "
        f"{fi:.1f} for fresh initialization. A lower held-out error therefore need not mean "
        "fewer iterations to a specified calibration loss. This distinction was obscured by "
        "the previous short-budget comparison.")
    fitting_table=["| Initialization | Budget | Calibration fit | Teacher-fit RMSE | Physical RMSE |",
                   "|---|---:|---:|---:|---:|"]
    for budget in (120,480,1920):
        for method in ("fresh","warm"):
            fitting_table.append(f"| {method} | {budget} | "+" | ".join(
                f"{fm(method,'fixed',metric,budget):.5f}" for metric in
                ("calibration_fit_rmse","teacher_fit_rmse","id_rmse"))+" |")
    mixed,narrow=im("mixed",64),im("narrow",256)
    known_narrow=im("narrow",256,"known")
    broad64=select(ident,design="broad",budget=64,access="selected")
    successes=int(sum(r["correct_family"] for r in broad64))
    broad_joint=next(r for r in summary["identification_summary"]
                     if r["budget"]==64 and r["design"]=="broad").get("joint_seed_success")
    placement=(f"With the true family label withheld, 64 mixed queries gave mean changed-dynamics "
        f"physical RMSE {mixed:.4f}, compared with {narrow:.4f} for 256 narrow queries. The paired "
        f"mixed-minus-narrow contrast was {fmt(summary['mixed64_minus_narrow256'])}. Broad "
        f"64-query calibration selected the correct family in {successes}/{n} observed source "
        "models. This is an empirical count from two candidate families, not a guarantee for "
        "other mechanisms or future models.")
    identification_table=["| Budget | Design | Correct family | Selected physical RMSE | Known-family RMSE |",
                          "|---:|---|---:|---:|---:|"]
    for budget in (16,64,256):
        for design in ("narrow","broad","mixed"):
            sub=select(ident,design=design,budget=budget,access="selected")
            correct=int(sum(r["correct_family"] for r in sub))
            identification_table.append(f"| {budget} | {design} | {correct}/{n} | "
                f"{im(design,budget):.4f} | {im(design,budget,'known'):.4f} |")
    section="\n\n".join([
        "## 8. Longer fitting and family uncertainty on fresh source models",
        "### 8.1 Similar calibration fit does not imply similar held-out behavior",
        "We freeze a third protocol in Git commit "+manifest["git_commit"][:7]+
        " after development on seeds 0-1, then train 24 new source models using seeds 40-51. "
        "The original architecture, physical interface, and combined intervention are unchanged. "
        "For the optimization diagnostic we fit the symbolic teacher inferred from 128 broad "
        "queries. L-BFGS retains its history between 30-iteration chunks and runs up to a "
        "1,920-iteration budget. We record the first calibration RMSE <=0.002 and fixed "
        "checkpoints at 120, 480, and 1,920 iterations, continuing after any threshold crossing. "
        "Checkpoint selection never uses held-out physical or teacher labels.",
        matching,iterations,"\n".join(fitting_table),
        "Table 4. Fixed-budget fitting curves on the third seed range. L-BFGS may numerically "
        "converge before its iteration cap. Physical error and held-out teacher error must be "
        "distinguished from calibration error.",
        "At the full 1,920-iteration budget, the paired physical-error difference was "+
        fmt(summary["final_warm_minus_fresh"])+". Extrapolation RMSE at that budget was "
        f"{fm('warm','fixed','ood_rmse',1920):.4f} for warm and "
        f"{fm('fresh','fixed','ood_rmse',1920):.4f} for fresh students. A remaining gap is "
        "consistent with an initialization-dependent solution bias or optimization path, "
        "but this experiment does not identify a unique causal mechanism. Effects of "
        "initialization and feature distortion are established topics [11,12].",
        "![Longer fitting control](../results/cycle3-heldout/figures/long_fitting.png)",
        "Figure 5. Calibration error continues to fall as optimizer work increases, while "
        "held-out teacher error can follow a different curve. Curves summarize the same "
        "fixed source models, not independent replications at each checkpoint.",
        "### 8.2 Query placement can matter more than query count",
        "The earlier query baseline was supplied the correct family. We now provide both "
        "four-term candidate libraries while withholding the true family label. Fixed "
        "four-fold cross-validation on source-network outputs chooses a candidate, which "
        "is then refit using all queried points. The candidate set still contains the true "
        "family; this is limited family uncertainty rather than arbitrary physics discovery. "
        "We compare budgets 16/64/256 and three placements: uniform narrow queries in "
        "[-0.25,0.25]^3, uniform broad queries in [-1.5,1.5]^3, and eight narrow queries "
        "followed by broad queries. The known-family comparator gets only the family label "
        "and estimates its coefficients from the same outputs.",
        placement,"\n".join(identification_table),
        "Table 5. Family-label access and query design. Correct-family counts are descriptive "
        "observations on 24 source networks clustered into 12 seeds; they do not establish "
        "population-perfect identification. No changed-environment labels are supplied.",
        (f"For the supplementary event that both models in a seed are identified correctly "
         f"at 64 broad queries, {broad_joint['successful_seeds']}/{broad_joint['seeds']} independent "
         f"seeds succeed. The exact binomial 95% interval is [{broad_joint['low']:.3f}, "
         f"{broad_joint['high']:.3f}], rather than a claim of certainty from observing all successes."
         if broad_joint else "Observed all-success counts do not imply population-perfect identification."),
        f"Even with the correct family supplied, 256 narrow queries produced physical RMSE "
        f"{known_narrow:.4f}. Thus family classification alone does not explain every failure. "
        "Narrow inputs weakly excite nonlinear features, and least-squares inversion can "
        "amplify the learned source model's approximation errors. The error identity in "
        "Section 4 exposes this route. More noiseless evaluations of an approximate "
        "network do not eliminate its systematic local approximation error. This is a "
        "classical identification issue, not evidence for a uniquely weight-based remedy [1].",
        "![Query placement study](../results/cycle3-heldout/figures/query_placement.png)",
        "Figure 6. Selected-family dynamics error and observed family accuracy by query "
        "budget and placement. Query design changes the portion of the learned function "
        "that is observed. All queries access the source model; broad physical exploration "
        "of a real changed environment is not assumed.",
        "### 8.3 Implications for evaluating weight access",
        "The two controls separate different failure modes. Similar training-point losses "
        "need not imply similar extensions of the learned dynamics. Conversely, a poor "
        "query baseline can reflect concentrated inputs or an uncertain mechanism library. "
        "A meaningful weight-access claim should therefore specify both the optimizer "
        "comparison and the query interface. Since executable weights permit function "
        "queries, restrictions on where queries may be made must be justified by the application."
    ])
    text=source.read_text(encoding="utf-8")
    text=text.replace("# What Does Weight Access Buy? Targets, Initialization, and Representation in Dynamics Editing",
        "# What Does Weight Access Buy? Initialization and Identification in Learned Dynamics",1)
    text=text.replace("Working manuscript, version 2","Working manuscript, version 3",1)
    text=text.replace("Two controlled numerical studies with held-out model evaluation.",
        "Three controlled numerical studies with fresh-model evaluation.",1)
    text=text.replace("## 8. Interpretation and limitations",section+"\n\n## 9. Interpretation and limitations",1)
    text=text.replace("## 9. Research decision and next experiment","## 10. Research decision and next experiment",1)
    text=text.replace("The current draft provides a reproducible methodological case study.",
        "The matched-fit and query-placement controls strengthen the empirical diagnosis. "
        "The current draft provides a reproducible methodological case study.",1)
    start=text.index("## Abstract")
    end=text.index("## 1. Introduction",start)
    abstract=("## Abstract\n\nDirect weight editing is an appealing interface for modifying learned "
        "dynamics, but its value depends on behavioral baselines, initialization, and the "
        "information supplied to an editor. We study these factors in small neural acceleration "
        "models using three controlled numerical studies. Structured query teachers provide "
        "strong baselines when a mechanism library is known. In a fixed deployment architecture, "
        "pretrained initialization improves student accuracy under sparse and dense synthetic "
        "supervision. On a third set of 24 fresh source models, approximately matched calibration "
        f"fit yields physical errors {warm:.4f} for warm and {fresh:.4f} for fresh students. "
        "With the true family withheld from a two-candidate selector, query placement also "
        f"matters: 64 mixed queries give error {mixed:.4f}, versus {narrow:.4f} for 256 narrow "
        "queries. These findings motivate separating optimization, functional generalization, "
        "and identification assumptions in evaluations of weight access. They characterize "
        "this controlled setting rather than establish semantic understanding of weights or "
        "a language-model programming capability.\n\n")
    text=text[:start]+abstract+text[end:]
    text=text.replace("The follow-up adds 24 held-out models, 2,688 factorial measurements, and 96 dense-control measurements.",
        "The second study adds 24 held-out models, 2,688 factorial measurements, and 96 dense-control "
        f"measurements. The third adds 24 fresh source models, {len(fit)} fitting checkpoints, and "
        f"{len(ident)} identification measurements.")
    text += "\n[11] Kumar, A., Raghunathan, A., Jones, R., Ma, T., and Liang, P. (2022). Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution. https://arxiv.org/abs/2202.10054\n\n"
    text += "[12] Anguita, N., Locatello, F., Saxe, A. M., Mondelli, M., Mancini, F., Lippl, S., and Domine, C. C. J. (2026). A Theory of How Pretraining Shapes Inductive Bias in Fine-Tuning. ICML, PMLR 306, 2893-2920. https://proceedings.mlr.press/v306/anguita26a.html\n"
    source.write_text(text,encoding="utf-8")
    discoveries=("# Cycle 3 discoveries\n\n## 1. Fitting versus generalization\n\n"+matching+"\n\n"+
        iterations+"\n\n## 2. Query placement versus count\n\n"+placement+"\n\n"+
        f"Knowing the true family still leaves RMSE {known_narrow:.4f} with 256 narrow queries. "
        "Coefficient identification, not just family selection, can fail under concentrated inputs.\n\n"
        "## Scope\n\nThese are controlled findings on two supplied physical families and small "
        "networks. They strengthen the experimental diagnosis; novelty remains a separate "
        "question. They do not test LLM weight interpretation or discovery of arbitrary dynamics.\n")
    (root/"docs/cycle3-discoveries.md").write_text(discoveries,encoding="utf-8")
    decision=(root/"docs/decision.md").read_text(encoding="utf-8")
    decision+="\n## Third-cycle update\n\n"+matching+"\n\n"+placement+"\n\n"
    decision+="The next research question should specify identifiability and query access, "
    decision+="rather than treating a poor behavioral baseline as evidence of semantic weight "
    decision+="understanding. Longer fitting alone does not settle the comparison.\n"
    (root/"docs/decision.md").write_text(decision,encoding="utf-8")
    return source
