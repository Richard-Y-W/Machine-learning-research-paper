"""Integrate the held-out deployment study and strong dense query control."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .analyze_deployment import LABELS, cluster, difference, fmt, read_rows, subset


def verify_snapshot(folder):
    manifest=json.loads((folder/"manifest.json").read_text())
    if manifest["status"]!="complete":
        raise RuntimeError(f"Incomplete run {folder}")
    h=hashlib.sha256()
    for path in sorted((folder/"source").glob("*.py")):
        h.update(path.name.encode());h.update(path.read_bytes())
    if h.hexdigest()!=manifest["source_sha256"]:
        raise AssertionError(f"Source snapshot mismatch in {folder}")
    return manifest


def extend(source,heldout,dense):
    root=source.parents[1]
    manifest=verify_snapshot(heldout)
    dense_manifest=verify_snapshot(dense)
    rows=read_rows(heldout/"raw.csv")
    drows=read_rows(dense/"raw.csv")
    if manifest["config"]["seed_start"]!=20 or manifest["config"]["seeds"]!=12:
        raise AssertionError("Expected frozen held-out seed split")
    expected=12*2*2*2*2*2*7
    if len(rows)!=expected or len(drows)!=12*2*2*2:
        raise AssertionError("Missing follow-up rows")
    if any(r["parameters"]!=1217 for r in rows+drows):
        raise AssertionError("Deployment parameter counts differ")
    if any(r["base_queries"]!=128 or r["synthetic_labels"]!=4096 for r in drows):
        raise AssertionError("Incorrect dense-control access accounting")
    focal=subset(rows,budget=128,schedule="long",gauge="original")
    def metric(method,teacher,name="id_rmse"):
        return cluster(subset(focal,method=method,teacher=teacher),name)["mean"]
    primary=difference(subset(focal,teacher="symbolic"),"method","warm_lbfgs","cold_lbfgs","id_rmse")
    dense_long=subset(drows,steps=2400)
    dense_contrast=difference(dense_long,"method","warm_dense","cold_dense","id_rmse")
    def dm(method,steps=2400,name="id_rmse"):
        return cluster(subset(drows,method=method,steps=steps),name)["mean"]
    warm,cold=metric("warm_lbfgs","symbolic"),metric("cold_lbfgs","symbolic")
    dw,dc=dm("warm_dense"),dm("cold_dense")
    primary_text=(f"Under the fixed long L-BFGS schedule with symbolic targets, original "
        f"representation, and 128 queries, warm initialization achieved mean physical RMSE "
        f"{warm:.4f}, compared with {cold:.4f} for a fresh student. The predeclared warm-minus-fresh "
        f"contrast was {fmt(primary)}. Families are averaged within each of the 12 new seed clusters. "
        "This comparison isolates initialization under this optimizer and sparse-label schedule. "
        "It establishes neither semantic interpretation of weights nor superiority over all query procedures.")
    table=["| Solver | Residual physical | Symbolic physical | Symbolic teacher fit | Symbolic seconds |",
           "|---|---:|---:|---:|---:|"]
    for method in manifest["config"]["methods"]:
        table.append("| "+LABELS[method]+" | "+" | ".join(f"{v:.4f}" for v in
            [metric(method,"residual"),metric(method,"symbolic"),
             metric(method,"symbolic","teacher_fit_rmse"),metric(method,"symbolic","edit_seconds")])+" |")
    target_lines=[]
    for method in ("warm_lbfgs","cold_lbfgs","relative_jacobian"):
        delta=difference(subset(focal,method=method),"teacher","symbolic","residual","id_rmse")
        target_lines.append(LABELS[method]+": "+fmt(delta))
    rescaled=subset(rows,budget=128,schedule="long",gauge="rescaled",teacher="symbolic")
    gauge_text="; ".join(f"{LABELS[m]} {cluster(subset(rescaled,method=m),'gauge_disagreement')['mean']:.3g}"
                        for m in ("euclidean_jacobian","column_jacobian","relative_jacobian"))
    dense_table=["| Student | Adam steps | Physical RMSE | Teacher-fit RMSE | Construction seconds |",
                 "|---|---:|---:|---:|---:|"]
    for steps in (600,2400):
        for method,label in (("cold_dense","Fresh dense"),("warm_dense","Warm dense")):
            dense_table.append("| "+label+f" | {steps} | "+" | ".join(f"{dm(method,steps,m):.4f}" for m in
                ("id_rmse","teacher_fit_rmse","edit_seconds"))+" |")
    dense_text=(f"With 4,096 free synthetic labels and 2,400 minibatch Adam steps, fresh and warm "
        f"students achieved physical RMSE {dc:.4f} and {dw:.4f}, respectively. The paired warm-minus-fresh "
        f"contrast was {fmt(dense_contrast)}. Both use only the same 128 base-model queries. "
        "The dense schedules are a separate cost curve; their optimizer, number of updates, and "
        "input distribution differ from sparse L-BFGS. Cross-curve differences cannot be attributed "
        "to input density alone. Within each dense schedule, initialization is the only changed factor.")
    section="\n\n".join([
        "## 7. Same-architecture follow-up on held-out models",
        "### 7.1 Separating initialization from target objectives",
        "The first study leaves two confounds unresolved. Structured teachers deploy outside the "
        "original architecture, and symbolic reconstruction has a different objective from preserving "
        "the base function plus an intervention. We therefore cross teacher objective with student "
        "initialization and solver. Every student is a 1,217-parameter network with the original "
        "architecture. Fresh students are initialized independently of the source weights; warm "
        "students copy them. This explicitly treats copying weights as weight access rather than "
        "calling a warm-started procedure query-only.",
        "The protocol and numerical solvers were frozen in Git commit "+manifest["git_commit"][:7]+
        " after development on seeds 0-1 and before evaluating seeds 20-31. We train 24 new base "
        "models, apply the fixed combined intervention, and cross two budgets (32/128 queries), "
        "two targets, two schedules, two representations, and seven solvers, yielding 2,688 "
        "condition-method rows. Short/long schedules use 150/600 Adam steps, at most 30/120 "
        "L-BFGS iterations, or 3/12 Jacobian iterations. These are not matched FLOP budgets. "
        "No solver hyperparameter was tuned on the held-out editing results.",
        primary_text,
        "Calibration teacher-fit RMSE was "
        f"{metric('warm_lbfgs','symbolic','calibration_fit_rmse'):.4f} for warm L-BFGS and "
        f"{metric('cold_lbfgs','symbolic','calibration_fit_rmse'):.4f} for fresh L-BFGS, versus "
        f"held-out teacher-fit errors {metric('warm_lbfgs','symbolic','teacher_fit_rmse'):.4f} and "
        f"{metric('cold_lbfgs','symbolic','teacher_fit_rmse'):.4f}. Initialization therefore changes "
        "both fitting progress and held-out behavior under this schedule. We do not isolate "
        "optimization difficulty from an implicit functional prior.",
        "\n".join(table),
        "Table 2. Held-out matched systems, 128 queries, long schedule, original representation. "
        "Physical error and teacher-fitting error answer different questions. Entries are "
        "seed-cluster means; paired uncertainty is reported in the text and full report.",
        "### 7.2 A better teacher need not produce a better student",
        "Symbolic-minus-residual physical-error contrasts under the long schedule were "+"; ".join(target_lines)+
        ". The unrestricted symbolic function's low physical error is therefore insufficient "
        "to predict the quality of its same-architecture approximation. Student fitting, "
        "generalization, and teacher bias must be measured separately. On a fixed evaluation "
        "distribution the triangle inequality gives the elementary bound below; it is not "
        "a new theoretical result.",
        r"$$ \|s-f^*\|_2\leq\|s-t\|_2+\|t-f^*\|_2. $$",
        "![Factorial deployment study](../results/deployment-heldout/figures/factorial.png)",
        "Figure 3. Held-out physical errors by solver and target, with seed-cluster intervals. "
        "The dashed line is the unrestricted teacher's physical error, not a lower bound on "
        "student error. Short and long computational schedules are descriptive choices.",
        "### 7.3 Representation stability and accuracy are distinct",
        "We add a parameter-relative Jacobian diagnostic with D=diag(|theta|), solving in "
        "coordinates JD and mapping the step back with D. Exactly zero coordinates remain "
        "frozen. It avoids dividing by small calibration derivatives and remains equivariant "
        "to the positive diagonal transformations tested here. Relative parameter scaling and "
        "rescaling-aware optimization are established ideas [2,3]; no optimizer novelty is claimed.",
        r"$$ \Delta\theta=D(JD)^T[(JD)(JD)^T+\lambda I]^{-1}r. $$",
        "After rescaling, mean disagreement with the edited original function under the long "
        "symbolic schedule was "+gauge_text+". Accuracy comparisons remain those in Table 2; "
        "representation stability alone is not evidence of physical correctness.",
        "![Construction cost and error](../results/deployment-heldout/figures/cost_accuracy.png)",
        "Figure 4. Mean construction time versus physical error. Points connect each method's "
        "short and long schedules. Timings exclude source-model training and are measured on "
        "one CPU configuration with some overlapping background experiment jobs. They are "
        "descriptive observations, not isolated latency benchmarks or universal complexity estimates.",
        "### 7.4 Strengthening query distillation with free synthetic labels",
        "Sparse student supervision is not the strongest use of the shared physics prior. "
        "Once the query baseline identifies a symbolic teacher, it can evaluate that teacher "
        "at new inputs without spending additional base-model queries. We added a separate "
        "control protocol in Git commit "+dense_manifest["git_commit"][:7]+" before reading "
        "held-out editing metrics. It reuses the 24 held-out checkpoints and generates 4,096 "
        "fresh symbolic labels per model. Warm and fresh students share inputs, targets, "
        "minibatch order, and a learning-rate schedule. They use 600 or 2,400 minibatch "
        "Adam updates of size 256, yielding 96 additional rows.",
        dense_text,"\n".join(dense_table),
        "Table 3. Dense symbolic supervision at the same 128-query budget. All students retain "
        "the original architecture. Source-model training is excluded from construction time; "
        "that exclusion is appropriate only when the pretrained source is already available.",
        "The known-library setting still permits deployment of the much smaller symbolic "
        "model if the architecture constraint is relaxed. That option must remain in the "
        "comparison: a rigid deployment contract is an application assumption, not an intrinsic "
        "reason to prefer weight editing. The held-out evaluation covers new coefficient draws "
        "and network seeds, not new physical mechanism families."
    ])
    text=source.read_text(encoding="utf-8")
    text=text.replace("# Before Programming the Weights: Structured Query Baselines for Editing Learned Dynamics",
        "# What Does Weight Access Buy? Targets, Initialization, and Representation in Dynamics Editing",1)
    text=text.replace("**Exploratory research draft - November 10, 2026.** This document reports a completed feasibility study, not a submission-ready paper. Author list and publication venue are intentionally unset. No novel optimization algorithm or LLM capability is claimed.",
        "**Working manuscript, version 2 - November 10, 2026.** Two controlled numerical studies "
        "with held-out model evaluation. Submission readiness and publication novelty remain under review.")
    text=text.replace("These observations motivate stronger behavioral baselines",f"A second study "
        f"on 24 held-out models finds warm versus fresh L-BFGS errors of {warm:.4f} versus {cold:.4f} "
        f"under a fixed architecture and sparse supervision. With free dense symbolic labels, "
        f"warm and fresh students obtain {dw:.4f} and {dc:.4f} under the longer Adam schedule. "
        "These observations motivate stronger behavioral baselines",1)
    text=text.replace("The study makes three limited contributions: an executable comparison with shared semantic information; a paired test of editing under equivalent network representations; and a decision about whether this particular benchmark warrants further investment. The algebra below explains the expected behavior of the baselines. It is elementary and is not presented as new theory. Publication novelty remains to be established.",
        "We separate three questions: whether a structured query teacher solves the intervention, "
        "whether pretrained weights improve same-architecture realization, and whether an edit "
        "depends on arbitrary parameter representation. The artifact contains both an initial "
        "screening sweep and a frozen follow-up on new model seeds. Elementary error identities "
        "expose the assumptions of these comparisons; they are not claimed as new theory.")
    text=text.replace("## 7. Interpretation and limitations",section+"\n\n## 8. Interpretation and limitations",1)
    text=text.replace("## 8. Research decision and next experiment","## 9. Research decision and next experiment",1)
    start=text.index("## 9. Research decision and next experiment")
    end=text.index("## Reproducibility and disclosure",start)
    decision=("The evidence supports a narrower interpretation of weight access: pretrained "
        "parameters can provide useful initialization for realizing an intervention in a fixed "
        "network. The fresh-versus-warm contrast does not establish semantic programming of "
        "weights. Structured query teachers, target objectives, and synthetic supervision "
        "remain essential baselines. A claim about language-model weight programming would "
        "require a language-model editor and capabilities beyond these numerical procedures.")
    text=(text[:start]+"## 9. Research decision and next experiment\n\n"+decision+"\n\n"+
        "A next experiment should vary mechanism uncertainty or transfer to a new architecture, "
        "while retaining fresh and warm numerical competitors. Increasing the number of toy "
        "runs alone would not resolve novelty. For an LLM study, freeze the access arms, prompts, "
        "token and query budgets, and evaluation models before testing. Separate numerical "
        "weight output from an agent that writes and executes adaptation code.\n\n"+
        "The current draft provides a reproducible methodological case study. It still needs "
        "a finding beyond expected initialization benefits and known optimization geometry "
        "before making a strong publication claim.\n\n"+text[end:])
    text=text.replace("Twelve seed clusters characterize variability within this protocol rather than the population of all world models.",
        "Each main study uses 12 seed clusters, with disjoint seed ranges; uncertainty describes "
        "these protocols rather than the population of all world models.")
    text=text.replace("No paid LLM APIs or rented accelerators were used.",
        "The follow-up adds 24 held-out models, 2,688 factorial measurements, and 96 dense-control "
        "measurements. Its development results are kept separate. No paid LLM APIs or rented accelerators were used.")
    related="\n\n**Distillation and initialization.** Distillation transfers teacher behavior to a "
    related+="student [9]. Prior diffusion work also separates teacher supervision from initialization "
    related+="[10]. We cite the explicitly versioned 2025 report for that analysis; the arXiv work "
    related+="was subsequently revised. An initialization benefit is therefore not a new general "
    related+="principle. Our factorial comparisons test how it affects the interpretation of a "
    related+="specified dynamics-editing task.\n\n"
    text=text.replace("## 3. Problem and access contracts",related+"## 3. Problem and access contracts",1)
    text += "\n[9] Hinton, G., Vinyals, O., and Dean, J. (2015). Distilling the Knowledge in a Neural Network. https://arxiv.org/abs/1503.02531\n\n"
    text += "[10] Zhang, M., He, J., Chen, W., Ou, Z., Hernandez-Lobato, J. M., Schoelkopf, B., and Barber, D. (2025). Towards Training One-Step Diffusion Models Without Distillation. Version 1 technical report; later revised under a different title. https://arxiv.org/abs/2502.08005v1\n"
    source.write_text(text,encoding="utf-8")
    report=["# Follow-up decision and dense-control results","",decision,"",primary_text,"",dense_text,"",
            "## Dense symbolic distillation","","\n".join(dense_table),"",
            "The primary sparse-supervision contrast was frozen before held-out evaluation. "
            "The separately committed dense control strengthens the query-only comparator. "
            "Neither comparison is a language-model evaluation."]
    (dense/"INTERPRETATION.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    (root/"docs/decision.md").write_text("# Updated research decision\n\n"+decision+"\n\n"+
        primary_text+"\n\n"+dense_text+"\n\n"+
        "**Status:** stronger empirical draft, with publication novelty still unestablished. "
        "Prioritize unknown-mechanism or architecture-transfer questions over enlarging this "
        "known-library oscillator benchmark.\n",encoding="utf-8")
    audit=dict(heldout_source_snapshot="passed",dense_source_snapshot="passed",
        heldout_rows=len(rows),dense_rows=len(drows),deployment_parameters=1217,
        primary_contrast=primary,dense_warm_minus_fresh=dense_contrast,
        nonfinite_rollout_rows=int(sum(r["rollout_nonfinite"] for r in rows+drows)),
        dense_nonfinite_metrics={m:int(sum(not np.isfinite(r[m]) for r in drows))
                                for m in ["id_rmse","ood_rmse","teacher_fit_rmse","rollout_rmse"]})
    (dense/"audit.json").write_text(json.dumps(audit,indent=2,allow_nan=False))
    return source
