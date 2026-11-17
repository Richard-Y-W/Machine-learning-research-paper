# Research direction: why direct weight editing works

Updated November 12, 2026. This replaces the earlier size-comparison proposal.
The transformer study is proposed; the completed evidence remains the small
dynamics pilot.

## Central question

**Why does direct weight editing work in pretrained transformers, and how do the
model's representations, the requested change, and the editor's decisions determine
whether the change generalizes while preserving other capabilities?**

The question stays broad. The study should distinguish useful diagnosis, the
construction of a weight change, interference with other computations, and the
contribution of feedback and search. A location-hint experiment is one diagnostic
within this study, not its whole purpose.

The [TDD implementation plan](2026-11-12-why-weight-editing-works.md)
defines the hypotheses, model and task coverage, software contracts, tests,
execution stages, and limits on interpretation.

## Intended evidence

Use controlled repairs to establish when a repair exists and to test interventions
with known causes. Use mathematical problem solving, factual updates, and
instruction following to test useful changes beyond artificial damage. Freeze
explanations and predictions before testing them on untouched tasks and models.

Develop the harness and explanations on Qwen3-1.7B. Test them on Qwen3-8B and
Llama-3.1-8B-Instruct, with Qwen3-32B as a later scale test when the execution budget
supports it. Each checkpoint must have a recorded revision, environment, and
evaluation protocol. Released model sizes also differ in training and architecture;
their comparison alone does not identify the causal effect of parameter count.

Keep the editor and access conditions fixed within a paired comparison. Allow
tensor inspection and numerical changes with metered forward feedback; prohibit
gradients and fitting during editing. Evaluate unseen target examples, specified
protected behavior, and cost. Count invalid attempts, no-change choices, and
budget stops. Repeated editor attempts are not independent pretrained models.

Software tests establish that the experiment records these comparisons correctly.
They do not require any research hypothesis to succeed.

## Contribution boundary

The central question is established, and novelty is not claimed for the question
itself. [Hu et al.](https://arxiv.org/html/2611.09382v1) already study direct editing
across transformer sizes and tasks. [Hase et al.](https://arxiv.org/abs/2301.04213)
study the relationship between localization and editing, and
[Sutton et al.](https://arxiv.org/abs/2406.12670) connect representation geometry to
editability for specific editing methods.

The intended contribution is a supported explanation that distinguishes competing
causes and predicts outcomes under a declared editing protocol. Its strength will
depend on the evidence. Artificial repairs alone do not establish an explanation
of natural editing, and a failed search does not establish that a model is
intrinsically uneditable. The plan records additional directly relevant sources.

## Scope and execution status

The old fine-tuning, distillation, Jacobian, and residual baseline suite is outside
the active study. Include only comparisons needed to distinguish an explanation or
support a specific comparative claim. The dynamics pilot supplies motivation for
separating a chosen change from its execution; extending it is not a prerequisite.

Implement and verify the offline harness before running the pretrained-model
study. A live run requires a concrete manifest specifying checkpoint revisions,
data hashes, editor identity, hardware, replication, feedback limits, and a total
spending cap. The old pilot's $3 ledger is not the budget for this study.

No transformer checkpoint has been downloaded or edited, and no new paid editor
call has been made for this plan. The existing pilot manuscript remains a report
of completed work; transformer results must not be written into it before they
exist.
