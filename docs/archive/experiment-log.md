# Experiment log

## 2026-11-10: initial implementation

- Eight numerical/physics checks passed before executing the smoke experiment.
- Smoke: 2 seeds, width 16, 400 training steps, 8/32 queries. Completed 756 rows
  in 18.7 seconds. This run had seven methods and predates the Adam comparator.
- Smoke output showed query correction competitive with numerical weight edits.
  It is a development result and excluded from main uncertainty estimates.
- Added a standard full-parameter Adam comparator (150 full-batch steps, lr=.003)
  before the main run; avoids comparing weight editing only to closed-form edits.
- The first attempted large run (`results/pilot`) was interrupted by the session
  cancellation after one model. Preserve its partial output; do not analyze it as
  a complete experiment. The complete run is `results/main`.
- Main config: 12 seed clusters, three systems (two matched, one misspecified),
  width 32, 2400 training steps, 8/32/128 queries, three interventions, three
  equivalent representations, eight comparators. Fixed before reading main results.
- No LLM API requests or paid compute have been used.

## Main run and audit

- `results/main`: complete; 36 trained networks, 7,776 measurements, 354.8 seconds
  on local CPU. No nonfinite rollout rows. Base symmetry discrepancy <=1.33e-15.
- Matched-library, original representation, 128-query acceleration RMSE:
  query residual 0.01492; query symbolic 0.005622; Jacobian ridge 0.01743;
  head ridge 0.02660; normalized Jacobian 0.04064; Adam fine-tuning 0.02137.
- These exploratory results do not justify scaling the original demonstration
  framing. See the generated decision memo and manuscript for qualifications.
- Re-executed seed 3 / Duffing / 32 queries / damping intervention from training
  through evaluation, across all eight methods and three representations.
  All 192 numerical values in the 24 rerun rows matched the main run exactly.
- Added Adam to the representation figure after main analysis so the display
  includes every non-oracle weight editor. This is a presentation change only.
- The normalized Jacobian is representation-stable here but has poorer mean
  physical error than the unnormalized Jacobian. Equivariance alone is not
  sufficient for accuracy; this is not claimed as a new scientific principle.

## Scope decisions

- The first draft will characterize a pilot, not claim a novel algorithm.
- All methods receive a low-dimensional physical library. Good query baseline
  performance therefore does not refute weight access under unknown mechanisms.
- Equivalent representations and state samples are repeated measurements, not
  independent statistical units. Bootstrap independent training seeds.
- Rollouts assess prediction; they do not establish planning or control utility.

## Same-architecture follow-up development

- Added a crossed experiment for residual versus symbolic teachers, warm versus
  fresh initialization, two numerical schedules, and original/rescaled models.
- Added L-BFGS as a stronger ordinary distillation comparator and a relative-weight
  Jacobian diagnostic. The latter is explicitly not claimed as novel optimization.
- All 12 implementation checks passed before development evaluation.
- Development seeds 0-1: 448 measurements, 91.1 CPU seconds. Warm L-BFGS had lower
  mean physical error than fresh L-BFGS for the fixed primary setting. Symbolic
  teacher targets did not automatically improve same-architecture student errors.
- No solver hyperparameters were changed after this development result. Freeze
  the code/protocol in Git before executing held-out seeds 20-31. The held-out
  comparison is a new-model evaluation within these two system families, not
  generalization to an unseen physical mechanism family.

## Held-out same-architecture evaluation

- Frozen runtime/protocol commit: `ecc0abf`. Seeds 20-31, two systems, 24 new
  source models, 2,688 condition-method rows. Completed in 539.2 CPU seconds.
- Fixed primary comparison: at 128 queries with symbolic targets and the long
  L-BFGS schedule, warm physical RMSE 0.01953 versus fresh 0.03773. Paired
  warm-minus-fresh difference -0.01820, 95% seed-cluster interval
  [-0.02255, -0.01401]. This is an initialization comparison, not semantic WSP.
- Warm L-BFGS physical RMSE was 0.01750 on residual targets versus 0.01953 on
  symbolic targets. Better unrestricted teacher accuracy did not guarantee a
  better student under this sparse schedule.
- Before reading those editing metrics, separately committed a dense symbolic
  control (`64bee21`) because a fitted library permits free synthetic labels.
- Dense-control development on seeds 0-1 completed 16 rows in 24.7 seconds.
  Both warm and fresh students improved with more updates. No hyperparameter
  changes followed development; evaluate seeds 20-31 with the fixed schedules.
- All 12 numerical and access-isolation checks pass.

## Dense held-out control and draft revision

- Dense evaluation: 96 rows on the 24 held-out checkpoints, 186.9 CPU seconds.
  At 2,400 updates, physical RMSE was 0.01752 for a fresh student and 0.01189
  for warm initialization. The same 128 base queries supplied both; 4,096
  symbolic training labels required no further base-model access.
- Fresh dense training is stronger than the sparse fresh comparator. Therefore
  the sparse L-BFGS result alone must not be treated as a general query-only
  limitation. Cross-schedule comparisons change compute and optimizer as well.
- Independently reran six factorial rows and four dense-control rows; all 80
  compared numerical metrics matched exactly. Four archived source hashes pass.
- Canonical manuscript revised to include both studies, target/initialization
  distinctions, dense query control, paired uncertainty, and additional references.
- Total main measurements: 7,776 + 2,688 + 96 = 10,560. Development and reproduction
  rows remain separately labeled and are excluded from main effect estimates.

## Cycle 3 development and freeze

- Added matched-calibration-fit and longer-optimization controls, plus a two-family
  identification study varying query placement and count. See cycle3-protocol.md.
- All 14 implementation checks passed before evaluation.
- Development seeds 0-1: four source models, 32 fitting checkpoints, 72 library
  identification measurements; 61.3 seconds on CPU. Both initialization types
  crossed calibration RMSE 0.002 in all four conditions. The held-out error gap
  persisted at threshold crossing and at the fixed 1,920-iteration budget.
- On development models, broad queries selected the correct family consistently;
  narrow queries were unreliable even with larger query counts. This motivates
  the fixed placement-versus-count contrast, not a claim of arbitrary mechanism
  discovery.
- No hyperparameters changed after development. Commit the protocol, driver,
  analysis, and tests before evaluating the fresh seed range 40-51.

## Cycle 3 fresh-seed results

- Runtime/protocol freeze: `9349874`. Seeds 40-51 completed 24 new source models,
  192 fitting checkpoints, and 432 identification conditions in 288.6 CPU seconds.
- Every warm and fresh student crossed calibration RMSE 0.002. At first crossing,
  mean physical errors were 0.02079 warm versus 0.03245 fresh; paired difference
  -0.01166, 95% seed-cluster interval [-0.01509, -0.00833]. All 24 model conditions
  are paired. The gap also persists at the fixed 1,920-iteration budget.
- Family-hidden query comparison: 64 mixed queries RMSE 0.007175 versus 0.19843
  for 256 narrow queries; paired difference -0.19126 [-0.24611, -0.13695]. The
  known-family comparator still has error 0.12003 with 256 narrow queries.
- Broad 64-query selection is correct for all 24 observed models. A supplementary
  exact interval for joint success on both families across independent seeds
  avoids treating an all-success percentile bootstrap as population certainty.
  This uncertainty presentation was added after seeing counts; the primary
  RMSE contrasts and fitting thresholds are unchanged.
- Re-executed seed 43 independently: 16 fitting rows, 36 identification rows,
  and 256 trace rows. All 1,200 checked numerical cells reproduce exactly.
- Draft revised to version 3 with the new controls and literature on established
  initialization/generalization effects. No new optimization or identification
  algorithm is claimed. No APIs or rented accelerators were used.

## Cycle 4 development and freeze

- New task: actual noisy observations from the changed environment, with no
  mechanism library or shift formula supplied to adapters. Source predictions
  remain callable. This is distinct from the earlier specification-only task.
- Four source systems, including new contact and state-dependent damping models.
  Added output-only linear/cubic/kernel/neural residuals and a frozen learned-head
  comparator. ARD kernel lengths were added before development execution.
- All 18 checks passed, including analytic leave-one-out validation, exact head
  folding, callable-only kernel access, and new-shift/source label separation.
- Development seeds 0-1: eight source models, 288 rows, 118.3 CPU seconds.
  For nonlinear shifts at 64 observations, warm/full RMSE 0.08433, fresh/full
  0.08414, isotropic RBF 0.07152, ARD RBF 0.03417, neural residual 0.05667.
- No solver or kernel hyperparameters changed after development. Freeze the
  runtime, protocol, analysis, and checks before seeds 60-71. Isotropic RBF
  comparisons remain the primary contrasts; ARD results are an additional
  prespecified numerical control, not a replacement primary chosen after results.

## Cycle 4 fresh-model results and version 4 manuscript

- Freeze commit `5edaad8`; seeds 60-71, four source systems, 48 new source
  networks, 1,728 method-condition rows; 715.6 wall-clock seconds on CPU.
- Added-mechanism shift, 64 noisy observations: source-only RMSE 0.29823;
  warm full 0.08559; fresh full 0.08936; learned head 0.13380; isotropic residual
  0.07406; coordinate-scaled residual 0.03723; neural residual 0.06752.
- Primary isotropic residual minus warm full contrast: -0.01153, 95% seed-cluster
  interval [-0.02128,-0.00181]. The additional coordinate-scaled kernel contrast
  is -0.04836 [-0.05786,-0.03934]. These are descriptive, unadjusted intervals.
- Kernel and neural corrections retain the callable source at inference; their
  performance must not be interpreted as equal-architecture superiority.
- Added a parameter-free-source-access check for the fresh neural residual;
  all 19 implementation checks pass. No runtime hyperparameters changed.
- Final independent rerun, after compilation checks, uses seed 63/contact and
  five methods for both shifts at 64 labels. All 140 checked values reproduce
  exactly; source snapshot hashes pass. The preliminary development rerun stays
  local and is excluded from the release evidence.
- Rewrote the main manuscript around four stages and two distinct tasks, rather
  than comparing specification-editing and observed-shift errors as one task.
  Earlier draft versions and all prior raw results remain available in Git.
- Kernel regression and residual dynamics prior work were added to the novelty
  audit. These baselines are established methods; no new residual algorithm or
  LLM capability is claimed.

## Actual LLM WSP pilot: development and freeze

- Prior cycles are numerical adaptation baselines and do not answer LLM WSP.
  Added an independent API arm receiving architecture, raw weights, and an edit
  request; no tools, probes, forward passes, target data, or feedback.
- Available model IDs checked through Anthropic's API. Selected Opus 5.5 with
  adaptive thinking, medium effort, 4,096 output tokens. Official standard
  rates checked at $4 input/$20 output per million tokens. Shared $3 cap covers
  development and pilot generations; credentials are never archived.
- All 21 implementation tests pass. Development source seed 89: one action-gain
  call completed normally, returned 12 numerical changes, and passed held-out
  functional evaluation (NRMSE about 4e-16). Estimated cost $0.028552.
- No prompt changes followed that transport/format check. Main pilot uses
  source seeds 90-91, dense/spare capacity, original/transformed representation,
  and three tasks, totaling 24 stateless calls. Fixture hashes and prompt/code
  are committed before the main requests. No evaluated failures are retried.

## Direct WSP results and independent-source follow-up

- Primary 24 calls completed normally: 24 valid patches; action gain 8/8,
  known drag 8/8, semantic damping 0/8.
- Before inspecting primary outcomes, froze a six-new-source follow-up in 3473d63
  and sealed its 12 fixtures before generation. Same prompt/model settings, spare
  capacity and transformed representation; action gain 6/6, semantic 1/6.
- Total 36 evaluated calls represent eight independent source functions. No failed
  generations were retried, repaired, or omitted. Total estimated list-price usage
  including development was $1.275608 under the shared $3 cap.
- All semantic edits accurately compiled their own coefficient estimate (maximum
  own-estimate NRMSE 6.89e-7). Post hoc, seven of eight source-level estimates were
  within 0.01 of the network's local slope, which differed from the physical c.
- Common-condition coefficient MAE: LLM 0.15102; query fit 0.07550; prior 0.14373;
  local slope 0.15782. No statistical superiority claim. Source 97 is poorly fit
  and even its query estimate is biased; retained in full.
- Both primary learned-width sources contain inactive units; additive edit success
  does not demonstrate editing with fully occupied capacity. OOD errors reported.
- Replaced current manuscript with a six-page WSP draft; preserved the previous
  manuscript as paper/numerical-baselines.md. Novelty and identification validity
  remain open rather than claimed resolved.
