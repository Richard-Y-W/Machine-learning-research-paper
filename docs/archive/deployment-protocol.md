# Follow-up: architecture, targets, and initialization

Written on 2026-11-10 before running this follow-up. Exploratory protocol, not a
registered confirmatory study. The first experiment's seeds 0-11 are development
data. New seeds 20-31 are reserved for the frozen follow-up evaluation.

## Why change the experiment?

The first experiment has an important objective difference: residual targets retain
base approximation errors; symbolic targets reconstruct the supplied physics. Its
ordinary fine-tuning arm is already warm-started distillation on residual targets.
Calling that arm something new would not constitute a new experiment.

We therefore cross teacher objective with initialization and numerical solver.
Every deployable student is the same 3-32-32-1 ReLU network (1,217 parameters).
The query residual and symbolic functions are teacher references only. A fresh
student is the actual query-only deployment comparator; copying pretrained
parameters explicitly requires weight access and is classified as warm-starting.

## Fixed grid

- Two matched systems: Duffing and pendulum. Misspecification remains in the first
  experiment and is not pooled into this matched-teacher comparison.
- Combined intervention, multiplying first restoring coefficient by 1.5 and
  damping by 2. This single fixed intervention avoids a large redundant sweep.
- Calibration budgets 32 and 128, from identical nested random query sets.
- Teachers: residual f(x)+phi(x) M c_hat; symbolic phi(x)(I+M)c_hat.
- Computational schedules: short and long. Adam uses 150 or 600 full-batch steps;
  Jacobian methods use 3 or 12 iterations; L-BFGS uses at most 30 or 120
  iterations with strong-Wolfe line search. These are schedules, not matched FLOPs.
- Warm Adam, cold Adam, warm L-BFGS, cold L-BFGS, Euclidean Jacobian, column-normalized
  Jacobian, and parameter-relative Jacobian. All share the same teacher targets.
- Representations: original and positive ReLU rescaling. Cold students always
  have the same random initialization independent of teacher representation, so
  teacher rescaling cannot give them parameter information. For warm methods,
  the pretrained model is transformed before editing. Calibration outputs remain
  shared from the original representation.

The parameter-relative Jacobian uses J diag(|theta|) in the damped linear solve,
and maps the update back with diag(|theta|). Exactly zero coordinates are frozen.
It is a scale-aware numerical baseline, not a claimed novel optimizer. Unlike
column normalization, it does not divide by tiny nonzero calibration derivatives.
We make no prediction that invariance guarantees accuracy.

## Measurements

Physical acceleration RMSE, extrapolation RMSE, held-out teacher-fitting RMSE,
100-step rollout RMSE, training-target RMSE, representation disagreement, actual
construction time, forward evaluation points, gradient evaluation points, and
Jacobian output points. Every deployed student has exactly 1,217 parameters.
Two hundred fifty-six prediction points are used for latency measurement after
warmup, with repeated calls; teacher-reference latency includes its wrapper.
Latency repetitions are timing measurements, not independent scientific units.

The primary contrast is warm versus cold L-BFGS for symbolic targets at 128
queries under the long schedule and original representation. This asks whether
the existing weights provide useful initialization under a fixed numerical
procedure. It is not evidence of semantic interpretation of raw weights.
Secondary contrasts: residual versus symbolic targets within each solver, and
rescaling disagreement versus physical accuracy for the three Jacobian methods.
Average families within each independent seed; bootstrap seed clusters for
descriptive 95% intervals. Retain every condition and report failures explicitly.

## Development and freeze

Run a small development grid on seeds 0-1 before the held-out run. Any numerical
changes are logged. Freeze solver hyperparameters before executing seeds 20-31.
No held-out tuning or post-hoc replacement of a failing method. The main outcome
may still be that this setting lacks publishable novelty.
