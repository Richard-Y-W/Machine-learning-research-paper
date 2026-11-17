# Dense symbolic distillation control

Added while the frozen same-architecture held-out run was in progress, before
reading its editing metrics. The original primary contrast remains unchanged.
This is a separately logged strengthening of the query baseline.

With a known feature library, fitting coefficients from 128 queries defines a
symbolic teacher everywhere. Restricting student supervision to the 128 queried
inputs would unnecessarily restrict a query-only method. We therefore generate
4,096 fresh synthetic inputs, label them using only the fitted symbolic teacher,
and compare warm and fresh students under the identical architecture.

The calibration set and feature library match the deployment experiment. No
true physical coefficient values, changed-environment labels, or additional
base-model queries are available to the optimizers. Synthetic labels do not count
as base-model queries. State semantics and the supplied library remain strong
shared assumptions.

Fixed schedules: 600 and 2,400 Adam steps, minibatch size 256, initial lr 0.003
reduced fivefold after 75% of updates. Both initializations receive the same
synthetic dataset and minibatch order. The original source network is used for
the warm initialization; fresh initialization is independent of its weights.
Only the original representation is tested in this additive control.

Use seeds 0-1 for implementation development and 20-31 for evaluation. Reuse
the corresponding stored base checkpoints; do not retrain or select them.
Measure physical RMSE, teacher fit, extrapolation, rollout error, and construction
time. The comparison is an accuracy/cost curve, not a claim of equal end-to-end
training compute. Original base-model training is a sunk cost only if it already
exists for another purpose. Report this control even if it eliminates the
apparent advantage of weight access under sparse supervision.
