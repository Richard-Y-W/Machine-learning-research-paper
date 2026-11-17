# Exploratory protocol, 2026-11-10

Written before the first experiment execution. This is a dated exploratory
protocol, not a preregistration. Changes after observing results must be logged.

## Question and scope

Does direct numerical access to a learned acceleration network offer a practical
advantage for specified multiplicative changes in physical mechanisms, relative
to a query-only correction with an explicit low-dimensional physics prior?

This first experiment does NOT test LLM weight-space programming. It tests whether
these tasks have enough headroom over simple system identification to justify
spending an API budget. Full weight access subsumes executable model queries;
there is no information-theoretic separation without additional restrictions.

## Systems

State is position q and velocity v; input u is force. dq/dt=v and dv/dt=f(q,v,u).
Two matched mechanism libraries are used: Duffing [-q,-v,-q^3,u], and pendulum
[-sin(q),-v,-v|v|,u]. A third Duffing condition adds 0.3 sin(3q), deliberately
outside the supplied library. It is a misspecification control, not a matched task.
Physical coefficients vary deterministically by independent seed. Every family
gets an independently trained, two-hidden-layer ReLU MLP. Training supervision
is noiseless acceleration at sampled states/actions, not rollout observations.
The state meanings, signs, feature library, and intervention are supplied to all
methods. Physical coefficient values are withheld from all non-oracle editors.

## Intervention and access

Three fixed interventions: damping x2; first restoring coefficient x1.5; both.
All deployed changes are computed from the same least-squares coefficient fit
to n base-model queries (n=8,32,128). Synthetic edit targets are base predictions
plus the fitted semantic correction. No measurements of the changed environment
are supplied. This is specification editing, not adaptation from observed shifts.

Methods:
- Unedited model.
- Query residual: add the fitted correction as a four-feature wrapper. Retains
  the base network and adds a structured path. Requires base execution at runtime.
- Query symbolic replacement: replace the base by the fitted, intervened library.
  Explicitly has a different inductive bias and deployment architecture.
- Head ridge: fit an additive output-head edit from hidden activations.
- Full Jacobian ridge: three damped, linearized weight updates with line search.
- Column-normalized Jacobian ridge: same procedure with Jacobian column scaling.
- Oracle residual: reference using true coefficients; excluded from competitions.

The ridge factor is fixed at 1e-4 times mean kernel diagonal. Normalized columns
with exactly zero calibration norm are frozen. These are numerical diagnostic
baselines, not proposed novel algorithms. No hyperparameter sweep on test points.

## Reparameterization checks

Original network, random hidden-neuron permutation, and positive ReLU hidden-unit
rescaling (log-uniform factors 0.1 to 10). The forward function is checked before
each edit. The same calibration and test sets are used for each representation.
Methods are rerun on each representation; no raw edit vector is copied between
networks. Gauge disagreement is measured against the original representation's
edited predictions, in function space.

## Metrics and statistical unit

Held-out in-domain acceleration RMSE; extrapolation RMSE outside the training
cube; normalized edit discrepancy (realized functional change minus requested
true change); collateral discrepancy on the intervention's zero-effect set;
and 100-step semi-implicit Euler rollout RMSE at dt=0.03. Rollouts compare against
the identically discretized intervened analytic system. This is prediction, not
closed-loop planning. Nonfinite rollouts are counted, never silently dropped.
Original-versus-reparameterized edited prediction disagreement is also recorded.
Per-condition rows are retained. Aggregate uncertainty uses a paired bootstrap
over independent seed clusters, averaging families/interventions within a seed.
Equivalent parameterizations are not independent replications. Report matched
and misspecified libraries separately.

## Go/no-go

A go decision requires more than beating the unedited model: an advantage over
the structured query baseline under a useful deployment constraint, or a stable
failure mechanism that predicts held-out results. If the query baseline solves
the matched setting, do not claim a need for LLM weight editing. If all findings
follow immediately from classical regression/optimization, treat the outcome
as a benchmark sanity check rather than a publishable contribution.

## Same-day deliverable

Runnable code, checks, raw measurements, plots, a preliminary manuscript, and an
explicit assessment of whether this direction merits a multiweek investment.
Submission readiness, novelty, and publication are not promised.
