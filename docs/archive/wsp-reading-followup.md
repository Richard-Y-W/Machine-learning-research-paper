# Additional independent-network reading cohort

Specified while the first 24-case API pilot was in progress and before inspecting
its editing outcomes. No prompt or model setting changes.

Use six additional independently trained sources, seeds 92-97, with spare units
and permuted/rescaled representation. Each gets action-gain and semantic-damping
requests, totaling 12 additional calls if the shared $3 ledger permits them.
This tests coefficient reading on more than two underlying source functions.
It is reported separately from the original capacity/representation pilot.

Compare reported damping estimates to hidden physical coefficients, the physical
family fit from 128 source queries, and a benchmark-prior constant estimate 0.4.
The latter is given the benchmark distribution and is explicitly a prior-only
reference, not a fair description of the model's available prompt information.
Functional compilation of the model's own estimate is measured separately.

The trained network is an approximation to its physical generator. A coefficient
estimate can therefore differ from the generator even when it describes the
network reasonably well. Report source-query coefficient estimates and source
approximation errors rather than attributing every discrepancy to reasoning.

All model calls remain tool-free, data-free at inference, stateless, and without
feedback. No failed call is regenerated. The original pilot and this cohort use
the same global spending cap, including the development call. Public prompts,
final patches, usage, and evaluator outcomes are archived.
