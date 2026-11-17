# Cycle 3: matched fitting and hidden-family identification

Written before executing cycle 3. Previous seeds 0-31 are development evidence.
The new evaluation uses seeds 40-51, two systems, and 24 independently trained
source networks. This is a frozen exploratory follow-up, not a preregistration.

## A. Does more fitting remove the initialization advantage?

Use the symbolic teacher inferred from 128 broad source-network queries and the
fixed combined intervention from the previous cycle. Both students retain the
1,217-parameter architecture. Warm initialization copies the source network;
fresh initialization uses the independent initialization rule already implemented.

Run L-BFGS for up to 1,920 iterations, preserving optimizer history between
30-iteration chunks. Record fixed checkpoints at 120, 480, and 1,920 iterations,
and the first chunk whose calibration teacher-fit RMSE is <=0.002. The threshold
uses calibration data only. Continue after crossing so fixed-budget comparisons
remain available. Numerical convergence may stop updates before the iteration
cap; record actual accepted iterations and closure counts.

Primary diagnostic: warm/fresh held-out physical and teacher-fitting errors at
the first threshold crossing, together with threshold attainment rates. If either
student fails to cross, retain the failure and report matched-subset estimates
separately; do not treat successful subsets as an unconditional causal estimate.
This threshold comparison reduces fitting mismatch but does not prove a unique
mechanism for any remaining initialization effect. The fixed-budget curves guard
against selecting a favorable convergence checkpoint after seeing test results.

## B. What if the query baseline is not told the correct family?

Offer a choice between the Duffing and pendulum four-term libraries. The true
family label is withheld from the selection routine; candidate libraries, state
semantics, and coefficient-level intervention semantics are still supplied.
This is discrete family uncertainty, not discovery of arbitrary unknown physics.

Select a family by fixed four-fold cross-validation on source-network query
outputs, then fit its coefficients on all queried points. Compare with an oracle
family-label baseline that is still required to estimate coefficients from the
same queries. This is an oracle only for the family label, not for coefficient
values or changed-environment behavior.

Query budgets: 16, 64, 256. Designs: narrow uniform cube [-0.25,0.25]^3; broad
uniform cube [-1.5,1.5]^3; and mixed, with eight narrow queries followed by broad
queries. Designs and folds are fixed by seed. Broad queries are evaluations of
the learned source model, not new interactions with the physical environment.
Primary contrast: 64 mixed queries versus 256 narrow queries for the selected
library's changed-dynamics physical RMSE. This explicitly tests whether placement
can matter more than count under the supplied candidate prior.

Report family selection accuracy, held-out acceleration and extrapolation errors,
coefficient-library condition number, source-query approximation error, and
cross-validation errors. Use one-step errors in this identification study;
do not make control or planning claims. All budgets/designs and both family-access
conditions remain in the raw output.

## Analysis and integrity

Average system families within each independent seed, and bootstrap 12 seed
clusters with 5,000 resamples for exploratory 95% intervals. Record all source
checkpoints, raw curves, source snapshots, manifests, convergence failures, and
nonfinite values. Re-execute a subset independently. No tuning on seeds 40-51.
No language-model requests are involved in this cycle. All optimizer variants,
family-selection rules, and query designs must pass implementation checks before
the frozen evaluation is launched.
