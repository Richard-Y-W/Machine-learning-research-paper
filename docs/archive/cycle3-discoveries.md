# Cycle 3 discoveries

## 1. Fitting versus generalization

The calibration RMSE threshold of 0.002 was attained by 24/24 warm and 24/24 fresh students. At the first crossing, warm and fresh physical errors were 0.0208 and 0.0324; the paired warm-minus-fresh difference was -0.011658 [-0.015093, -0.0083289] across 24 jointly attained model conditions. Thresholds are evaluated only every 30 optimizer iterations, so this is approximately matched fitting rather than exact equality of training losses. If attainment is incomplete, this contrast is conditional on the jointly attained subset.

Mean L-BFGS iteration counters at first crossing were 587.5 for warm and 542.5 for fresh initialization. A lower held-out error therefore need not mean fewer iterations to a specified calibration loss. This distinction was obscured by the previous short-budget comparison.

## 2. Query placement versus count

With the true family label withheld, 64 mixed queries gave mean changed-dynamics physical RMSE 0.0072, compared with 0.1984 for 256 narrow queries. The paired mixed-minus-narrow contrast was -0.19126 [-0.24611, -0.13695]. Broad 64-query calibration selected the correct family in 24/24 observed source models. This is an empirical count from two candidate families, not a guarantee for other mechanisms or future models.

Knowing the true family still leaves RMSE 0.1200 with 256 narrow queries. Coefficient identification, not just family selection, can fail under concentrated inputs.

## Scope

These are controlled findings on two supplied physical families and small networks. They strengthen the experimental diagnosis; novelty remains a separate question. They do not test LLM weight interpretation or discovery of arbitrary dynamics.
