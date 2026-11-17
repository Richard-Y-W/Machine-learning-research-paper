# Cycle 3 findings

24 new source models; 192 fitting checkpoints; 432 identification measurements; 4.81 minutes on CPU.

Intervals are exploratory 95% seed-cluster bootstrap intervals; families are averaged within each seed. Threshold crossings are selected using calibration error only.

## 1. Longer optimization and similar calibration fit

Calibration threshold <=0.002 attained by 24/24 warm and 24/24 fresh students.

At first threshold crossing, warm minus fresh physical RMSE on the jointly attained subset: -0.011658 [-0.015093, -0.0083289]; 24/24 paired model conditions. All evaluated model conditions enter this contrast; it does not uniquely separate optimization path from initialization-dependent solution bias.

At the full fixed 1,920-iteration budget, warm minus fresh physical RMSE: -0.01102 [-0.014015, -0.0080407].

| Initialization | Checkpoint | Calibration fit | Held-out teacher fit | Physical RMSE | Iterations reported |
|---|---|---:|---:|---:|---:|
| fresh | 120 | 0.010445 [0.0094759, 0.011448] | 0.035824 [0.03121, 0.040479] | 0.036311 [0.031735, 0.04088] | 120 [120, 120] |
| warm | 120 | 0.0061062 [0.0053513, 0.0069289] | 0.01996 [0.017878, 0.021964] | 0.020273 [0.018251, 0.022248] | 120 [120, 120] |
| fresh | 480 | 0.0022244 [0.0019218, 0.0025356] | 0.032033 [0.027681, 0.036387] | 0.03253 [0.0282, 0.036859] | 480 [480, 480] |
| warm | 480 | 0.0022646 [0.0019456, 0.0026189] | 0.020342 [0.017919, 0.022736] | 0.020632 [0.018358, 0.022834] | 480 [480, 480] |
| fresh | 1920 | 0.0003313 [0.00023304, 0.00048128] | 0.032316 [0.028149, 0.036436] | 0.032777 [0.028645, 0.036869] | 1698.2 [1608.6, 1783.7] |
| warm | 1920 | 0.00067401 [0.00048231, 0.00087028] | 0.021537 [0.019073, 0.023926] | 0.021756 [0.019398, 0.023987] | 1763.5 [1607.4, 1882.5] |
| fresh | first threshold | 0.0019058 [0.0018811, 0.001929] | 0.031967 [0.027652, 0.036309] | 0.032447 [0.028167, 0.036763] | 542.5 [485, 607.53] |
| warm | first threshold | 0.0019528 [0.0019402, 0.0019641] | 0.020488 [0.01798, 0.022935] | 0.020789 [0.018445, 0.02305] | 587.5 [493.75, 687.53] |

## 2. Family uncertainty and query placement

Primary query-design contrast, 64 mixed minus 256 narrow queries, selected family: -0.19126 [-0.24611, -0.13695]. Negative differences favor fewer, broader queries. All queries evaluate the source model, not a changed environment.

| Budget | Design | Selected-family accuracy | Selected physical RMSE | Known-family physical RMSE |
|---:|---|---:|---:|---:|
| 16 | narrow | 11/24 (45.8%) | 0.51556 [0.32368, 0.77866] | 0.4117 [0.27436, 0.57561] |
| 16 | broad | 24/24 (100.0%) | 0.01325 [0.009877, 0.016694] | 0.01325 [0.009877, 0.016694] |
| 16 | mixed | 24/24 (100.0%) | 0.028189 [0.021853, 0.034734] | 0.028189 [0.021853, 0.034734] |
| 64 | narrow | 14/24 (58.3%) | 0.1681 [0.12528, 0.20885] | 0.13684 [0.090496, 0.19391] |
| 64 | broad | 24/24 (100.0%) | 0.0064273 [0.0044263, 0.0086858] | 0.0064273 [0.0044263, 0.0086858] |
| 64 | mixed | 24/24 (100.0%) | 0.007175 [0.0051573, 0.0095403] | 0.007175 [0.0051573, 0.0095403] |
| 256 | narrow | 10/24 (41.7%) | 0.19843 [0.14469, 0.25365] | 0.12003 [0.074746, 0.1871] |
| 256 | broad | 24/24 (100.0%) | 0.0044704 [0.003247, 0.0056779] | 0.0044704 [0.003247, 0.0056779] |
| 256 | mixed | 24/24 (100.0%) | 0.0046447 [0.0033609, 0.0059039] | 0.0046447 [0.0033609, 0.0059039] |

Observed family-accuracy counts are not population guarantees. For the joint event that both models in a seed are correctly identified at 64 broad queries, 12/12 seeds succeed; the exact binomial 95% interval over independent seed events is [0.735, 1.000]. This supplementary interval avoids interpreting an all-success percentile bootstrap as certainty.

## Interpretation limits

The candidate library contains only two supplied families, one of which is correct. This does not demonstrate arbitrary mechanism discovery. Broader source-model queries are allowed here; their availability is an application assumption. No semantic weight-reading or LLM claim is tested.

A residual initialization gap at similar calibration fit can reflect an implicit prior, optimization path, or solution selection. It does not uniquely identify one mechanism.
