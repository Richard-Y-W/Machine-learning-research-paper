# Pilot results

7776 rows; 12 independent seeds; 354.8 seconds. Maximum symmetry error: 1.33e-15.

Intervals are exploratory 95% percentile bootstraps over seed clusters. No multiplicity correction; no claim of confirmatory hypothesis testing.

## Matched libraries, original representation, 128 calibration queries

| Method | Acceleration RMSE | Extrapolation RMSE | Edit NRMSE | Rollout RMSE |
|---|---:|---:|---:|---:|
| Unedited | 0.378 [0.3378, 0.4153] | 0.7113 [0.6444, 0.7712] | 1 [1, 1] | 0.1144 [0.109, 0.1202] |
| Query + residual | 0.01492 [0.01363, 0.01636] | 0.162 [0.1435, 0.1814] | 0.01026 [0.007413, 0.01345] | 0.005105 [0.004451, 0.005792] |
| Query symbolic | 0.005622 [0.00479, 0.006421] | 0.00978 [0.008314, 0.01143] | 0.04498 [0.03644, 0.05416] | 0.00192 [0.001474, 0.002429] |
| Head ridge | 0.0266 [0.02504, 0.02817] | 0.1871 [0.1732, 0.2017] | 0.05729 [0.05337, 0.0619] | 0.01015 [0.009517, 0.0108] |
| Jacobian ridge | 0.01743 [0.01626, 0.01866] | 0.1932 [0.1742, 0.213] | 0.02594 [0.02272, 0.0291] | 0.005864 [0.005278, 0.006473] |
| Normalized Jacobian | 0.04064 [0.02349, 0.06383] | 0.2387 [0.2099, 0.2696] | 0.09244 [0.04637, 0.1526] | 0.01164 [0.007198, 0.01767] |
| Adam fine-tuning | 0.02137 [0.0202, 0.02256] | 0.2158 [0.1981, 0.2333] | 0.04193 [0.03995, 0.04404] | 0.008048 [0.007136, 0.008965] |
| Oracle residual | 0.01388 [0.01256, 0.01537] | 0.1642 [0.1456, 0.1837] | 2.004e-16 [1.842e-16, 2.194e-16] | 0.004901 [0.004269, 0.005542] |

## Paired differences against the query residual

Positive differences favor the query residual baseline.

| Weight editor | Acceleration RMSE difference | Rollout RMSE difference |
|---|---:|---:|
| Head ridge | 0.01168 [0.009427, 0.01404] | 0.005046 [0.004423, 0.005775] |
| Jacobian ridge | 0.002509 [0.001917, 0.003251] | 0.0007597 [0.0005428, 0.00101] |
| Normalized Jacobian | 0.02571 [0.008053, 0.04969] | 0.006533 [0.001834, 0.01293] |
| Adam fine-tuning | 0.00645 [0.005233, 0.00761] | 0.002943 [0.002142, 0.003808] |

## Integrity

Nonfinite rollout rows: 0.0.

See summary.json for all family/budget/representation strata and nonfinite counts.
Query symbolic replacement and query residual use different deployment architectures from weight editors. Their comparison establishes a task baseline, not equal-capacity superiority.
