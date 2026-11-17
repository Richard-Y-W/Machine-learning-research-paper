# Same-architecture follow-up

2,688 rows; 12 independent seed clusters; 8.99 CPU minutes.

All deployed students have the same architecture. Warm starts use the pretrained weights; fresh starts do not. All receive the same calibration queries and physics library. The two schedules are not matched FLOP budgets.

## Predeclared primary contrast

Warm minus fresh L-BFGS, symbolic targets, 128 queries, long schedule, original representation: -0.0182 [-0.02255, -0.01401] acceleration RMSE. Negative values favor initialization from pretrained weights.

Exploratory 95% percentile intervals bootstrap independent seed clusters; average families within each seed. No multiplicity adjustment.

## Residual targets, 128 queries, long schedule, original representation

| Method | Physical RMSE | Teacher-fit RMSE | Rollout RMSE | Construction seconds |
|---|---:|---:|---:|---:|
| Warm Adam | 0.01952 [0.01811, 0.021] | 0.01354 [0.01205, 0.0151] | 0.006736 [0.00618, 0.007272] | 0.7137 [0.6223, 0.7998] |
| Fresh Adam | 0.04953 [0.0459, 0.05364] | 0.0506 [0.04699, 0.05472] | 0.01868 [0.0163, 0.02128] | 0.6474 [0.5576, 0.7312] |
| Warm L-BFGS | 0.0175 [0.01639, 0.01874] | 0.01042 [0.009182, 0.01171] | 0.006059 [0.005633, 0.006481] | 0.2088 [0.1777, 0.237] |
| Fresh L-BFGS | 0.04008 [0.0365, 0.04426] | 0.04095 [0.03752, 0.04498] | 0.01355 [0.01212, 0.01515] | 0.2084 [0.1727, 0.2441] |
| Euclidean Jacobian | 0.01716 [0.01609, 0.01825] | 0.009824 [0.008686, 0.01101] | 0.005821 [0.00532, 0.006352] | 0.1052 [0.09263, 0.1186] |
| Column Jacobian | 0.03503 [0.01916, 0.0651] | 0.0294 [0.01301, 0.0603] | 0.01093 [0.005971, 0.02] | 0.1086 [0.09414, 0.1232] |
| Relative Jacobian | 0.0173 [0.01626, 0.01839] | 0.01042 [0.009013, 0.01202] | 0.005791 [0.005449, 0.006098] | 0.1171 [0.1004, 0.1321] |

## Symbolic targets, 128 queries, long schedule, original representation

| Method | Physical RMSE | Teacher-fit RMSE | Rollout RMSE | Construction seconds |
|---|---:|---:|---:|---:|
| Warm Adam | 0.02045 [0.01911, 0.02193] | 0.02006 [0.0186, 0.02156] | 0.006924 [0.006237, 0.007649] | 0.6367 [0.5631, 0.704] |
| Fresh Adam | 0.04902 [0.04535, 0.05312] | 0.04879 [0.04524, 0.05288] | 0.01817 [0.01591, 0.02065] | 0.6916 [0.5976, 0.7809] |
| Warm L-BFGS | 0.01953 [0.01821, 0.02094] | 0.01915 [0.01772, 0.02064] | 0.00626 [0.005656, 0.006936] | 0.2339 [0.2087, 0.2637] |
| Fresh L-BFGS | 0.03773 [0.03349, 0.04226] | 0.03744 [0.03316, 0.04195] | 0.01312 [0.01131, 0.0151] | 0.21 [0.1848, 0.237] |
| Euclidean Jacobian | 0.021 [0.01921, 0.02299] | 0.02061 [0.01868, 0.02273] | 0.007013 [0.005958, 0.008208] | 0.1071 [0.09535, 0.1194] |
| Column Jacobian | 0.04224 [0.02448, 0.07274] | 0.04218 [0.02422, 0.07279] | 0.01197 [0.006893, 0.02111] | 0.1024 [0.08142, 0.1221] |
| Relative Jacobian | 0.01988 [0.01835, 0.02172] | 0.01934 [0.01767, 0.02134] | 0.006008 [0.005255, 0.00685] | 0.1185 [0.1076, 0.1303] |

## Target-objective contrasts

Symbolic minus residual targets at 128 queries, long schedule, original representation. Negative physical-error contrasts favor physics reconstruction; teacher fitting measures a different objective and must be reported separately.

| Solver | Physical RMSE difference | Rollout RMSE difference |
|---|---:|---:|
| Warm Adam | 0.0009352 [0.0003703, 0.001564] | 0.0001885 [-0.0002465, 0.000675] |
| Fresh Adam | -0.0005115 [-0.001262, 0.0001722] | -0.0005141 [-0.001241, 0.0002462] |
| Warm L-BFGS | 0.002029 [0.001545, 0.002617] | 0.0002008 [-0.0003264, 0.0007696] |
| Fresh L-BFGS | -0.00235 [-0.004357, -6.843e-05] | -0.0004278 [-0.001753, 0.0008151] |
| Euclidean Jacobian | 0.003839 [0.002464, 0.005251] | 0.001192 [0.0005126, 0.001948] |
| Column Jacobian | 0.007207 [0.003327, 0.01167] | 0.001043 [0.0002055, 0.001922] |
| Relative Jacobian | 0.002574 [0.001373, 0.003885] | 0.0002178 [-0.0005279, 0.000957] |

## Integrity

Nonfinite rollout rows: 0.
The complete JSON includes all budgets, schedules, representations, and nonfinite counts. This study does not evaluate semantic weight reading or language models.
