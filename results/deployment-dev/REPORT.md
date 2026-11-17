# Same-architecture follow-up

448 rows; 2 independent seed clusters; 1.52 CPU minutes.

All deployed students have the same architecture. Warm starts use the pretrained weights; fresh starts do not. All receive the same calibration queries and physics library. The two schedules are not matched FLOP budgets.

## Predeclared primary contrast

Warm minus fresh L-BFGS, symbolic targets, 128 queries, long schedule, original representation: -0.01914 [-0.02297, -0.01532] acceleration RMSE. Negative values favor initialization from pretrained weights.

Exploratory 95% percentile intervals bootstrap independent seed clusters; average families within each seed. No multiplicity adjustment.

## Residual targets, 128 queries, long schedule, original representation

| Method | Physical RMSE | Teacher-fit RMSE | Rollout RMSE | Construction seconds |
|---|---:|---:|---:|---:|
| Warm Adam | 0.02057 [0.02048, 0.02065] | 0.01238 [0.009645, 0.01511] | 0.006621 [0.005704, 0.007537] | 0.7284 [0.605, 0.8519] |
| Fresh Adam | 0.05481 [0.04932, 0.06029] | 0.05449 [0.04897, 0.06001] | 0.02321 [0.01781, 0.0286] | 0.5607 [0.396, 0.7255] |
| Warm L-BFGS | 0.01869 [0.01727, 0.02012] | 0.0101 [0.008777, 0.01143] | 0.005895 [0.004497, 0.007293] | 0.1407 [0.1175, 0.1639] |
| Fresh L-BFGS | 0.03606 [0.03173, 0.04039] | 0.03577 [0.03167, 0.03987] | 0.01263 [0.01208, 0.01319] | 0.1253 [0.1238, 0.1268] |
| Euclidean Jacobian | 0.01815 [0.01604, 0.02027] | 0.00931 [0.007691, 0.01093] | 0.005011 [0.003619, 0.006403] | 0.09549 [0.0859, 0.1051] |
| Column Jacobian | 0.0206 [0.0205, 0.02069] | 0.01325 [0.00926, 0.01724] | 0.005712 [0.004661, 0.006762] | 0.1109 [0.1025, 0.1192] |
| Relative Jacobian | 0.01775 [0.016, 0.0195] | 0.009213 [0.007357, 0.01107] | 0.005244 [0.003783, 0.006705] | 0.1196 [0.101, 0.1383] |

## Symbolic targets, 128 queries, long schedule, original representation

| Method | Physical RMSE | Teacher-fit RMSE | Rollout RMSE | Construction seconds |
|---|---:|---:|---:|---:|
| Warm Adam | 0.02133 [0.0206, 0.02206] | 0.02009 [0.01893, 0.02124] | 0.007498 [0.007069, 0.007927] | 0.7879 [0.5984, 0.9773] |
| Fresh Adam | 0.05339 [0.04878, 0.05799] | 0.05104 [0.04757, 0.05452] | 0.02255 [0.01771, 0.0274] | 0.809 [0.7952, 0.8229] |
| Warm L-BFGS | 0.02033 [0.01824, 0.02241] | 0.01905 [0.01657, 0.02152] | 0.006951 [0.006028, 0.007874] | 0.2005 [0.131, 0.2701] |
| Fresh L-BFGS | 0.03947 [0.03773, 0.04121] | 0.03679 [0.03401, 0.03956] | 0.01485 [0.0141, 0.0156] | 0.2084 [0.1231, 0.2938] |
| Euclidean Jacobian | 0.0214 [0.01976, 0.02304] | 0.02052 [0.01881, 0.02224] | 0.007111 [0.006889, 0.007332] | 0.0922 [0.06989, 0.1145] |
| Column Jacobian | 0.02413 [0.02329, 0.02498] | 0.02291 [0.02199, 0.02383] | 0.007122 [0.005675, 0.008569] | 0.1069 [0.09161, 0.1221] |
| Relative Jacobian | 0.01879 [0.01748, 0.0201] | 0.01749 [0.01628, 0.01869] | 0.006342 [0.005488, 0.007196] | 0.1255 [0.1096, 0.1414] |

## Target-objective contrasts

Symbolic minus residual targets at 128 queries, long schedule, original representation. Negative physical-error contrasts favor physics reconstruction; teacher fitting measures a different objective and must be reported separately.

| Solver | Physical RMSE difference | Rollout RMSE difference |
|---|---:|---:|
| Warm Adam | 0.0007608 [-5.27e-05, 0.001574] | 0.0008772 [0.0003895, 0.001365] |
| Fresh Adam | -0.001418 [-0.0023, -0.0005352] | -0.0006551 [-0.001207, -0.0001033] |
| Warm L-BFGS | 0.001632 [0.0009782, 0.002286] | 0.001056 [0.0005808, 0.001531] |
| Fresh L-BFGS | 0.003414 [0.0008284, 0.006] | 0.002217 [0.0009098, 0.003525] |
| Euclidean Jacobian | 0.003248 [0.002768, 0.003727] | 0.0021 [0.0009292, 0.003271] |
| Column Jacobian | 0.003537 [0.002789, 0.004285] | 0.00141 [0.001014, 0.001806] |
| Relative Jacobian | 0.00104 [0.0006006, 0.00148] | 0.001098 [0.0004908, 0.001705] |

## Integrity

Nonfinite rollout rows: 0.
The complete JSON includes all budgets, schedules, representations, and nonfinite counts. This study does not evaluate semantic weight reading or language models.
