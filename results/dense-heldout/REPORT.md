# Dense symbolic distillation

96 measurements; 12 seed clusters; 3.11 CPU minutes.

128 base queries determine a symbolic teacher; 4,096 synthetic labels require no additional base queries. Each deployed student has 1,217 parameters.

| Initialization | Steps | Physical RMSE | Teacher-fit RMSE | Seconds |
|---|---:|---:|---:|---:|
| Fresh | 600 | 0.03718 [0.03427, 0.03997] | 0.03686 [0.0341, 0.03961] | 0.8828 [0.7236, 1.047] |
| Warm | 600 | 0.01586 [0.01497, 0.01687] | 0.01503 [0.01411, 0.01601] | 0.7631 [0.6624, 0.8689] |
| Fresh | 2400 | 0.01752 [0.0162, 0.01887] | 0.01672 [0.01543, 0.01803] | 3.122 [2.815, 3.461] |
| Warm | 2400 | 0.01189 [0.01104, 0.01284] | 0.01077 [0.009791, 0.01183] | 2.948 [2.56, 3.377] |
