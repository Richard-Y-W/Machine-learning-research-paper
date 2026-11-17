# Dense symbolic distillation

16 measurements; 2 seed clusters; 0.41 CPU minutes.

128 base queries determine a symbolic teacher; 4,096 synthetic labels require no additional base queries. Each deployed student has 1,217 parameters.

| Initialization | Steps | Physical RMSE | Teacher-fit RMSE | Seconds |
|---|---:|---:|---:|---:|
| Fresh | 600 | 0.0394 [0.03834, 0.04046] | 0.03655 [0.0365, 0.0366] | 0.9962 [0.5133, 1.479] |
| Warm | 600 | 0.01841 [0.01697, 0.01984] | 0.01547 [0.01434, 0.0166] | 0.5349 [0.509, 0.5607] |
| Fresh | 2400 | 0.02055 [0.0204, 0.0207] | 0.01777 [0.01696, 0.01859] | 2.489 [2.072, 2.906] |
| Warm | 2400 | 0.01354 [0.01164, 0.01545] | 0.00953 [0.008082, 0.01098] | 2.097 [2.049, 2.146] |
