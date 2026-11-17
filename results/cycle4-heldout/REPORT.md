# Cycle 4: predictions versus internal features

48 source models; 1728 condition-method rows; 11.93 minutes on CPU.

This is adaptation from noisy changed-environment observations, not specification-only editing. No physics library or intervention formula is supplied to the adapters. Source predictions are callable during adaptation and deployment; residual wrappers have a different architecture/storage contract from in-place edits.

## Primary setting: added mechanisms, 64 observations

| Method | Physical RMSE | Extrapolation RMSE | Rollout RMSE | Adaptation seconds |
|---|---:|---:|---:|---:|
| Source only | 0.29823 [0.29666, 0.29969] | 0.38001 [0.37074, 0.38862] | 0.13451 [0.12518, 0.14397] | 3.4271e-05 [3.1227e-05, 3.7298e-05] |
| Warm full | 0.085592 [0.078113, 0.094019] | 0.27095 [0.2565, 0.28463] | 0.041597 [0.035764, 0.046924] | 1.4011 [1.2211, 1.5893] |
| Fresh full | 0.089356 [0.080198, 0.099024] | 0.35856 [0.32496, 0.38999] | 0.038585 [0.031367, 0.045989] | 1.445 [1.3105, 1.5818] |
| Learned head | 0.1338 [0.12794, 0.14045] | 0.31576 [0.29462, 0.33949] | 0.062815 [0.056235, 0.06907] | 0.0026298 [0.0023899, 0.0028742] |
| Linear residual | 0.17185 [0.16567, 0.17813] | 0.24773 [0.23719, 0.25932] | 0.08358 [0.076446, 0.091198] | 0.0017024 [0.00156, 0.0018576] |
| Cubic residual | 0.089431 [0.083624, 0.094823] | 0.60176 [0.55305, 0.65784] | 0.035204 [0.031776, 0.038573] | 0.0020823 [0.0018881, 0.0022837] |
| RBF residual | 0.074057 [0.070126, 0.078464] | 0.27202 [0.24852, 0.30024] | 0.027934 [0.023652, 0.03268] | 0.0061491 [0.0057215, 0.0065742] |
| ARD RBF residual | 0.037229 [0.034701, 0.039814] | 0.21892 [0.20488, 0.23181] | 0.013139 [0.011387, 0.015017] | 0.087265 [0.081994, 0.091955] |
| Neural residual | 0.067517 [0.058566, 0.077464] | 0.25703 [0.24239, 0.27205] | 0.02763 [0.021746, 0.034035] | 1.444 [1.2029, 1.6759] |

## Paired access contrasts

Negative differences favor the prediction-only residual method. Intervals bootstrap independent seeds, averaging all four systems within a seed; no multiplicity adjustment.

| Contrast | Physical RMSE difference | Rollout RMSE difference |
|---|---:|---:|
| RBF residual minus Warm full | -0.011534 [-0.021276, -0.0018148] | -0.013663 [-0.020233, -0.0068529] |
| RBF residual minus Learned head | -0.059743 [-0.067416, -0.052122] | -0.034881 [-0.040025, -0.029672] |
| ARD RBF residual minus Warm full | -0.048363 [-0.057862, -0.039339] | -0.028457 [-0.03479, -0.021957] |
| Neural residual minus Warm full | -0.018075 [-0.03204, -0.0025974] | -0.013967 [-0.023309, -0.0044397] |

## System-specific physical error, nonlinear shift, 64 observations

| System | Warm full | Learned head | RBF | ARD RBF | Neural residual |
|---|---:|---:|---:|---:|---:|
| duffing | 0.092338 [0.078124, 0.10861] | 0.12387 [0.1123, 0.13608] | 0.076975 [0.072346, 0.082794] | 0.040455 [0.03746, 0.043253] | 0.06396 [0.056984, 0.071371] |
| pendulum | 0.079468 [0.071142, 0.088176] | 0.11828 [0.1075, 0.12907] | 0.069844 [0.063622, 0.076904] | 0.033983 [0.031078, 0.037056] | 0.0662 [0.056703, 0.077663] |
| contact | 0.080201 [0.069522, 0.094137] | 0.13494 [0.1261, 0.14489] | 0.075141 [0.070119, 0.080614] | 0.035271 [0.031637, 0.038904] | 0.072114 [0.058988, 0.088721] |
| vanderpol | 0.090358 [0.078668, 0.10188] | 0.1581 [0.13851, 0.18355] | 0.074269 [0.069802, 0.078311] | 0.039206 [0.035839, 0.042822] | 0.067792 [0.057519, 0.078629] |

## Parameter-shift control, 16 observations

| Method | Physical RMSE |
|---|---:|
| Source only | 0.46012 [0.41361, 0.50824] |
| Warm full | 0.055554 [0.0459, 0.067173] |
| Fresh full | 0.20634 [0.18828, 0.22401] |
| Learned head | 0.10133 [0.082567, 0.1244] |
| Linear residual | 0.015359 [0.013803, 0.016992] |
| Cubic residual | 0.15075 [0.12066, 0.18528] |
| RBF residual | 0.03931 [0.032834, 0.047214] |
| ARD RBF residual | 0.04211 [0.034315, 0.052099] |
| Neural residual | 0.070728 [0.05572, 0.088029] |

## Integrity and limits

Nonfinite rollout rows: 0.
No LLM is tested. Generic polynomial and kernel smoothness assumptions replace the earlier named mechanism library; this is not assumption-free adaptation. Neural early stopping and kernel hyperparameters use calibration observations only. Inference wrapper costs differ; timings include validation/refitting work but are descriptive single-machine measurements.
