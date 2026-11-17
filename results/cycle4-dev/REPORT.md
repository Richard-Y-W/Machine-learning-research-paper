# Cycle 4: predictions versus internal features

8 source models; 288 condition-method rows; 1.97 minutes on CPU.

This is adaptation from noisy changed-environment observations, not specification-only editing. No physics library or intervention formula is supplied to the adapters. Source predictions are callable during adaptation and deployment; residual wrappers have a different architecture/storage contract from in-place edits.

## Primary setting: added mechanisms, 64 observations

| Method | Physical RMSE | Extrapolation RMSE | Rollout RMSE | Adaptation seconds |
|---|---:|---:|---:|---:|
| Source only | 0.30257 [0.2979, 0.30724] | 0.36365 [0.35351, 0.37378] | 0.13968 [0.11259, 0.16676] | 3.2375e-05 [2.945e-05, 3.53e-05] |
| Warm full | 0.084331 [0.079016, 0.089646] | 0.25481 [0.20692, 0.3027] | 0.032811 [0.028584, 0.037038] | 0.96237 [0.85993, 1.0648] |
| Fresh full | 0.084141 [0.080524, 0.087757] | 0.3142 [0.26827, 0.36013] | 0.043021 [0.030824, 0.055219] | 0.98402 [0.7079, 1.2601] |
| Learned head | 0.14266 [0.13655, 0.14877] | 0.31416 [0.27341, 0.3549] | 0.063827 [0.055769, 0.071885] | 0.0021961 [0.0021678, 0.0022245] |
| Linear residual | 0.18026 [0.18026, 0.18027] | 0.22453 [0.20886, 0.2402] | 0.08985 [0.076732, 0.10297] | 0.0016173 [0.0016166, 0.0016179] |
| Cubic residual | 0.081194 [0.079402, 0.082986] | 0.63536 [0.61128, 0.65944] | 0.034862 [0.027802, 0.041922] | 0.0017841 [0.0017026, 0.0018655] |
| RBF residual | 0.071522 [0.064411, 0.078634] | 0.23691 [0.2237, 0.25013] | 0.030294 [0.028697, 0.031891] | 0.0057683 [0.0057057, 0.0058309] |
| ARD RBF residual | 0.03417 [0.033254, 0.035085] | 0.19676 [0.1943, 0.19922] | 0.0121 [0.0080959, 0.016104] | 0.085845 [0.079333, 0.092358] |
| Neural residual | 0.056667 [0.047163, 0.066172] | 0.23654 [0.20633, 0.26676] | 0.023159 [0.010278, 0.03604] | 1.321 [0.8619, 1.78] |

## Paired access contrasts

Negative differences favor the prediction-only residual method. Intervals bootstrap independent seeds, averaging all four systems within a seed; no multiplicity adjustment.

| Contrast | Physical RMSE difference | Rollout RMSE difference |
|---|---:|---:|
| RBF residual minus Warm full | -0.012809 [-0.025235, -0.00038248] | -0.0025167 [-0.0083408, 0.0033074] |
| RBF residual minus Learned head | -0.071137 [-0.084355, -0.05792] | -0.033533 [-0.039994, -0.027071] |
| ARD RBF residual minus Warm full | -0.050161 [-0.056392, -0.043931] | -0.020711 [-0.028942, -0.01248] |
| Neural residual minus Warm full | -0.027664 [-0.042483, -0.012845] | -0.0096514 [-0.02676, 0.0074569] |

## System-specific physical error, nonlinear shift, 64 observations

| System | Warm full | Learned head | RBF | ARD RBF | Neural residual |
|---|---:|---:|---:|---:|---:|
| duffing | 0.091173 [0.077903, 0.10444] | 0.14417 [0.12147, 0.16686] | 0.072719 [0.066176, 0.079262] | 0.032686 [0.030776, 0.034595] | 0.05924 [0.049114, 0.069366] |
| pendulum | 0.090473 [0.082659, 0.098286] | 0.13574 [0.13048, 0.14099] | 0.070249 [0.060976, 0.079523] | 0.034518 [0.033417, 0.035619] | 0.051419 [0.045786, 0.057053] |
| contact | 0.077312 [0.077239, 0.077385] | 0.13043 [0.11319, 0.14767] | 0.070487 [0.064163, 0.076812] | 0.032443 [0.032318, 0.032569] | 0.060513 [0.05439, 0.066637] |
| vanderpol | 0.078368 [0.078118, 0.078618] | 0.1603 [0.13954, 0.18107] | 0.072634 [0.066329, 0.078939] | 0.037032 [0.032436, 0.041628] | 0.055498 [0.039364, 0.071631] |

## Parameter-shift control, 16 observations

| Method | Physical RMSE |
|---|---:|
| Source only | 0.46992 [0.33731, 0.60253] |
| Warm full | 0.051524 [0.038633, 0.064416] |
| Fresh full | 0.18574 [0.14657, 0.2249] |
| Learned head | 0.10483 [0.088603, 0.12107] |
| Linear residual | 0.014419 [0.013184, 0.015654] |
| Cubic residual | 0.14189 [0.092428, 0.19136] |
| RBF residual | 0.044974 [0.024211, 0.065737] |
| ARD RBF residual | 0.047004 [0.028271, 0.065737] |
| Neural residual | 0.069317 [0.050795, 0.087839] |

## Integrity and limits

Nonfinite rollout rows: 0.
No LLM is tested. Generic polynomial and kernel smoothness assumptions replace the earlier named mechanism library; this is not assumption-free adaptation. Neural early stopping and kernel hyperparameters use calibration observations only. Inference wrapper costs differ; timings include validation/refitting work but are descriptive single-machine measurements.
