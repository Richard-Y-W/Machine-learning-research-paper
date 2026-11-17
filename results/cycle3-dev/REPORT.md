# Cycle 3 findings

4 new source models; 32 fitting checkpoints; 72 identification measurements; 1.02 minutes on CPU.

Intervals are exploratory 95% seed-cluster bootstrap intervals; families are averaged within each seed. Threshold crossings are selected using calibration error only.

## 1. Longer optimization and similar calibration fit

Calibration threshold <=0.002 attained by 4/4 warm and 4/4 fresh students.

At first threshold crossing, warm minus fresh physical RMSE on the jointly attained subset: -0.013387 [-0.015427, -0.011347]; 4/4 paired model conditions. This subset comparison is not an unconditional effect estimate.

At the full fixed 1,920-iteration budget, warm minus fresh physical RMSE: -0.012196 [-0.014261, -0.010131].

| Initialization | Checkpoint | Calibration fit | Held-out teacher fit | Physical RMSE | Iterations reported |
|---|---|---:|---:|---:|---:|
| fresh | 120 | 0.011337 [0.010791, 0.011883] | 0.03861 [0.035204, 0.042017] | 0.041316 [0.039183, 0.043449] | 120 [120, 120] |
| warm | 120 | 0.0058633 [0.004795, 0.0069316] | 0.019092 [0.017282, 0.020901] | 0.020208 [0.018683, 0.021733] | 120 [120, 120] |
| fresh | 480 | 0.0021291 [0.0015477, 0.0027106] | 0.031509 [0.029339, 0.033678] | 0.034296 [0.033482, 0.03511] | 480 [480, 480] |
| warm | 480 | 0.0023825 [0.0019089, 0.0028562] | 0.018739 [0.01731, 0.020168] | 0.019773 [0.018253, 0.021294] | 480 [480, 480] |
| fresh | 1920 | 0.00044368 [0.00017602, 0.00071133] | 0.03094 [0.029548, 0.032332] | 0.033582 [0.033275, 0.033888] | 1855 [1835.5, 1874.5] |
| warm | 1920 | 0.00053737 [0.0003326, 0.00074215] | 0.020455 [0.018241, 0.022669] | 0.021386 [0.019014, 0.023757] | 1666 [1623.5, 1708.5] |
| fresh | first threshold | 0.0018911 [0.0018691, 0.0019131] | 0.030855 [0.02944, 0.03227] | 0.033608 [0.033593, 0.033624] | 532.5 [450, 615] |
| warm | first threshold | 0.0019228 [0.0018674, 0.0019782] | 0.019169 [0.017315, 0.021022] | 0.020222 [0.018197, 0.022247] | 607.5 [510, 705] |

## 2. Family uncertainty and query placement

Primary query-design contrast, 64 mixed minus 256 narrow queries, selected family: -0.19102 [-0.20833, -0.17372]. Negative differences favor fewer, broader queries. All queries evaluate the source model, not a changed environment.

| Budget | Design | Selected-family accuracy | Selected physical RMSE | Known-family physical RMSE |
|---:|---|---:|---:|---:|
| 16 | narrow | 0.5 [0.5, 0.5] | 0.35912 [0.27942, 0.43883] | 0.41614 [0.36844, 0.46383] |
| 16 | broad | 1 [1, 1] | 0.0067421 [0.0054564, 0.0080279] | 0.0067421 [0.0054564, 0.0080279] |
| 16 | mixed | 0.75 [0.5, 1] | 0.039414 [0.020079, 0.05875] | 0.024705 [0.020079, 0.029332] |
| 64 | narrow | 0.25 [0, 0.5] | 0.24579 [0.20672, 0.28485] | 0.09844 [0.05882, 0.13806] |
| 64 | broad | 1 [1, 1] | 0.004175 [0.0034881, 0.004862] | 0.004175 [0.0034881, 0.004862] |
| 64 | mixed | 1 [1, 1] | 0.0052346 [0.0037418, 0.0067274] | 0.0052346 [0.0037418, 0.0067274] |
| 256 | narrow | 0.5 [0.5, 0.5] | 0.19626 [0.17746, 0.21506] | 0.15744 [0.15136, 0.16351] |
| 256 | broad | 1 [1, 1] | 0.0053919 [0.004862, 0.0059217] | 0.0053919 [0.004862, 0.0059217] |
| 256 | mixed | 1 [1, 1] | 0.0049737 [0.0048158, 0.0051316] | 0.0049737 [0.0048158, 0.0051316] |

## Interpretation limits

The candidate library contains only two supplied families, one of which is correct. This does not demonstrate arbitrary mechanism discovery. Broader source-model queries are allowed here; their availability is an application assumption. No semantic weight-reading or LLM claim is tested.

A residual initialization gap at similar calibration fit can reflect an implicit prior, optimization path, or solution selection. It does not uniquely identify one mechanism.
