# Direct weight-space programming: combined diagnostic report

Primary and follow-up outcomes remain separately available. This pooled report is descriptive; 36 prompts represent eight trained sources, not 36 independent networks.

| Task | Passed | Mean edit NRMSE | Mean OOD NRMSE |
|---|---:|---:|---:|
| action_gain | 14/14 | 3.8844e-16 | 3.5836e-16 |
| add_drag | 8/8 | 7.3256e-16 | 0.033275 |
| semantic_damping | 1/14 | 0.3452 | 0.33994 |

## Coefficient estimates: one condition per source

| Seed | True c | LLM | Local slope | Query fit |
|---|---:|---:|---:|---:|
| 90 | 0.59335 | 0.72000 | 0.71969 | 0.59289 |
| 91 | 0.28506 | 0.42000 | 0.41712 | 0.31119 |
| 92 | 0.58749 | 0.70000 | 0.70918 | 0.59236 |
| 93 | 0.57990 | 0.70200 | 0.70203 | 0.70150 |
| 94 | 0.45517 | 0.46000 | 0.51196 | 0.45644 |
| 95 | 0.58590 | 0.70400 | 0.70033 | 0.60057 |
| 96 | 0.58027 | 0.74718 | 0.74718 | 0.59687 |
| 97 | 0.34716 | 0.76930 | 0.76934 | 0.76557 |

Coefficient MAE: {"llm": 0.15102412818328959, "query": 0.07550139585809776, "prior": 0.14373249172371802, "local": 0.1578155195621727}

Maximum own-estimate compilation NRMSE: 6.88509e-07.

Local-slope analysis is post hoc. It tests an explanation of the failure, not a prespecified superiority claim. A piecewise-linear approximation's slope at one point need not equal the physical coefficient.

Estimated list-price spend including development: $1.275608. No additional paid calls used in analysis.
