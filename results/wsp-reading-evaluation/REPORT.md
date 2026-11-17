# Direct LLM weight-space programming results

Model: claude-opus-5-5; tool-free, no forward passes, no data, no feedback.

12/12 calls attempted; 12 valid patches; 7 successful functional edits. Estimated generation cost: $0.4251.

These prompts derive from only 6 independently trained source networks. Capacity and representation variants are paired controls, not independent replications.

| Task | Capacity | Valid / attempted | Successful | Mean edit NRMSE |
|---|---|---:|---:|---:|
| action_gain | dense | 0/0 | 0/0 | unavailable |
| action_gain | spare | 6/6 | 6/6 | 4.0032e-16 |
| add_drag | dense | 0/0 | 0/0 | unavailable |
| add_drag | spare | 0/0 | 0/0 | unavailable |
| semantic_damping | dense | 0/0 | 0/0 | unavailable |
| semantic_damping | spare | 6/6 | 1/6 | 0.35298 |

## Interpretation

Action-column scaling is an architectural arithmetic control. Spare-unit drag edits have an exact elementary ReLU compiler. Matching these controls demonstrates direct numerical programming, not a new optimization algorithm or general semantic understanding.

Semantic damping combines reading an approximate physical coefficient with compiling a correction. The details file separates coefficient error and fidelity to the model's own estimated change. Oracle references use hidden physical coefficients; query references use source evaluations unavailable to the tool-free LLM arm.

Invalid/truncated responses count as failures. Budget-stopped prompts are reported as unattempted. No prompt tuning or retries are performed on evaluated pilot outcomes.
