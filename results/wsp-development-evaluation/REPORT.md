# Direct LLM weight-space programming results

Model: claude-opus-5-5; tool-free, no forward passes, no data, no feedback.

1/1 calls attempted; 1 valid patches; 1 successful functional edits. Estimated generation cost: $0.0286.

These prompts derive from only 1 independently trained source networks. Capacity and representation variants are paired controls, not independent replications.

| Task | Capacity | Valid / attempted | Successful | Mean edit NRMSE |
|---|---|---:|---:|---:|
| action_gain | dense | 1/1 | 1/1 | 4.0831e-16 |
| action_gain | spare | 0/0 | 0/0 | unavailable |
| add_drag | dense | 0/0 | 0/0 | unavailable |
| add_drag | spare | 0/0 | 0/0 | unavailable |
| semantic_damping | dense | 0/0 | 0/0 | unavailable |
| semantic_damping | spare | 0/0 | 0/0 | unavailable |

## Interpretation

Action-column scaling is an architectural arithmetic control. Spare-unit drag edits have an exact elementary ReLU compiler. Matching these controls demonstrates direct numerical programming, not a new optimization algorithm or general semantic understanding.

Semantic damping combines reading an approximate physical coefficient with compiling a correction. The details file separates coefficient error and fidelity to the model's own estimated change. Oracle references use hidden physical coefficients; query references use source evaluations unavailable to the tool-free LLM arm.

Invalid/truncated responses count as failures. Budget-stopped prompts are reported as unattempted. No prompt tuning or retries are performed on evaluated pilot outcomes.
