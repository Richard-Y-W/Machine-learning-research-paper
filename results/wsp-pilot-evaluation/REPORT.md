# Direct LLM weight-space programming results

Model: claude-opus-5-5; tool-free, no forward passes, no data, no feedback.

24/24 calls attempted; 24 valid patches; 16 successful functional edits. Estimated generation cost: $0.8220.

These prompts derive from only 2 independently trained source networks. Capacity and representation variants are paired controls, not independent replications.

| Task | Capacity | Valid / attempted | Successful | Mean edit NRMSE |
|---|---|---:|---:|---:|
| action_gain | dense | 4/4 | 4/4 | 3.7597e-16 |
| action_gain | spare | 4/4 | 4/4 | 3.8308e-16 |
| add_drag | dense | 4/4 | 4/4 | 9.0131e-16 |
| add_drag | spare | 4/4 | 4/4 | 5.6382e-16 |
| semantic_damping | dense | 4/4 | 0/4 | 0.33794 |
| semantic_damping | spare | 4/4 | 0/4 | 0.34079 |

## Interpretation

Action-column scaling is an architectural arithmetic control. Spare-unit drag edits have an exact elementary ReLU compiler. Matching these controls demonstrates direct numerical programming, not a new optimization algorithm or general semantic understanding.

Semantic damping combines reading an approximate physical coefficient with compiling a correction. The details file separates coefficient error and fidelity to the model's own estimated change. Oracle references use hidden physical coefficients; query references use source evaluations unavailable to the tool-free LLM arm.

Invalid/truncated responses count as failures. Budget-stopped prompts are reported as unattempted. No prompt tuning or retries are performed on evaluated pilot outcomes.
