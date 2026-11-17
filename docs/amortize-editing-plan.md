# Amortizing Inference-Time Computation Through Iterative Weight-Space Programming

**Status: DRAFT for review (2026-11-15). Nothing in this plan has been run. No edits, no API
spend, until the user approves each stage.**

## 1. Research question

To what extent can iterative, reasoning-driven weight edits (no gradients, no added modules) reduce
the inference-time computation a pretrained model needs for a family of problems, and what
mechanism produces any reduction?

The plan is built to separate three explanations for a gain:

| Explanation | Meaning |
|---|---|
| Format change | The model stops thinking sooner or writes less, with no change to how it computes |
| Better retrieval | The model finds values more reliably but still externalizes every step |
| Amortization | Steps the base model had to write out are now done inside the forward pass |

## 2. Starting point (measured)

Qwen/Qwen3-0.6B (chat-trained), sampled decoding at the model card's settings, baseline seed
`baseline-v1` (`results/amortize/baseline_v1.json`):

| Task (100 cases) | Thinking ≤1024 | ≤512 | ≤384 | ≤256 | Direct ≤64 | Direct + 3 examples |
|---|---|---|---|---|---|---|
| Code tracing, no arithmetic | .99 | .86 | .75 | .08 | .16 | .06 |
| Code tracing, one addition | 1.00 | .99 | .80 | .14 | .02 | .09 |

Median thinking length is 341 and 331 tokens. A cut of roughly a quarter of the thinking tokens
would move most cases under 256. That makes a gain from trimming alone plausible, so section 8
matters as much as the headline number.

## 3. Targets

- **Primary:** thinking-mode accuracy at a 256-token budget (answer finished and correct within
  256 generated tokens).
- **Secondary:** the same at 384 tokens.
- **Retention control:** thinking-mode accuracy at 1,024 tokens must not fall.
- **Reported separately:** non-thinking mode (direct ≤64 tokens, and direct with 3 examples).
  It is never used for selecting edits.

## 4. Data

Ten generated program families. Variable names and values are randomized in every item.

| Family | Structure | Role |
|---|---|---|
| F1 | Reassignment chains (the baseline template) | Development |
| F2 | Swap through a temporary variable | Development |
| F3 | Augmented assignment (`x += k`) | Development |
| F4 | One `if`/`else` on a comparison | Development |
| F5 | Short `for` loop accumulating a value | Development |
| F6 | List index and update | Validation only (held-out structure) |
| F7 | Dictionary lookup chain | Validation only (held-out structure) |
| F8 | String concatenation and `len` | Test only (held-out structure) |
| F9 | Small function call with arguments | Test only (held-out structure) |
| F10 | Tuple unpacking | Test only (held-out structure) |

Splits, each with its own seed:

- **Development (editor-facing).** A fresh 40-case sample from F1–F5 every round, plus a fresh
  40-case confirmation sample. The editor sees development prompts, traces and scores only.
- **Validation.** 60 fixed cases: 30 from F1–F5 (new instances) and 30 from F6–F7. Scored after
  each accepted edit. The editor sees only aggregate validation numbers, never the items.
- **Test.** 300 fixed cases, 30 per family, plus the 4-link lookup task as a transfer check.
  Scored once, after the run is frozen.
- `baseline-v1` items are not reused.

A family enters the study only if the base model shows the gap on it (stage 0a, section 9).

## 5. Editor interface

One Opus 5.5 call per round (effort medium), JSON output, as in the earlier pilot.

**What the editor receives each round:**

- Current scores: development accuracy at 256/384, token statistics, share hitting the cap,
  validation aggregates, protected-skill margins.
- Six development prompts with the current model's full thinking traces (three that finished
  within 256 tokens, three that did not).
- A breakdown of where thinking tokens go (restating the problem, per-line state updates,
  re-checking, final answer), computed by rule-based tagging.
- Diagnostics: logit lens and head/neuron attribution toward the end-of-thinking token and toward
  re-checking tokens such as " Wait", at tagged trace positions.
- Results of the inspection requests it made in the previous round.
- The history of all earlier proposals and their measured effects.

**Inspection requests (weights and activations).** Each response may include up to six requests,
answered in the next round's input at no extra API call:

- Weight inspection: norms, top singular values, the token readout of a singular vector, or a raw
  block of up to 16×16 entries of any named tensor.
- Activation inspection: mean activation or attention pattern of a named head or neuron at a
  tagged trace position.

**Edits.** The existing edit language (scale, ablate, rank-1, raw set, revert; at most 8 operations
and at most 50% relative change per touched tensor per edit). Reference vectors are added for
activations at tagged trace positions and for output-token directions, with differences allowed.
No gradients, optimizers, adapters or new parameters anywhere.

## 6. Round procedure (15 rounds maximum)

1. Score the current model on this round's development sample (cap 384 tokens, which yields the
   256 and 384 results from one run). Build the editor input.
2. The editor proposes four candidate edits and its inspection requests.
3. Apply each candidate separately to the current model and score it on the same sample with the
   same random seeds per item.
4. Take the best candidate by accuracy at 256 (ties: accuracy at 384, then fewer mean tokens).
   It must beat the current model on that sample.
5. Confirm on the fresh confirmation sample: the candidate must be at least as good as the current
   model there too.
6. Interference checks: protected multiple-choice fitness and WikiText loss within 0.02 of the
   base model.
7. If all pass, accept: append the edit to the ordered edit list and save the checkpoint.
8. After each accepted edit, score validation (cap 512) by family, and non-thinking mode. Every
   third accepted edit, and at the end, score 40 validation cases at cap 1,024 for retention.

Each accepted edit is stored separately and replays exactly, so any prefix, subset or
leave-one-out version of the final model can be rebuilt.

**Logged after every accepted edit:** accuracy at 256/384/512 on development, confirmation and
validation (by family); mean and median generated tokens; share hitting the cap; non-thinking
accuracy; protected fitness and WikiText loss; number of parameters changed and relative size of
the change; cumulative API cost; CPU minutes.

## 7. Stopping rules

| Trigger | Action |
|---|---|
| API cost for this attempt would exceed **$4.00** (worst-case reservation before each call) | Stop the run |
| Four consecutive rounds with no accepted edit | Stop: editing has stalled |
| Fewer than half of proposals valid for three consecutive rounds | Halt and fix the interface before continuing |
| Development gain ≥ 15 points while validation gain is under one third of it, for two accepted edits in a row | Stop: overfitting |
| Validation accuracy at 256 falls more than 8 points below its best so far | Stop and roll back to the best checkpoint |
| Retention: validation accuracy at 1,024 more than 5 points below base | Reject that edit |
| Three consecutive best candidates rejected for interference | Stop: excessive interference |
| Round 15 completed | Stop |

The project-wide ledger stop ($44.50 total) still applies on top of the $4 cap.

## 8. Telling amortization from format change or better retrieval

| Test | Format change predicts | Better retrieval predicts | Amortization predicts |
|---|---|---|---|
| **Token cost against program length.** Fit tokens = overhead + per-line cost × lines, base vs edited | Overhead falls; per-line cost unchanged | Little change in either | Per-line cost falls |
| **Trace content.** Share of tokens in restating, state updates, re-checking | Restating and re-checking shrink; every state update still written | Fewer wrong values, same steps | State-update steps are skipped and answers stay correct |
| **Non-thinking accuracy** (3–4 token answers) on test | Unchanged | Small rise on single-lookup items | Rises on multi-step items |
| **Early decodability.** A linear probe for the final answer at the last prompt token, before any thinking | Unchanged | Unchanged | Answer becomes more decodable |
| **Single-lookup vs multi-step items** | Similar gain on both | Gain concentrated on single-lookup | Gain on multi-step |
| **Trivial-baseline frontier** (section 9, stage 0b) | The edit is matched by a "think briefly" prompt or a bias on the end-of-thinking token | Not matched | Not matched |

A claim of amortization requires the per-line cost to fall and at least one of the non-thinking
or early-decodability tests to move, on test families. A gain matched by the end-of-thinking bias
is reported as a format change.

**Edit dependency (after the run, no API).** Leave-one-out removal of each accepted edit, each
edit applied alone to the base model, and prefix-only versions, all scored on validation. A later
edit depends on an earlier one if removing the earlier edit removes more than the earlier edit's
own standalone gain. Dependency shows the sequence mattered; it does not by itself show new
circuitry.

## 9. Stages, costs and approvals

Times are estimates derived from the timing run and are not yet measured for this loop.

| Stage | What | API | CPU (estimate) | Needs approval |
|---|---|---|---|---|
| 0a | Build the ten families; screen each for the gap (keep if thinking ≥ .80 at 1,024 and ≤ .40 at 256) | $0 | 2–3 h | Yes |
| 0b | Trivial baselines on validation: "think briefly" prompts, bias sweep on the end-of-thinking token, steering vector, random edits of matched size | $0 | 2–3 h | Yes |
| 0c | Dry run of one full round with random edits to measure real time per round | $0 | ~1 h | Yes |
| 1 | Iterative editing attempt, 15 rounds maximum | ≤ $4.00 (expected about $2) | 8–12 h | Yes |
| 2 | Post-run analysis: test set once, section 8 tests, edit dependency | $0 | 6–8 h | Yes |
| 3 | Matched-budget one-shot baseline: the same number of calls and candidates, every candidate applied to the base model, edits allowed the chain's total operation budget | about $2 | 6–8 h | Yes, separately |
| 4 | Fine-tuning reference: a short low-rank fine-tune on the model's own short correct traces | $0 | 2–4 h | Yes |

Stage 0b runs before any API spend on purpose: if a simple bias on the end-of-thinking token
already lifts 256-token accuracy, that sets the bar the edits must beat, or shows the target is
trivial.

Budget position: $13.39 spent, cap $45.08, so stages 1 and 3 together would use about $4–6 of
the $31.69 remaining.

## 10. Strongest plausible positive result

On the untouched test set, including the three unseen program structures:

- accuracy at 256 tokens rising from about .10 to .50–.70,
- accuracy at 1,024 tokens within 3 points of base,
- median thinking length falling from about 335 to about 200 tokens,
- the gain exceeding every trivial baseline and the one-shot baseline,
- and edit-dependency tests showing later edits rely on earlier ones.

That would be a persistent improvement in the accuracy–compute trade-off from direct weight edits.
Most likely it would be explained as trimming wasted reasoning. Evidence of true amortization
(lower per-line cost, higher non-thinking accuracy) would be a stronger and less likely result.

Rough odds (judgment, not measurement): a clear held-out gain beyond the trivial baselines,
20–30%; evidence of amortization, 5–10%.

## 11. Conditions that would make the project unpromising

- No edit is accepted in the first five rounds.
- Gains appear on development families and vanish on held-out structures.
- A "think briefly" prompt or an end-of-thinking bias matches the edited model.
- Gains come only with a loss of 1,024-token accuracy or of protected skills.
- The one-shot baseline matches the iterative chain and no edit dependency is found. The result
  could still stand as a weight-editing result, but not as evidence for iteration.

## 12. Known limitations

- One small model; a repeat on Qwen3-1.7B would be a replication, not a scaling law.
- Thinking and non-thinking modes may differ in behaviour, not only in compute.
- The editor makes one call per round with limited inspection, far less than MIT's agent setup.
  A null result may reflect the editor more than the idea.
- Sampled decoding adds evaluation noise; paired seeds and the confirmation sample reduce it but
  do not remove it.
- Fine-tuning will very likely be cheaper for this model. The contribution, if any, is scientific:
  what direct edits can and cannot move.

## Amendment A1 (2026-11-15, after stage 0a; before any validation, test or editing run)

**Screening result (20 cases per family, thinking mode):** F1–F7 and F10 pass (accuracy ≥ .80 at
1,024 tokens and ≤ .40 at 256). F8 (.55) and F9 (.75) fail the 1,024-token bar.

**Changes to family roles:**

- F8 and F9 are removed from selection and from the main test result. They are still scored at
  the end and reported separately as below-threshold families.
- F7 moves from validation to test, so the test set keeps two unseen structures (F7, F10).
  Validation's held-out structure is F6 alone.

**Correction to the baseline's non-thinking numbers.** The baseline's code-tracing prompt said
"Answer with only the number" (an intended wording change had not been applied), and its three
examples reused the query's variable names. A check on 100 cases shows both depressed direct
accuracy:

| Condition (reassignment chains, non-thinking) | Accuracy |
|---|---|
| Baseline wording, examples sharing variable names | .03 |
| Baseline wording, examples with different names | .50 |
| `Answer: <number>` wording, three examples | .65 |
| `Answer: <number>` wording, no examples | .66 |

So on F1 and F2 (pure copy chains) the model answers about 65% correctly with no thinking at
all, in about five tokens. That beats thinking mode at 256 tokens on those families. On the
arithmetic, loop, list, dictionary and tuple families, non-thinking accuracy is 0–25% in the
screening. The baseline's thinking-mode results are unaffected.

**Consequences:**

- Non-thinking mode (no examples) is added to the trivial baselines as the cheapest point on the
  accuracy–compute curve.
- Results are reported in two groups: direct-solvable families (F1, F2) and reasoning-needed
  families (F3–F5 development, F6 validation, F7 and F10 test). The primary claim rests on the
  reasoning-needed group.
- For any family, an edited model's 256-token accuracy must beat non-thinking accuracy to count
  as an improvement in the accuracy–compute trade-off.
