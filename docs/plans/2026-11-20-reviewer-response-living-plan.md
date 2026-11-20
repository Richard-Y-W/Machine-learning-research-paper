# Living plan: answering the review of *Routing Edits*

Started 2026-11-20. Update this file whenever a run finishes, a decision is made, or a result changes the plan.
Newest entries go at the top of the log at the bottom of this file.

## Goal

The current preprint shows one causally supported intervention: succession, in two Qwen3 models, measured on small
development samples. The review asks for evidence that the effect is **reliable**, **transferable**, and **not only
succession**, and for **properly matched baselines**. Every item here is forward passes or small fits on existing
pretrained models. No API spend is planned. The API cap of $44.50 total is unchanged.

Preferred order (reviewer and us): independent evaluation → simple baselines → unrelated-family replication →
second-operation discovery.

## Rules for every run

- Declare the protocol in the script docstring **before** the run. That covers the evaluation sets, the selection
  rules and the gains. Do not change it after seeing results; if a change is needed, add a new, labelled run.
- Never overwrite archived results (`results/compose*`). New results go to new directories (`results/robustness_*`,
  `results/xfamily_*`, `results/second_op2_*`).
- Run one model process at a time. This is a 31 GB CPU-only machine, and two concurrent model jobs got killed for
  low memory on 2026-11-19.
- Report negative results as prominently as positive ones.

## Work items

| # | Item | Script | Output | Status |
|---|------|--------|--------|--------|
| 1 | Independent evaluation: 1,800 fresh successor prompts (third key set, 3 formats × 5 value types), 64k WikiText-103 **validation** tokens, 700 native sequence prompts, fresh word lookups | `experiments/compose/robustness.py` | `results/robustness_{0p6b,1p7b}/robustness.json` | **0.6B and 1.7B done** |
| 1b | Refit stability: 5 fits on disjoint fifths of the word pool, own seeds, own ridge selection, gain fixed at 3 | same | same, plus `refit{k}.pt` | **done: stable (0.6B 0.569–0.609; 1.7B 0.386–0.422)** |
| 4a | Prompting baseline: instruction, 3-shot composed demos from other value types, same value type, instruction + same type; on unedited and edited models | `experiments/compose/prompting.py` | `results/robustness_*/prompting.json` | **done. 0.6B: best prompt 0.28 vs edit 0.68. 1.7B: best prompt 0.47 ≈ edit 0.45; edit + instruction 0.70** |
| 4b | Direct vs copied-head edit at **matched success**: gain sweeps of both designs, compared on success/damage curves. Spare head chosen by a fixed rule (h4 on 0.6B) | `robustness.py` | `robustness.json` | **done, both models: copied dominates on sequences, equal WikiText cost** |
| 4c | Learning comparisons on the independent evaluation: Q (same routing labels and same parameters, gradient), Fm (same parameters, composed answers), F (layers 18–24, composed answers) | `experiments/compose/learning_ind.py` | `results/robustness_0p6b/learning_ind.json` | **0.6B done: Q beats closed form (0.66 vs 0.58; seq 0.62 vs 0.23); F 0.97** |
| 2 | Unrelated family: Pythia-1.4B-deduped under the protocol declared in the script. Head located by rule; routing in the non-rotary dimensions; gain chosen on days only; controls included | `experiments/compose/xfamily.py` | `results/xfamily_pythia1p4b/xfamily.json` | **done: replicates (arrow 0.29→0.56, WikiText +0.4%)** |
| 3 | Second operation, staged: native competence ≥ 0.80 → attribution-ranked heads → value-patch vs output-patch oracles for single heads and groups of 2/4/8/16. No routing fit unless a value patch of ≤ 8 heads reaches ≥ 0.50 | `experiments/compose/second_op2.py` | `results/second_op2_0p6b/second_op2.json` | **0.6B: 7 negatives + uppercase candidate (0.68). Pythia: all negative, uppercase 0.00 via value path. `uppercase_route.py`: gate failed (plain-lookup oracle 0.23 < 0.30; controls 0.00). Succession is the only routable operation found** |
| 4d | Replace vs add: same programme, native q/k kept (band-only / added) | `experiments/compose/additive.py` | `results/robustness_0p6b/additive.json` | **done: keeping native rows preserves sequences but kills routing (successor 0.09–0.15)** |
| — | LLM-designer equal-budget search baseline | — | — | **Deferred.** Only needed if the paper claims the LLM designer is an effective discovery method. Otherwise keep that section exploratory. |

## Overnight queue (2026-11-20)

`tmp/queue3.sh` (restart-safe; replaced `queue2.sh`), log in `tmp/queue3.log`. A cron watchdog relaunches it every
20 min if it was killed. Order, one job at a time (second-operation screen on Pythia appended at the end):

1. robustness 0.6B (resumes from saved conditions)
2. prompting 0.6B
3. Pythia replication
4. second-operation screen 0.6B
5. learning comparisons 0.6B
6. robustness 1.7B
7. prompting 1.7B

The queue log is `tmp/queue2.log`; each job also writes `run.log` in its output directory.
Rough CPU estimate: 0.6B jobs take 1–3 h each, and 1.7B robustness about 6–8 h, so the queue will likely run into the day.

## Decision points (what each result changes)

- **Refits disagree** (spread of successor rate > ~10 points, or some refits fail): the construction is unstable.
  The paper should then report the distribution, not the single fit.
- **Prompting already works** (unedited few-shot ≥ edited): the edit installs a default and does not overcome an
  inability. Reframe the claim accordingly.
- **Copied-head damage is lower at matched success**: this supports the copied design as the recommended edit.
  If the damage is equal, drop the preservation claim.
- **Pythia replicates** (clear gains, causal checks pass): this becomes the main transfer evidence.
  **Pythia fails**: report it, then diagnose stage by stage: no head located, routing attention not learned, or OV
  does not carry the successor.
- **Second-operation oracle passes**: build a routing fit for that operation (a new declared script).
  **Oracle fails everywhere**: write up why succession is special. Its OV reads the item token directly; the other
  operations are not carried in any small head group's value path.

## Log

### 2026-11-20 (14:40): queue complete. Replace vs add, and overall summary

**Replace vs add (0.6B, archived programme, independent sets):**

| Install mode | Gain | Successor | Native seq | Word lookup | WikiText-val |
|---|---|---|---|---|---|
| unedited | — | 0.089 | 0.593 | 0.650 | 2.943 |
| replace (paper: native q/k rows zeroed, programme in band) | 3 | 0.579 | 0.230 | 0.525 | 3.270 |
| band only (programme overwrites band dims, other 64 dims native) | 2 / 3 | 0.089 / 0.094 | 0.634 / 0.644 | 0.615 / 0.605 | 3.081 / 3.260 |
| add (programme added to native rows) | 2 / 3 | 0.127 / 0.149 | 0.617 / 0.623 | 0.650 / 0.645 | 2.949 / 2.972 |

- **Prediction refuted.** Keeping the native query/key weights preserves sequences, but it also removes the
  routing: the head's native attention dominates, and the closed-form programme cannot override it. Replacement is
  what makes the closed form work, and it is also what destroys the head's native function. In the closed form,
  success and preservation are coupled through replacement.
- So gradient training (Q: 0.66 success with sequences 0.62) is doing something the closed form cannot do: it
  learns rows that route lookups while keeping native sequence attention. This is a genuine advantage of
  optimisation over this construction, not just an artefact of replacing versus adding.
- Side observation: band-only at gain 3 costs as much WikiText loss (3.260) as the full edit (3.270) with no
  successor gain. So much of the language-modeling cost comes from amplifying the head's output, not from the
  routing. This is consistent with the copied-head design's identical WikiText cost at matched gain.

**Overall: what the reviewer's experiments established**

| Reviewer item | Result | Effect on the paper |
|---|---|---|
| Independent LM evaluation (64k WikiText-val tokens) | Cost confirmed: +11% (0.6B), +7% (1.7B) at gain 3; largely driven by the output gain | Replace dev-sample numbers |
| Larger native-sequence evaluation (700 prompts) | **Hidden cost found:** sequences 0.59 → 0.23 (0.6B), 0.61 → 0.37 (1.7B) | Must be reported; the 10-prompt probe missed it |
| Stability across refits (5 per model) | Stable: 4-point spread on both models | Resolved |
| Replication outside Qwen (Pythia-1.4B, declared protocol) | **Replicates:** about +0.24 success, clean routing controls, +0.4% WikiText; weaker swap evidence | Main transfer evidence |
| Second redirected operation (11 operations × 2 families, staged) | **None.** Seven clean negatives; the uppercase candidate failed its plain-lookup gate and did not transfer | Report as a principled negative; explains why succession is special |
| Prompting baseline | 0.6B: the edit beats prompting (0.68 vs 0.28). 1.7B: prompting matches it (0.47 vs 0.45); edit + instruction 0.70. Demonstrations break the edit | Capability claim per size; long-prompt limitation |
| Matched gradient baseline | Gradient routing (same data and parameters) beats the closed form (0.66 vs 0.58) and preserves sequences; fine-tuning with answers reaches 0.97 | Drop any "better method" framing; the contribution is the mechanism |
| Direct vs copied at matched success | Copied preserves native sequences much better at every success level on both models; WikiText cost equal | Claim "spares native function", not "less damage" |
| LLM-designer search baseline | Deferred (not needed unless the paper claims the LLM designer is effective) | Keep that section exploratory |

**Suggested next steps**
1. Rewrite the paper around the mechanism. Use the independent-evaluation tables, the Pythia replication, the
   second-operation negatives with the "context-free OV default" explanation, and the closed-form vs gradient finding.
2. Optional cheap experiments:
   - gradient-trained routing (Q) for the copied-head design, which may give the best success/preservation trade-off;
   - Q on 1.7B and Pythia;
   - a second-operation screen with multi-token answers, to close the tokenizer gap.

### 2026-11-20 (14:20): uppercase routing on 0.6B fails its declared gate; stopped before any fit
- A0, native plain lookups of lowercase letters: lowercase 0.92, uppercase 0.00.
- A1, oracle in plain lookups: the 8 heads' outputs are set to γ × their value vectors at the queried letter's
  position. Uppercase 0.10 / 0.12 / **0.23** at γ = 1 / 2 / 3. The same oracle with 8 random heads from the same
  layers gives 0.00 uppercase (lookup intact at 0.77–0.96).
- Gate (declared at 0.30) failed, so no routing was fitted. The head group carries some uppercase signal, specific to
  these heads (0.23 vs 0.00 for controls), but without a task cue it is too weak to make uppercase the default answer.
  The few-shot oracle's 0.68 depended on the task context.
- **Conclusion for item 3:** across 11 operations and two families, succession remains the only operation whose
  answer can be redirected by changing where heads read. This is a negative result for "a second operation", and a
  substantive positive result for the paper's explanation: routing edits work where a head's OV already computes
  item → answer as a context-free default, and succession is the one such case found.

### 2026-11-20 (14:00): item 3 on Pythia-1.4B: no routable second operation; the uppercase candidate does not transfer

| Operation (Pythia) | Native | Best value patch (≤ 8 heads) | Best output patch (any group) |
|---|---|---|---|
| plural | 1.00 | 0.00 | 0.00 |
| past tense | 1.00 | 0.00 | 0.00 |
| antonym | 0.92 | 0.00 | 0.02 |
| lowercase → uppercase letter | 1.00 | **0.00** | 0.67 (top 16 only) |
| country → capital | 1.00 | 0.00 | 0.00 |
| male → female | 0.72 | — | — (native competence below 0.80; stopped) |
| capitalise, cardinal → ordinal, word ↔ digit, English → French | — | — | — (too few single-token pairs in the Pythia tokenizer) |

- Across two model families, no non-succession operation is carried through a small head group's value path, except
  uppercase on Qwen. On Pythia, uppercase information reaches the answer only when 16 heads' full outputs are copied,
  never through their value path.
- **The uppercase candidate is therefore Qwen-specific at best.** Its routing test (`uppercase_route.py`) is now
  running on Qwen 0.6B; early oracle numbers in plain lookups are 0.10–0.23 uppercase (gate 0.30).
- **Screen design gap:** the single-token-pair requirement removed 3 operations on Qwen and 4 on Pythia. A follow-up
  could score multi-token answers by their first token, or use operations with single-token items in both tokenizers.
- `queue3` finished at 13:5x; `tmp/queue3.done` exists.

### 2026-11-20 (12:40): item 4a complete for 1.7B: prompting matches the edit on the larger model
Mean successor rate over the five value types (TEST_KEYS, arrow format, 120 prompts per type):

| Prompt | 0.6B unedited | 0.6B edited (g3) | 1.7B unedited | 1.7B edited (g3) |
|---|---|---|---|---|
| plain | 0.04 | **0.68** | 0.12 | 0.45 |
| instruction | 0.07 | 0.60 | 0.10 | **0.70** |
| 3-shot, other value types | 0.26 | 0.41 | 0.37 | 0.37 |
| 3-shot, same value type | 0.25 | 0.25 | 0.37 | 0.23 |
| instruction + same-type shots | 0.28 | 0.26 | **0.47** | 0.26 |

- **0.6B:** the edit (0.68) far exceeds the best prompt (0.28), so it overcomes an inability prompting cannot.
- **1.7B: negative for the "new capability" framing.** The best prompt for the unedited model (0.47) matches the
  plain-prompt edit (0.45). On the larger model the edit installs a default; it does not unlock a capability. The
  paper must state this by model size.
- **Positive: edit and instruction combine on 1.7B (0.70),** well above either alone.
- **Confirmed on both sizes:** prepending composed demonstrations hurts the edited model (it drops to 0.23–0.41).
  This is the long-context routing limitation.
- Next: the second-operation screen on Pythia is running (plural negative so far).

### 2026-11-20 (11:40): items 1, 1b and 4b complete for Qwen3-1.7B: same picture as 0.6B
Same sets and protocol as 0.6B. The spare head by rule is again h4 (group h4/h5); removing it slightly *lowers* dev
WikiText loss.

| Edit (1.7B) | Successor | Native seq | Word lookup | WikiText-val |
|---|---|---|---|---|
| unedited | 0.104 | 0.611 | 0.640 | 2.619 |
| direct g1 / g2 / **g3 (paper)** / g4 | 0.288 / 0.396 / **0.437** / 0.469 | 0.493 / 0.406 / **0.373** / 0.361 | 0.605 / 0.595 / 0.565 / 0.475 | 2.652 / 2.709 / **2.811** / 2.951 |
| copied g1 / g1.5 / g2 / g3 / g4 / g6 | 0.324 / 0.384 / 0.413 / 0.443 / 0.470 / 0.510 | 0.624 / 0.606 / 0.563 / 0.466 / 0.411 / 0.370 | 0.625 / 0.610 / 0.580 / 0.540 / 0.470 / 0.275 | 2.639 / 2.665 / 2.702 / 2.809 / 2.954 / 3.332 |
| refits 1–5 (direct g3) | 0.386–0.422 | 0.413–0.490 | 0.540–0.585 | 2.789–2.802 |

- **Independent evaluation:** at the reported gain 3, successor rises from 0.10 to 0.44 on fresh prompts. The cost
  is native sequences 0.61 → 0.37 and WikiText-val loss +7.3%. The sequence cost was again invisible in the
  10-prompt development probe.
- **Refits: stable.** All five pick ridge 1.0, with held-out attention 0.71–0.79. Successor 0.386–0.422, a
  4-point spread; the archived fit (0.437) is at the top of the range. The bilinear forms differ more from the
  archived fit than on 0.6B (cosine ~0.32 vs 0.55), yet behaviour is the same: different routing matrices, same
  function. The refits are gentler on sequences (0.41–0.49) than the archived fit (0.37).
- **Copied vs direct: the copied design dominates at every matched success level, as on 0.6B.** At ≈0.40 success:
  copied g1.5–2 keeps sequences at 0.56–0.61, against 0.41 for direct g2. At ≈0.44: copied g3 0.47 vs direct g3 0.37.
  WikiText cost is equal at matched success (2.809 vs 2.811 at g3).
- **Ceiling:** success saturates around 0.45–0.51 on 1.7B (around 0.6 on 0.6B), and high gains destroy word lookup
  (copied g6: 0.275).

### 2026-11-20 (07:55): item 4c, learning comparisons on the independent sets: the closed form is not the best fitting method
Same sets as `robustness.json` (1,800 fresh successor prompts, 700 native sequences, 64k WikiText-val tokens).
300 Adam steps, batch 32. Fm and F train on composed examples from numwords/days/ordinals; letters and months are
held out.

| Edit (0.6B) | Parameters trained | Successor | letters\* | months\* | Native seq | Word lookup | WikiText-val |
|---|---|---|---|---|---|---|---|
| unedited | — | 0.089 | 0.02 | 0.01 | 0.593 | 0.650 | 2.943 |
| closed form, direct g3 (paper) | 0 (closed form) | 0.579 | 0.53 | 0.59 | 0.230 | 0.525 | 3.270 |
| closed form, copied g3 | 0 (closed form) | 0.597 | 0.55 | 0.61 | 0.366 | 0.515 | 3.255 |
| **Q**: same routing labels + same params, gradient, g3 | 262k | 0.656 | 0.68 | 0.63 | **0.617** | 0.520 | 3.185 |
| **Fm**: same params, composed answers | 393k | 0.781 | 0.39 | 0.78 | 0.639 | 0.660 | 3.015 |
| **F**: layers 18–24 attention, composed answers | 44M | **0.974** | 0.88 | 0.99 | 0.631 | 0.850 | 2.973 |

\* held out from Fm/F training (never seen by the closed form or Q either). Per-type rates average the 3 formats.

- **Q is the critical comparison and the closed form loses it.** With identical supervision (routing labels only,
  no successor targets) and identical parameters, gradient training is better on success (0.66 vs 0.58), fully
  preserves native sequences (0.62 vs 0.23) and costs less WikiText loss. The preprint's "gradient training on routing
  labels performs similarly" came from the development set and does **not** hold on independent sets.
  Likely cause: the closed form **replaces** A's entire query/key rows (zeros outside the band), destroying the head's
  native sequence attention. Q starts from the native rows and adds the routing.
- **Fine-tuning with answers dominates everything** (F: 0.97, including 0.88/0.99 on held-out letters/months, and only
  +1% WikiText). Fm shows the 393k-parameter subspace can reach 0.78, but it transfers poorly to held-out letters
  (0.39).
- **Implication for the paper:** the contribution cannot be "a better editing method". It is the mechanism. A
  successor computation can be reused by changing only where one head reads, without successor targets; causal
  controls tie the effect to that head's OV; it transfers to Pythia; and succession is special because only there
  does a head's OV map item to answer. Present the closed form as the transparent construction that demonstrates
  this, with Q as the better-preserving way to fit the same edit.
- **Proposed next experiment (cheap, declared before running):** an *additive* closed form. Add the programme to
  A's native query/key rows instead of replacing them, or install it only in the band and leave the other dimensions
  native. Test whether it recovers Q's preservation. This would separate "closed form vs gradient" from
  "replace vs add".

### 2026-11-20 (07:15): item 3, second-operation screen on Qwen3-0.6B: one candidate, seven clean negatives
8-shot "x -> f(x)" prompts. The top 16 heads were ranked by attribution patching and verified by ablation. The
oracles replace the heads' final-position outputs with values from a donor prompt that is identical except for the
query item d. The score is the rate of answering f(d).

| Operation | Native | Best value patch (≤ 8 heads) | Best output patch (any group) | Verdict |
|---|---|---|---|---|
| capitalise | 1.00 | 0.00 | 0.00 | negative |
| plural | 1.00 | 0.00 | 0.00 | negative |
| past tense | 1.00 | 0.00 | 0.00 | negative |
| antonym | 0.90 | 0.00 | 0.00 | negative |
| male → female | 0.98 | 0.00 | 0.02 | negative |
| country → capital | 1.00 | 0.00 | 0.00 | negative |
| English → French | 1.00 | 0.00 | 0.00 | negative |
| **lowercase → uppercase letter** | 1.00 | **0.68 (top 8 jointly)** | 0.97 (top 8) | **routable candidate** |
| cardinal → ordinal, word ↔ digit | — | — | — | not tested: too few single-token pairs in Qwen (screen design gap) |

- **The negatives are strong.** Even copying the full outputs of 16 heads from the donor prompt never produces f(d);
  the model keeps answering f(own item). For these operations the item's identity reaches the answer through the
  item token's own residual stream (the MLP pathway), not through any head group's output. The heads that matter
  carry *which task*, not *which item*. A routing edit, which only changes where heads read, has nothing to redirect.
  This is the paper's explanation of why succession is special: the successor head's value and output projections
  map item to answer, so moving its read position moves the answer.
- **Uppercase:** single heads and groups of 2/4 give 0.00. The joint top 8 gives 0.68 (value patch) and 0.97
  (output patch). Patching all 16 heads drops it to 0.03 (interference from heads ranked 9–16).
  Caveat: uppercasing preserves letter identity, so reviewers may treat it as near-copying. It is weaker than antonym
  or translation would be, but it is not succession, and it needs a group of cooperating heads, as the reviewer
  suggested.
- **Follow-up:** `experiments/compose/uppercase_route.py` (protocol declared). It first tests the oracle in **plain
  lookups with no task cue**, where routing would have to make uppercasing the default, with a gate of uppercase
  ≥ 0.30. Only then does it fit routing into the 8 heads. It runs from `tmp/queue4.sh` after queue3.
- The next job, learning comparisons, is running (`learning_ind`).

### 2026-11-20 (05:40): item 2, Pythia-1.4B-deduped: replicates under the declared protocol
- **S1 locate:** the rule picked L14h6. Its oracle two-step rate is 0.26, the only head above the 0.20 threshold
  (next best L13h1 at 0.14). L6h3 has the largest margin drop (−1.09) but a two-step rate of 0.00. This is a much
  weaker successor head than Qwen's.
- **S2 native:** word lookup 0.37 (Qwen 0.6B: 0.65). Pythia is natively better at the composed task: successor
  0.18–0.38 on arrow prompts.
- **S3 fit:** ridge 1.0, held-out word attention on the queried value rises from 0.06 (native) to 0.47; argmax 0.84.
  The programme lives in the 96 non-rotary dimensions.
- **S4 gain:** the development curve on days is flat (0.625, 0.63, 0.62, 0.615, 0.56 for gains 1, 2, 3, 4, 6), so the
  rule chose gain 2.
- **S5 confirmation** (TEST_KEYS, 120 prompts per cell; days is a development type). Mean over value types:

  | Format | Native → edited | Wins / losses | A removed | Swap → donor's successor | Programme in 3 other heads | Wrong output weights |
  |---|---|---|---|---|---|---|
  | arrow | 0.290 → **0.558** | 181 / 20 | 0.197 | 0.307 | 0.276 | 0.212 |
  | colon | 0.240 → **0.400** | 119 / 23 | 0.137 | 0.282 | 0.226 | 0.155 |
  | equals | 0.212 → **0.467** | 172 / 19 | 0.127 | 0.332 | 0.201 | 0.137 |

  Excluding days: 0.217 → 0.458. The routing controls are clean in all 15 cells: the programme in other heads, or
  with another head's output projection, stays at or below native. The effect again requires this head's own
  value/output computation.
- **Weaker than Qwen:** the swap test moves the answer to the donor's successor on only about 0.3 of prompts (Qwen
  about 0.4–0.7). Letters gain only about +0.1 outside the arrow format, and colon/days is null (10 wins, 7 losses).
- **S6 cost: much cheaper than Qwen.** WikiText-val loss 2.917 → 2.930 (+0.4%). Native sequences 0.73 → 0.72. Word
  lookup 0.37 → 0.33. Probably due to the lower gain (2), a less load-bearing head, and the non-rotary band, which
  does not disturb position-dependent attention.
- **Verdict:** a positive replication in an unrelated family, with the head located by rule and gain chosen on one
  development type. The effect is about +0.24 absolute success at almost no language-modeling cost. Mechanistic
  evidence (the swap test) is weaker than in Qwen. Paper framing: "transfers, with a smaller and less exclusive
  effect".
- Queue thresholds were lowered (the queue had stalled waiting for 9 GB free to re-check finished Pythia work).
  The queue was relaunched; it skips finished jobs.

### 2026-11-20 (04:20): items 1, 1b, 4a and 4b (curve) complete for 0.6B
- **Infrastructure.** The earlier queue shells had survived their "kills" as orphans. `queue1` and `queue2` were
  then running duplicate jobs (two Pythia processes at once), which caused the memory pressure. Both were stopped and
  replaced by `tmp/queue3.sh`, which skips finished jobs, holds a lock, waits for free memory and waits for any orphan
  experiment process. A session cron watchdog (every 20 min) relaunches it if it is killed. The user authorized
  automatic retries. Docker stays open (the user needs it).
- **Refit stability (1b): stable.** Five fits on disjoint word pools all select ridge 1.0. Their held-out word
  attention is 0.73–0.86, and their bilinear forms have cosine ~0.55 with the archived fit (the archived fit used a
  different ridge). Successor rate at gain 3 is **0.569–0.609** (archived fit 0.579); native sequences 0.29–0.37;
  WikiText-val loss 3.21–3.32. The spread is about 4 points of success, below the 10-point instability threshold.
  The construction is stable, and the archived fit is typical, not a lucky draw.
- **Direct vs copied, full curve (4b): copied dominates at every matched success level.**

  | Success ≈ | Direct (gain): seq / WikiText | Copied (gain): seq / WikiText |
  |---|---|---|
  | 0.35 | g1: 0.303 / 3.007 | g1: 0.581 / 2.979 |
  | 0.54 | g2: 0.251 / 3.106 | g2: 0.464 / 3.083 |
  | 0.58–0.60 | g3: 0.230 / 3.270 | g3: 0.366 / 3.255 |
  | 0.61 | g4: 0.211 / 3.492 | g4: 0.297 / 3.485 |

  The copied design preserves native sequences far better (at low gain it is nearly untouched: 0.58 vs native 0.59).
  Its WikiText cost is equal or marginally lower. The direct edit loses about half of sequence ability even at gain 1.
  Revised claim: **copying spares the native successor function; neither design spares general language modeling,
  whose cost is set by the gain.**
- **Prompting baseline (4a): the edit overcomes a demonstrated inability, at least for 0.6B.** Test set: TEST_KEYS,
  arrow format, 120 prompts per value type. Mean successor rate:
  - unedited model: plain 0.04; instruction 0.07; 3-shot from other types 0.26; 3-shot from the same type 0.25;
    instruction plus same-type shots 0.28;
  - edited model, plain prompt: **0.68**.

  Demonstrations stop the copying (lookup falls from ~0.8 to ~0.02) but do not produce the successor.
  **New limitation:** the edit degrades when the prompt is longer. With demonstrations prepended, the edited model
  falls to 0.25–0.41, presumably because the routing programme mis-targets among the many earlier value tokens.
  Report this in the paper.
- **02:50: the queue shell was killed by the low-memory reaper again.** The robustness 0.6B Python process
  survived as an orphan and keeps saving conditions. The jobs after it will not start. At the time, the commit charge
  was 51 of 55 GB: Roblox Studio (2 instances, ~7 GB committed), the Docker/WSL VM (~8 GB) and the robustness process
  (~7 GB). The next queue needs Roblox and Docker closed.
- **Copied vs direct (0.6B, independent sets, interim):**

  | Edit | Successor | Native sequences | Word lookup | WikiText-val |
  |---|---|---|---|---|
  | unedited | 0.089 | 0.593 | 0.650 | 2.943 |
  | copied g1 | 0.373 | 0.581 | 0.675 | 2.979 |
  | copied g1.5 | 0.483 | 0.529 | 0.640 | 3.022 |
  | copied g2 | 0.541 | 0.464 | 0.625 | 3.083 |
  | copied g3 | 0.597 | 0.366 | 0.515 | 3.255 |
  | copied g4 | 0.612 | 0.297 | 0.400 | 3.485 |
  | copied g6 | 0.587 | 0.231 | 0.140 | 4.067 |
  | direct g3 (reported) | 0.579 | 0.230 | 0.525 | 3.270 |

  At matched success (about 0.58–0.60), the copied design keeps more native sequence ability (0.37 vs 0.23).
  Its language-modeling cost and word-lookup cost are the same as the direct edit's. So the paper may claim the copied
  design spares the successor head's native role; it should **not** claim that the copied design reduces general
  damage. Both designs' cost grows steeply with gain, and successor accuracy saturates near 0.6. Direct g1, g2 and g4
  are still to come for the full curve.
- The Pythia smoke test finished all six stages (S1–S6). The overnight queue (`tmp/queue2.sh`) was launched right after.
- The user closed other applications and asked for an overnight run under this living plan. Free memory was about
  8 GB of 31 GB, with Docker's WSL VM using about 5.4 GB. WikiText scoring batch reduced to 2.
- Pythia smoke test (tiny sizes, scan limited to layers 14–15) ran through S1–S5. Pipeline OK; its numbers are not
  meaningful.

### 2026-11-19
- Wrote `robustness.py`, `prompting.py`, `xfamily.py`, `second_op2.py` and `learning_ind.py`. Downloaded
  Pythia-1.4B-deduped. The Pythia adapter was checked: head row layout verified, native continuation and lookup work.
- **First independent-evaluation result (0.6B, reported edit, gain 3):** successor 0.089 → 0.579 (1,800 fresh prompts);
  **native sequences 0.593 → 0.230** (700 prompts); word lookup 0.650 → 0.525; WikiText-val loss 2.943 → 3.270 (+11%).
  The development probe (10 prompts) had hidden a large sequence-continuation cost. This must appear in the paper.
- The run was killed by the low-memory reaper (concurrent Pythia smoke test). Saved conditions resume.
- Earlier second-operation search (archived): no single head's value patch gives f(donor) for capitalise, plural,
  past or antonym, in either Qwen model (0.00 everywhere).
