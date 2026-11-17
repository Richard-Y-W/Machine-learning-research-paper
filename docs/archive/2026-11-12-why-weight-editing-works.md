# Why Direct Weight Editing Works: TDD Implementation Plan


**Goal:** Build an experiment harness that explains why direct weight editing succeeds or fails in pretrained transformers, and tests whether those explanations predict results on new tasks and models.

**Architecture:** Keep the frozen study definition, editor workspace, model execution service, and private evaluator separate. Record each attempted edit and its measured effects so diagnosis, numerical implementation, feedback, and collateral changes can be examined independently. Use small, randomly initialized transformers for software tests and pretrained transformers for research evidence.

**Tech stack:** Python 3.11+, PyTorch, Hugging Face Transformers and safetensors, NumPy, pytest, JSON/JSONL manifests, and the repository's `python -m experiments` command dispatch. A separate optional dependency file and lock record will keep transformer tooling out of archived-pilot replay requirements.

**Status:** Proposed work, November 12, 2026. No transformer experiment, new model download, or paid editor call is reported by this plan. Test snippets below are implementation specifications; they have not been executed as implemented features.

---

## 1. Central research question

> **Why does direct weight editing work in pretrained transformers, and how do the model's representations, the requested change, and the editor's decisions determine whether the change generalizes while preserving other capabilities?**

The intended contribution is an evidence-supported explanation of the process. The question covers useful natural edits, controlled repairs, model scale, and transfer between families. It is not restricted to fault localization, one coefficient, one kind of perturbation, or one benchmark.

An explanation must do more than describe a successful edit afterward. It must identify a measurable property, survive a relevant intervention, and make a prediction on cases that were not used to develop it. Different mechanisms may explain different classes of edits; the study must allow that result.

The question itself is established. Novelty would come from a specific supported explanation, a result that distinguishes competing explanations, or a predictor that transfers under a declared protocol. This plan does not claim that larger models, controlled corruption, location hints, or prediction of editability are individually new.

### Connection to the completed pilot

The small dynamics pilot found that a numerical patch can implement the editor's chosen correction accurately while missing the requested physical target. Source-model approximation also affects that comparison. Use this as motivation for separating intended change, chosen change, implemented change, and observed behavior. It does not establish a transformer mechanism or require further toy-network experiments before the new study.

The active study excludes the historical fine-tuning, distillation, Jacobian, and residual baseline suite. Keep archived results reproducible. Add a comparison only when it distinguishes an explanation or supports a stated comparative claim.

## 2. Explanations to test

These are competing or complementary hypotheses, not assumed findings.

| Explanation | Measurement made before final evaluation | Intervention that tests it | Evidence against the explanation |
|---|---|---|---|
| **Choosing a useful change is the bottleneck.** | The editor's proposed component, direction, and predicted effect on development examples. | Pair ordinary editing with independently verified diagnostic information; controlled cases supply true affected locations without giving the repair. | Diagnostic assistance has little effect, or proposed diagnoses fail direct behavioral checks. |
| **A useful behavioral change is hard to implement in weights.** | Effect of a temporary activation intervention and the editor's intended numerical operation. | Compare that temporary intervention with the permanent weight edit on the same development inputs; remove the temporary hook before testing the checkpoint. | Weight edits reproduce the intended internal change but behavior still fails to improve. |
| **The desired change interferes with other computations.** | Effects on target and protected development examples at several edit strengths. | Compare zero, half, full, and reversed edit strength; where relevant, remove one edited component at a time. | Target improvement and preservation do not follow the predicted strength or component dependence. |
| **Feedback and search account for apparent editing ability.** | Candidate history, evaluated examples/tokens, and quality of the candidate selected at each frozen budget checkpoint. | Replay candidates selected at half and full budget, with selection based only on development data. | Additional feedback does not improve held-out outcomes, or gains disappear outside the examples used for selection. |
| **The edit changes access to existing behavior or answer completion.** | Completion, repetition, correct answers, and responses under a fixed diagnostic prompt protocol. | Compare these outcomes with decoding and token limits held constant. | The hypothesized completion or elicitation effect is absent while accuracy changes. |

Temporary activation interventions and component ablations are diagnostic experiments, not a new benchmark tournament. Use them on the mechanism under investigation. A correlation between an activation and success does not establish causality. A failed edit does not establish absence of a capability, and an accuracy gain does not establish acquisition of a new capability.

Location information does not remove all diagnosis: it need not reveal the right direction, magnitude, or editing site. Report its effect as the value of that information, not a complete decomposition of all reasoning errors.

## 3. Models, tasks, and study stages

### Target models

| Checkpoint | Role |
|---|---|
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | Develop the harness and candidate explanations. |
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Main larger-model study using the frozen protocol. |
| [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Test predictions on a second family. |
| [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | Test a frozen explanation at substantially greater scale when the execution budget supports it. |

These are candidates, not evaluated models. Select exact checkpoint revisions and record file hashes, tokenizer, chat template, dtype, decoding, hardware, and software versions in each run manifest. Use unquantized BF16 or FP32 weights initially; quantization would change the intervention and requires a separately declared condition. Released sizes differ in training and architecture, so a size comparison alone cannot identify a causal effect of parameter count.

Use one fixed editor model for the first experiment block. Test a second editor only when evaluating whether an explanation transfers across editors. Record model version and sampling settings; repeated attempts are not independent pretrained checkpoints.

### Tasks

1. **Mathematical problem solving:** GSM8K and a frozen MATH subset. Measure correct answers, completion, and repetition separately. Select one as the development task and reserve the other for transfer. Do not select problems after inspecting edit results.
2. **Factual updates:** CounterFact-style requested changes with paraphrases and neighboring facts. The requested new fact belongs in the editing prompt; held-out paraphrases and neighboring questions remain private. A deliberate factual update is not necessarily repair of a natural model error.
3. **Instruction following:** A frozen partition of IFEval plus disjoint development examples using the same instruction types. Use the official checks. When instruction following is the editing target, use math and factual retention as protected behavior; when math is the target, instruction following supplies a preservation measure.

Split by underlying problem, fact, or instruction template before expanding paraphrases. Record overlap by normalized text and group identifier. Public benchmarks can have pretraining contamination; withholding them from the editor does not prove absence from either model's training data.

### Ordered study stages

- **A — Offline correctness:** Complete the TDD tasks with synthetic tensors, a tiny random transformer, scripted editor responses, and fixture data. No research conclusion follows from these tests.
- **B — Development on a pretrained model:** On Qwen3-1.7B, run controlled repairs and a small natural-edit cohort. Develop candidate explanations and measure runtime and variability. A first controlled cohort uses three independent perturbation realizations, two support patterns, and two information conditions: 12 attempts. A first natural cohort uses three independent editor attempts per task: nine attempts. These are exploratory counts, not a powered reliability claim.
- **C — Freeze predictions:** Freeze the explanation, predictor, primary contrast, success rule, exclusions, replication count, budget, and analysis code before opening the next cohort. Include an explicit prediction of no benefit when the explanation implies one. Register the cohort by a versioned manifest and Git commit.
- **D — Larger-model and family tests:** Evaluate the frozen predictions on Qwen3-8B and Llama-3.1-8B-Instruct. Reuse the same explanation and evaluation rules. If revised afterward, label the revision exploratory and use a new untouched cohort for its confirmation.
- **E — Scale and boundary:** Use Qwen3-32B for the specific predictions that distinguish scale-related explanations. Report omitted stages as untested. Do not run every mechanism, task, size, and editor combination automatically.

Choose confirmatory replication using development variability and a declared precision target, subject to a recorded spending cap. Do not extend a cohort until a significance threshold is crossed. If the affordable cohort cannot support the intended precision, report uncertainty and limit the claim.

### Controlled repairs

Keep a private clean checkpoint. Generate additive perturbations supported in one block or spread across several blocks. Match damage using a calibration split distinct from editor development and final evaluation; use a prespecified candidate order and report unmatched cases. Record relative perturbation norm, number of affected parameters, rank, component types, and initial task loss. Comparable accuracy loss does not make the two perturbation distributions identical.

Use fresh paired attempts on the same damaged checkpoint with and without true affected-location information. Keep all permitted edit locations and numerical budgets identical. The editor cannot access the clean weights, perturbation seed, inverse patch, calibration labels, or final evaluation data. Public filenames must not encode the condition or corruption location. The clean model proves a repair exists within the permitted edit scope; it does not prove that the damaged location is the only useful repair location.

The main outcomes concern behavior, not reconstruction of the original weights. Multiple different patches may implement an adequate repair.

## 4. What is a software test, and what is scientific evidence?

| Software invariant: must pass | Scientific hypothesis: may be supported or rejected |
|---|---|
| A failed patch leaves the checkpoint unchanged. | Selective edits preserve unrelated behavior. |
| Withheld data never reaches the editor service. | An explanation transfers to unseen tasks. |
| Replaying an accepted patch reproduces its stored tensors. | A temporary activation effect can be implemented by a permanent edit. |
| A paired attempt starts from the same checkpoint and budget. | Better diagnostic information improves editing. |
| Failed and incomplete attempts remain in the denominator. | Editing succeeds reliably under the selected budget. |
| Prediction files predate access to confirmation outcomes. | The predictor outperforms a simple headroom-only predictor. |

Never write a unit test requiring the research hypothesis to be true. Tests should catch corrupted evidence, incorrect arithmetic, leakage, mistaken pairing, and unrecorded resource use. Scientific null results must remain valid outputs of a passing harness.

## 5. Repository layout and contracts

Follow the existing `experiments` and `tests` organization. Create a focused subpackage rather than extending the archived `wsp_common.py` with transformer code.

```text
experiments/editing/
  __init__.py           package marker
  protocol.py           manifests, identifiers, disjoint splits, freeze hashes
  patches.py            finite validated tensor updates and checkpoint restoration
  perturbations.py      controlled damage, calibration pairing, private repair data
  backend.py            local model loading, forward passes, activation capture
  budget.py             evaluation/editor/cost reservations and durable ledger
  session.py            editor-visible messages, actions, and independent attempts
  evaluate.py           task scoring and per-example outcomes
  mechanisms.py         dose, component, and activation-versus-weight comparisons
  analyze.py            paired effects, clustered uncertainty, frozen predictions
  cli.py                prepare, dry-run, run, evaluate, analyze, verify commands
tests/editing/
  conftest.py           deterministic synthetic cases and tiny local transformer
  test_protocol.py
  test_patches.py
  test_perturbations.py
  test_backend.py
  test_budget.py
  test_session.py
  test_evaluate.py
  test_mechanisms.py
  test_analyze.py
  test_cli.py
requirements-editing.txt
docs/editing-protocol.md
```

Generated runs live under `results/editing/<study-id>/`. Keep model downloads, checkpoint shards, private evaluator data, transient activations, and editor workspaces outside Git. Commit compact manifests, public prompts, patch descriptions, metrics, and reproduction instructions when the cohort is ready for release. Private partitions become releasable only after their role as untouched evaluation data has ended.

### Required records

- **Study:** schema version; research question; stage; model and tokenizer revisions and hashes; editor identity; task and data hashes; dtype and decoding; component edit permissions; sample count; budget; success thresholds; primary contrast; prediction hash; environment record.
- **Case:** opaque case ID; private source/group ID; target checkpoint hash; task; condition; independent realization ID; attempt ID; permitted feedback partition. Source and condition mappings remain evaluator-private when they would reveal the experimental manipulation.
- **Action:** sequence number; public tool request; patch parent hash; response or error; reserved and actual resources; timestamp. Preserve public justifications, not hidden reasoning traces.
- **Outcome:** case ID; candidate hash; item ID; correctness; completion; repetition; preservation score; error category; elapsed time; usage; protocol status. An invalid attempt is recorded as unsuccessful with its reason.

Use JSON with `allow_nan=False`. State transitions are `prepared -> running -> frozen -> evaluated`; terminal alternatives are `budget_stopped`, `editor_error`, and `invalid_patch`. A research failure is not a harness crash. A process crash retains reservations and can resume only from its logged state, without generating a replacement attempt under the old ID.

## 6. TDD execution rule

For every task: create the specified test; run it and observe the intended failure; implement the smallest behavior that satisfies it; run the affected tests; refactor; commit. A missing import is only a setup failure: after adding a minimal module and function signature, obtain an assertion failure or explicit `NotImplementedError` before implementing the behavior. Save the failing and passing command output with the implementation review.

The code snippets define initial contracts, not a complete production implementation. Extend them with the explicitly listed cases in the same task before calling that task complete. All named public functions below must be implemented in the file assigned to them; fixtures are defined in the task that introduces them.

### Task 1 — Freeze identities and data boundaries

**Files:** Create `experiments/editing/__init__.py`, `experiments/editing/protocol.py`, and `tests/editing/test_protocol.py`.

- [ ] Write this failing test:

```python
import pytest
from experiments.editing.protocol import freeze_hash, assert_disjoint

def test_freeze_hash_ignores_key_order_but_detects_budget_change():
    assert freeze_hash({'seed': 7, 'budget': 10}) == freeze_hash({'budget': 10, 'seed': 7})
    assert freeze_hash({'budget': 10}) != freeze_hash({'budget': 11})

def test_paraphrases_cannot_cross_group_partitions():
    with pytest.raises(ValueError, match='group overlap'):
        assert_disjoint({'dev': {'fact-1'}, 'test': {'fact-1'}})
```

- [ ] Run `python -m pytest tests/editing/test_protocol.py -q`; after creating importable signatures, expect the assertions to fail on the missing behavior.
- [ ] Implement canonical hashing and group checks in `protocol.py`:

```python
import hashlib
import json

def freeze_hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()

def assert_disjoint(partitions):
    seen = set()
    for groups in partitions.values():
        if seen.intersection(groups):
            raise ValueError('group overlap')
        seen.update(groups)
```

- [ ] Add tests for overlapping normalized question text despite different IDs; changed tokenizer/decoding hashes; duplicate case IDs; missing required manifest fields; and rejection of mutable model revisions such as `main`. Implement field validation against the records in section 5. Do not silently supply live model versions or budgets.
- [ ] Rerun `python -m pytest tests/editing/test_protocol.py -q`; expect all cases to pass.
- [ ] Commit with `git add experiments/editing/__init__.py experiments/editing/protocol.py tests/editing/test_protocol.py` and `git commit -m "Add frozen editing manifests and split checks"`.

### Task 2 — Apply patches atomically and restore checkpoints exactly

**Files:** Create `experiments/editing/patches.py` and `tests/editing/test_patches.py`.

- [ ] Write the failure test before the patch function:

```python
import pytest
import torch
from experiments.editing.patches import apply_replacements

def test_invalid_second_tensor_cannot_partly_apply_first_tensor():
    source = {'a': torch.ones(2), 'b': torch.ones(2)}
    with pytest.raises(ValueError):
        apply_replacements(source, {'a': torch.zeros(2), 'b': torch.tensor([float('nan'), 0.])})
    assert torch.equal(source['a'], torch.ones(2))

def test_successful_patch_does_not_mutate_source():
    source = {'a': torch.ones(2)}
    result = apply_replacements(source, {'a': torch.zeros(2)})
    assert torch.equal(result['a'], torch.zeros(2))
    assert torch.equal(source['a'], torch.ones(2))
```

- [ ] Run `python -m pytest tests/editing/test_patches.py -q` and observe the intended failure.
- [ ] Implement validation before writing any tensor:

```python
import torch

def apply_replacements(source, replacements):
    prepared = {}
    for name, value in replacements.items():
        if name not in source or value.shape != source[name].shape:
            raise ValueError('unknown tensor or wrong shape')
        if not torch.isfinite(value).all():
            raise ValueError('nonfinite replacement')
        cast = value.detach().to(device=source[name].device, dtype=source[name].dtype)
        if not torch.isfinite(cast).all():
            raise ValueError('replacement overflows storage dtype')
        prepared[name] = cast.clone()
    return {name: prepared.get(name, value) for name, value in source.items()}
```

The returned mapping shares untouched tensors read-only. The serving process must not expose those tensors for mutation. For actual checkpoints, validate and stage all changed tensors in a new safetensors artifact, then atomically publish its manifest; do not clone the entire 32B model per candidate. Keep the original changed tensor bytes so restoration does not depend on subtracting a rounded delta.

- [ ] Add failing tests for storage-dtype overflow, wrong parent-checkpoint hash, disallowed tensors, tied parameters, duplicate writes in serialized patch lists, and byte-exact restoration after BF16 edits. Count changed stored values after casting, separately from requested changes.
- [ ] Rerun `python -m pytest tests/editing/test_patches.py -q`; expect all cases to pass.
- [ ] Commit with `git add experiments/editing/patches.py tests/editing/test_patches.py` and `git commit -m "Validate and replay transformer weight patches"`.

### Task 3 — Generate controlled damage without revealing its solution

**Files:** Create `experiments/editing/perturbations.py` and `tests/editing/test_perturbations.py`.

- [ ] Write the failing selection test:

```python
from experiments.editing.perturbations import match_damage

def test_unmatched_damage_is_reported_instead_of_cherry_picked():
    local = [{'id': 'l0', 'drop': .10}, {'id': 'l1', 'drop': .30}]
    spread = [{'id': 'd0', 'drop': .11}]
    pairs, unmatched = match_damage(local, spread, tolerance=.02)
    assert pairs == [('l0', 'd0')]
    assert unmatched == ['l1']
```

- [ ] Run `python -m pytest tests/editing/test_perturbations.py -q` and observe the intended failure.
- [ ] Implement deterministic matching using calibration outcomes only:

```python
def match_damage(local, spread, tolerance):
    remaining = {row['id']: row for row in spread}
    pairs, unmatched = [], []
    for left in sorted(local, key=lambda row: row['id']):
        candidates = sorted(remaining.values(), key=lambda row: (abs(left['drop'] - row['drop']), row['id']))
        if candidates and abs(left['drop'] - candidates[0]['drop']) <= tolerance:
            right = candidates[0]['id']
            pairs.append((left['id'], right))
            del remaining[right]
        else:
            unmatched.append(left['id'])
    return pairs, unmatched + sorted(remaining)
```

- [ ] Test deterministic perturbation generation from an evaluator-private seed, support in exactly one versus several declared blocks, finite stored values, and exact clean restoration. Normalize each sampled additive direction by its Frobenius norm before applying the declared scale. Calibration tries a fixed ordered scale grid and retains every attempted scale and rejection reason.
- [ ] Test that matching accepts only calibration records, not development or final records. Reject NaN drops, duplicate IDs, and negative tolerance. Test that editor-visible packages contain neither seed, clean tensor, inverse patch, condition-encoded name, nor private path. The information arm reveals only the allowed diagnostic fields.
- [ ] Rerun `python -m pytest tests/editing/test_perturbations.py -q`; expect all cases to pass.
- [ ] Commit with `git add experiments/editing/perturbations.py tests/editing/test_perturbations.py` and `git commit -m "Add paired transformer repair fixtures"`.

### Task 4 — Exercise the real model interface offline

**Files:** Create `experiments/editing/backend.py`, `tests/editing/conftest.py`, `tests/editing/test_backend.py`, and `requirements-editing.txt`.

- [ ] Define the local test fixture without downloading weights:

```python
# tests/editing/conftest.py
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

@pytest.fixture
def tiny_model():
    torch.manual_seed(7)
    config = GPT2Config(vocab_size=32, n_positions=32, n_embd=16,
                       n_layer=2, n_head=2, bos_token_id=1, eos_token_id=2)
    return GPT2LMHeadModel(config).eval()
```

- [ ] Write a failing test that checks actual outputs rather than a mocked backend:

```python
import torch
from experiments.editing.backend import next_logits

def test_forward_is_repeatable_and_does_not_build_gradients(tiny_model):
    ids = torch.tensor([[1, 4, 5]])
    first = next_logits(tiny_model, ids)
    second = next_logits(tiny_model, ids)
    assert torch.equal(first, second)
    assert not first.requires_grad
    assert all(parameter.grad is None for parameter in tiny_model.parameters())
```

- [ ] Run `python -m pytest tests/editing/test_backend.py -q` and observe the intended failure.
- [ ] Implement the forward operation:

```python
import torch

@torch.inference_mode()
def next_logits(model, input_ids):
    model.eval()
    return model(input_ids=input_ids).logits[:, -1, :].detach().clone()
```

- [ ] Add tests for patch -> forward -> restore equivalence; temporary activation hooks removed even on exceptions; left-padding and attention masks; tied embedding/head handling; fixed decoding; local save/load; and a loader refusing remote code or network downloads during tests. The production loader uses a prepared local checkpoint and checks its manifest before inference.
- [ ] Add the optional Transformers, safetensors, and dataset/scoring dependencies in `requirements-editing.txt`. Resolve and record a tested lock environment for Python 3.11+; do not change archived reproduction pins. Missing transformer dependencies must produce an explicit setup failure for this test suite, not silently skip it.
- [ ] Rerun `python -m pytest tests/editing/test_backend.py -q`; expect all cases to pass with network access disabled.
- [ ] Commit with `git add experiments/editing/backend.py tests/editing/conftest.py tests/editing/test_backend.py requirements-editing.txt` and `git commit -m "Add offline transformer execution tests"`.

### Task 5 — Count resources before granting feedback

**Files:** Create `experiments/editing/budget.py` and `tests/editing/test_budget.py`.

- [ ] Write the failing test:

```python
import pytest
from experiments.editing.budget import reserve

def test_large_batch_cannot_hide_behind_one_tool_call():
    remaining = {'sequences': 4, 'tokens': 100}
    with pytest.raises(ValueError, match='budget'):
        reserve(remaining, {'sequences': 5, 'tokens': 90})
    assert remaining == {'sequences': 4, 'tokens': 100}

def test_reservation_is_atomic():
    assert reserve({'sequences': 4, 'tokens': 100}, {'sequences': 2, 'tokens': 80}) == {'sequences': 2, 'tokens': 20}
```

- [ ] Run `python -m pytest tests/editing/test_budget.py -q` and observe the intended failure.
- [ ] Implement the reservation arithmetic:

```python
def reserve(remaining, requested):
    if set(requested) != set(remaining):
        raise ValueError('budget dimensions differ')
    if any(type(value) is not int or value < 0 for value in requested.values()):
        raise ValueError('invalid budget request')
    if any(requested[key] > remaining[key] for key in remaining):
        raise ValueError('budget exhausted')
    return {key: remaining[key] - requested[key] for key in remaining}
```

- [ ] Add tests for input tokens plus maximum output tokens, paired model evaluations charged twice, teacher-forced token counts, editor token/call limits, and integer cost units. Reserve conservatively before dispatch; settle only from known actual usage; retain reservations when usage is unknown.
- [ ] Add a durable single-writer ledger with atomic file replacement and restart tests. A replay cannot spend the same reservation twice. Defaults permit zero paid calls. The existing `results/wsp-budget.json` is not reused or modified.
- [ ] Rerun `python -m pytest tests/editing/test_budget.py -q`; expect all cases to pass.
- [ ] Commit with `git add experiments/editing/budget.py tests/editing/test_budget.py` and `git commit -m "Meter editing feedback and paid requests"`.

### Task 6 — Keep editor sessions independent and evaluator data private

**Files:** Create `experiments/editing/session.py` and `tests/editing/test_session.py`.

- [ ] Write the public-message test:

```python
from experiments.editing.session import public_case

def test_private_case_fields_never_enter_editor_prompt():
    case = {'id': 'case-a', 'task': 'repair', 'instruction': 'Recover accuracy',
            'weights_hash': 'abc', 'development_ids': ['d1'],
            'clean_weights': 'secret', 'perturbation_seed': 4, 'test_ids': ['t1'],
            'affected_components': ['layer.2']}
    ordinary = public_case(case, informed=False)
    informed = public_case(case, informed=True)
    assert set(ordinary) == {'id', 'task', 'instruction', 'weights_hash', 'development_ids'}
    assert informed['affected_components'] == ['layer.2']
    assert 'clean_weights' not in informed and 'test_ids' not in informed
```

- [ ] Run `python -m pytest tests/editing/test_session.py -q` and observe the intended failure.
- [ ] Implement an allowlist rather than deleting a few known private fields:

```python
def public_case(case, informed=False):
    fields = ('id', 'task', 'instruction', 'weights_hash', 'development_ids')
    result = {key: case[key] for key in fields}
    if informed:
        result['affected_components'] = list(case['affected_components'])
    return result
```

- [ ] Test the action contract: `inspect`, `calculate`, `forward`, `propose_patch`, and `finalize`. `inspect` reads declared tensor slices; `calculate` performs numerical analysis in a separate CPU workspace; `forward` reaches only the metered model service; `propose_patch` goes through Task 2; `finalize` freezes a development-selected candidate. Malformed and unknown actions return recorded errors.
- [ ] Integration-test separate processes/filesystem mounts: the editor worker can read only its damaged/current tensors and development material; the private evaluator and clean checkpoint are not mounted. Provider credentials stay in the controller. Disable worker network access. A sanitized JSON object alone is not isolation.
- [ ] Treat no gradients, no optimizers, no fitting loop, and no unmetered model execution as protocol rules. Disabling autograd alone does not enforce all of them. Log generated code and audit it; an observed violation invalidates the attempt and remains in the report. The worker is not claimed to be a security boundary against an adversarial editor.
- [ ] Test that a new attempt resets weights, tool state, transcript, and budget. Replay public scripted responses for offline tests; mock only the external provider transport. Transport errors retain their usage reservation and do not silently create another attempt.
- [ ] Rerun `python -m pytest tests/editing/test_session.py -q`; expect all cases to pass without API credentials.
- [ ] Commit with `git add experiments/editing/session.py tests/editing/test_session.py` and `git commit -m "Separate editing sessions from private evaluation"`.

### Task 7 — Score intended changes and preservation separately

**Files:** Create `experiments/editing/evaluate.py` and `tests/editing/test_evaluate.py`.

- [ ] Write paired-scoring tests:

```python
import pytest
from experiments.editing.evaluate import paired_accuracy

def test_improvements_and_regressions_remain_visible():
    result = paired_accuracy([False, True, True, False], [True, False, True, False])
    assert result == {'gain': 0.0, 'repairs': 1, 'regressions': 1, 'n': 4}

def test_empty_evaluation_is_not_a_perfect_score():
    with pytest.raises(ValueError):
        paired_accuracy([], [])
```

- [ ] Run `python -m pytest tests/editing/test_evaluate.py -q` and observe the intended failure.
- [ ] Implement paired accuracy:

```python
def paired_accuracy(before, after):
    if not before or len(before) != len(after):
        raise ValueError('paired nonempty outcomes required')
    if any(type(x) is not bool for x in [*before, *after]):
        raise ValueError('boolean correctness required')
    repairs = sum(not old and new for old, new in zip(before, after))
    regressions = sum(old and not new for old, new in zip(before, after))
    return {'gain': (repairs - regressions) / len(before),
            'repairs': repairs, 'regressions': regressions, 'n': len(before)}
```

- [ ] Add tests for matching by item ID rather than row order; missing/duplicate items; unchanged candidates; invalid attempts; and outputs cut off at the token limit. Missing answers score incorrect. A truncation flag is not automatically a repetition flag. Repetition checks must include legitimate repeated text and unfinished nonrepetitive output.
- [ ] Integrate official task scorers with known-answer fixtures: GSM8K numeric answer extraction, the selected MATH equivalence scorer, CounterFact target/paraphrase/neighborhood scores, and IFEval checks. Record scorer revisions. Preserve raw outputs and scoring errors for audit.
- [ ] Report raw task gain and protected-task change. For controlled repairs, also report `(edited - damaged) / (clean - damaged)` only when the denominator exceeds the frozen minimum; return an explicit undefined value otherwise. Do not clip overshoot or negative recovery. Compute protected retention relative to both damaged and clean anchors, labeling each.
- [ ] Define success thresholds in the frozen study manifest, not inside the scorer. Report confidence intervals and raw metrics alongside any binary success label. Accuracy among completed responses is descriptive because editing changes which responses complete.
- [ ] Rerun `python -m pytest tests/editing/test_evaluate.py -q`; expect all cases to pass.
- [ ] Commit with `git add experiments/editing/evaluate.py tests/editing/test_evaluate.py` and `git commit -m "Score editing gains and collateral changes"`.

### Task 8 — Test mechanisms with controlled interventions

**Files:** Create `experiments/editing/mechanisms.py` and `tests/editing/test_mechanisms.py`.

- [ ] Write a test for the edit-strength experiment:

```python
import torch
from experiments.editing.mechanisms import scaled_edit

def test_zero_full_and_reverse_have_declared_meanings():
    original = torch.tensor([1., 2.])
    edited = torch.tensor([3., 1.])
    assert torch.equal(scaled_edit(original, edited, 0.), original)
    assert torch.equal(scaled_edit(original, edited, 1.), edited)
    assert torch.equal(scaled_edit(original, edited, -1.), torch.tensor([-1., 3.]))
```

- [ ] Run `python -m pytest tests/editing/test_mechanisms.py -q` and observe the intended failure.
- [ ] Implement scaling from an immutable original:

```python
import math
import torch

def scaled_edit(original, edited, strength):
    if original.shape != edited.shape or not math.isfinite(strength):
        raise ValueError('invalid dose comparison')
    value = original.double() + strength * (edited.double() - original.double())
    cast = value.to(original.dtype)
    if not torch.isfinite(cast).all():
        raise ValueError('nonfinite scaled edit')
    return cast
```

- [ ] Test that each strength starts from the original checkpoint, not the previous strength. All probes use the same item IDs, decoding, and declared randomness. Store requested and realized strengths after rounding. Reject endpoint shape/dtype mismatches in the complete implementation.
- [ ] Add a real tiny-transformer test comparing a temporary activation modification with a weight patch intended to realize it. Measure the internal-output discrepancy on shared inputs and the downstream logits separately. Include a case where internal agreement does not establish task improvement. Hooks must be absent in final checkpoint evaluation.
- [ ] Test component removal from a multi-component edit and report both individual and joint effects. Do not label nonadditive components independently sufficient. A matched-norm random-direction comparison is included only if claiming the identified direction has a specific causal role.
- [ ] Test half/full feedback-budget candidates: each is selected using only feedback available at that checkpoint. Full-budget candidates must not leak backward into the half-budget result. Prefix comparisons estimate the effect of additional search along that procedure, not the effect of all possible budget-allocation policies.
- [ ] Rerun `python -m pytest tests/editing/test_mechanisms.py -q`; expect all cases to pass.
- [ ] Commit with `git add experiments/editing/mechanisms.py tests/editing/test_mechanisms.py` and `git commit -m "Add intervention tests for editing mechanisms"`.

### Task 9 — Evaluate explanations on new cases

**Files:** Create `experiments/editing/analyze.py` and `tests/editing/test_analyze.py`.

- [ ] Write the pairing test:

```python
import pytest
from experiments.editing.analyze import mean_hint_effect

def test_hint_effect_pairs_same_case_instead_of_pooling_unmatched_runs():
    blind = {'a': .20, 'b': .40}
    informed = {'b': .45, 'a': .30}
    assert mean_hint_effect(blind, informed) == pytest.approx(.075)
    with pytest.raises(ValueError):
        mean_hint_effect(blind, {'a': .30})
```

- [ ] Run `python -m pytest tests/editing/test_analyze.py -q` and observe the intended failure.
- [ ] Implement the paired effect:

```python
def mean_hint_effect(blind, informed):
    if not blind or blind.keys() != informed.keys():
        raise ValueError('complete matched cases required')
    return sum(informed[key] - blind[key] for key in blind) / len(blind)
```

- [ ] Add tests ensuring repeats are aggregated within their independent source/case before the primary analysis. Bootstrap whole paired cases, preserving information conditions and task groups; report checkpoint-specific results. Do not bootstrap prompts as though they were independently trained models. Return an explicit insufficient-replication status when uncertainty cannot be estimated.
- [ ] Create frozen prediction records with model/task/case IDs, allowed pre-edit diagnostic features, predicted direction or probability, predictor coefficients/rules, training-cohort hash, timestamp, and Git commit. Tests must reject post-edit fields, private corruption metadata on natural tasks, and changes after the confirmation stage starts. A predictor based on injected support alone cannot be claimed to predict natural failures.
- [ ] Test predictions using held-out case groups and an untouched model family. Compare a probability predictor with an intercept-only or initial-performance-only predictor if claiming predictive value; this is the specific comparison needed for that claim. Use a proper score such as Brier score, plus uncertainty and calibration where the sample permits. For directional predictions, report the paired effect and its interval.
- [ ] Add fixtures where the mechanism prediction fails, where it adds nothing beyond available headroom, and where the same explanation transfers. All must produce valid reports. Multiple mechanism analyses are secondary unless prespecified as primary; record revisions and avoid selecting the best explanation on the confirmation outcomes.
- [ ] Rerun `python -m pytest tests/editing/test_analyze.py -q`; expect all cases to pass.
- [ ] Commit with `git add experiments/editing/analyze.py tests/editing/test_analyze.py` and `git commit -m "Test paired explanations and prospective predictions"`.

### Task 10 — Provide one complete offline path

**Files:** Create `experiments/editing/cli.py`, `tests/editing/test_cli.py`, and `docs/editing-protocol.md`; modify `experiments/__main__.py` at `COMMANDS`.

- [ ] Write the CLI acceptance test before the command exists:

```python
import json
import subprocess
import sys

def test_dry_run_produces_replayable_offline_evidence(tmp_path):
    output = tmp_path / 'study'
    result = subprocess.run([sys.executable, '-m', 'experiments', 'editing',
                             'dry-run', '--output', str(output)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((output / 'summary.json').read_text())
    assert report['mode'] == 'synthetic_dry_run'
    assert report['paid_calls'] == 0
    assert report['attempted'] == 3
    assert report['statuses'] == ['frozen', 'invalid_patch', 'budget_stopped']
    assert report['research_results'] is False
```

- [ ] Run `python -m pytest tests/editing/test_cli.py -q`; expect the initial failure to be the unknown `editing` command.
- [ ] Add `"editing": "experiments.editing.cli"` to the existing command map. Implement these subcommands with argparse and the tested modules:

| Command | Required behavior |
|---|---|
| `prepare --config PATH --output DIR` | Validate and freeze a local study definition; reject an existing output directory. No provider call or implicit download. |
| `dry-run --output DIR` | Build the tiny local fixture and execute three scripted attempts: valid patch, invalid patch, exhausted budget. Produce the same artifact schema as a live run, visibly marked synthetic. |
| `run --study DIR` | Check the frozen manifest, prepared model, editor identity, remaining ledger, and permitted stage; execute independent attempts and freeze candidates before evaluation. A zero-spend manifest rejects paid dispatch. |
| `evaluate --study DIR` | Private evaluator reads only frozen candidates, scores the fixed partitions, and stores per-example outcomes. It supplies no feedback to the completed editor session. |
| `analyze --study DIR` | Compute paired effects, preserved capabilities, cost, and the prespecified prediction assessment. |
| `verify --study DIR` | Recompute hashes, check completeness and pairing, validate ledger accounting, replay saved patches, and detect altered predictions or metrics. No new editing call. |

- [ ] Add subprocess tests for each command's help and failure paths; output-directory collisions; tampered tensor/manifest/prediction files; interrupted attempts; and a complete dry-run -> evaluate -> analyze -> verify round trip. Offline verification must work with provider credentials absent.
- [ ] Document the exact commands, local data preparation, access boundaries, output meanings, and distinction between synthetic tests and research results in `docs/editing-protocol.md`. Link it from the README only when the implementation exists.
- [ ] Run `python -m pytest tests/editing -q`, then `python -m pytest tests/test_wsp.py -q` to check the existing pilot boundary. Expect the new suite and existing targeted tests to pass. Investigate new warnings; document pre-existing environment warnings rather than suppressing them globally.
- [ ] Commit with `git add experiments/editing/cli.py experiments/__main__.py tests/editing/test_cli.py docs/editing-protocol.md` and `git commit -m "Expose the transformer editing study workflow"`.

## 7. Execution readiness and study interpretation

The harness is ready for a pretrained-model experiment only after all offline acceptance tests pass and the run manifest contains real checkpoint revisions, actual data hashes, a concrete editor identity, hardware, per-run resource caps, a total spending cap, and evaluation thresholds. These are runtime inputs; an omitted field is a validation error, not permission to choose an expensive default. Preparing this plan does not execute that stage.

Before the first live cohort:

- [ ] Run `python -m experiments editing dry-run --output results/editing/offline-check` in the locked environment.
- [ ] Run `python -m experiments editing evaluate --study results/editing/offline-check`.
- [ ] Run `python -m experiments editing analyze --study results/editing/offline-check`.
- [ ] Run `python -m experiments editing verify --study results/editing/offline-check`.
- [ ] Measure one bounded forward-pass smoke test on the prepared target model, including peak memory and tokens/second. Include that resource use in the ledger. Verify fixed decoding and patch restoration before any editor search.
- [ ] Freeze the development cohort and expense estimate. Make every scheduled attempt visible, including provider errors, invalid patches, no-change choices, and budget stops.

Before a confirmation cohort:

- [ ] Specify which explanation the result could reject and which observation would distinguish it from an alternative.
- [ ] Freeze the predictor and feature definitions using development data only. If no defensible predictor has emerged, report an exploratory mechanism study; do not invent a general explanation to satisfy the title.
- [ ] Freeze replication and uncertainty analysis. Use model/task-specific headroom and raw outcome changes; do not interpret a ceiling-limited gain as intrinsic uneditability.
- [ ] Include at least one natural task and one untouched checkpoint/family before claiming transfer. Controlled repair evidence alone supports controlled repair claims.

The paper should explain the observed mechanism, its intervention evidence, its predictions, and its limits. Keep a small set of figures: the editing process and failure points; the intervention that distinguishes explanations; prediction versus observed outcome; and preservation/cost. The completed dynamics pilot becomes brief motivation or an appendix. No new experimental outcomes are written into the existing pilot manuscript before they exist.

## 8. Prior work and precise contribution boundary

| Primary source | What it already establishes or studies | Consequence for this plan |
|---|---|---|
| [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) | Direct editing demonstrations, including a transformer and a loop-related repair. | Transformer editing and recovery of obstructed behavior are not new claims. |
| [Hu et al., Exploring Weight-Space Programming](https://arxiv.org/html/2611.09382v1) | Outcomes across task families, model sizes, and editing trajectories; variable transfer and unsuccessful searches. | A bigger benchmark or size sweep is insufficient as the central contribution. |
| [Hase et al., Does Localization Inform Editing?](https://arxiv.org/abs/2301.04213) | Localization results need not identify the best weight-editing location in factual editing. | Test the relation between diagnosis and intervention instead of assuming it. |
| [Sutton et al., Stealth Edits to Large Language Models](https://arxiv.org/abs/2406.12670) | Theoretical and empirical links between representation geometry and editability for particular editing methods. | Predicting editability and representation-based explanations already have prior art; state the different access conditions and target behaviors. |
| [Barry, Can a Linear Program Do Weight-Space Programming?](https://nathan.rs/posts/weight-space-lp/) | A reported loop repair using a numerical solver, including results with the original editor's direction and a heuristic direction. | Distinguish discovering a change from constructing it. This is a single-model research report, not a general theorem. |
| [Salloum and Ali, Tree-Ising](https://arxiv.org/abs/2611.14210) | A gradient-free numerical repair procedure for loop-related failures, with an additional model test. | Gradient-free repair alone does not establish that LLM reasoning is the source of success. |

The candidate contribution is an account of which parts of the editing process cause useful behavioral changes, supported by interventions and tested prospectively across tasks and pretrained models. It remains a candidate until the experiments distinguish it from these existing explanations. Publication strength depends on the resulting evidence, not on the plan or the number of passing software tests.
