"""LLM mutation operator (Claude Opus 5.5) with a durable hard-cap cost ledger.

Before each call the worst-case cost is reserved; the call is refused if the
committed total plus the reservation would exceed the stop threshold. Actual
usage is settled afterwards. The ledger survives crashes (reservations of
crashed calls stay counted).
"""
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

MODEL = "claude-opus-5-5"
EFFORT = os.environ.get("EVO_EFFORT", "medium")
MAX_TOKENS = 12000
RATE_IN, RATE_OUT = 4 / 1e6, 20 / 1e6
RATE_CACHE_WRITE, RATE_CACHE_READ = 5 / 1e6, 0.20 / 1e6   # 5-minute cache write 1.25x; read $0.20/M
COST_SAFETY = 1.15   # 2026-11-14: console showed $11.84 spent vs ledger $10.52 (~12.5% under); pad future estimates
HARD_CAP = 45.08   # historical budget ceiling for the archived experiments
STOP_AT = 44.50
LEDGER = Path("results/evolution/ledger.json")


class BudgetExceeded(RuntimeError):
    pass


def _now():
    return datetime.now(timezone.utc).isoformat()


def _load(path):
    return json.loads(path.read_text()) if path.exists() else dict(hard_cap_usd=HARD_CAP, stop_at_usd=STOP_AT, entries=[])


def committed(path=LEDGER):
    return sum(e["charged_or_reserved_usd"] for e in _load(path)["entries"])


def reserve(label, reservation, path=LEDGER):
    path.parent.mkdir(parents=True, exist_ok=True)
    led = _load(path)
    total = sum(e["charged_or_reserved_usd"] for e in led["entries"])
    if total + reservation > STOP_AT:
        raise BudgetExceeded(f"committed ${total:.4f} + reservation ${reservation:.4f} > stop ${STOP_AT}")
    entry = dict(id=uuid.uuid4().hex, label=label, status="reserved", reserved_usd=reservation,
                 charged_or_reserved_usd=reservation, created_utc=_now())
    led["entries"].append(entry)
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(led, indent=1)); tmp.replace(path)
    return entry["id"]


def settle(entry_id, usage, status="completed", path=LEDGER):
    led = _load(path)
    for e in led["entries"]:
        if e["id"] == entry_id:
            e["status"] = status; e["usage"] = usage
            if usage is not None:
                e["charged_or_reserved_usd"] = cost(usage)
            e["settled_utc"] = _now()
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(led, indent=1)); tmp.replace(path)


def cost(u):
    return COST_SAFETY * (u.get("input_tokens", 0) * RATE_IN + u.get("output_tokens", 0) * RATE_OUT
            + u.get("cache_creation_input_tokens", 0) * RATE_CACHE_WRITE + u.get("cache_read_input_tokens", 0) * RATE_CACHE_READ)


def worst_case(n_input_tokens_est):
    return COST_SAFETY * (1.25 * n_input_tokens_est * RATE_CACHE_WRITE + MAX_TOKENS * RATE_OUT)   # 25% headroom on the input estimate


SYSTEM = """You are a weight-space programmer acting as the MUTATION OPERATOR in an evolutionary search over the weights of a pretrained language model (Qwen3-0.6B-Base, 28 decoder layers, hidden size 1024, 16 query heads and 8 key/value heads of dim 128, SwiGLU MLP with 3072 neurons, RMSNorm, tied input/output embeddings). No gradients or optimizers are used anywhere: you read diagnostics of the current parent model and directly write weight edits.

## The evolutionary loop
Each call you propose exactly 6 alternative MUTATIONS of the current parent. Each mutation is applied separately to the parent to make one offspring; every offspring is scored. The best offspring replaces the parent if it (a) raises TARGET fitness on the DEV items and (b) keeps PROTECTED fitness and WikiText NLL within the stated bounds of the ORIGINAL model. Accepted mutations are inherited by all descendants. You will see the lineage's history: earlier proposals and their measured effects. Final evaluation uses a large hidden TEST set from the same benchmarks, so prefer edits that plausibly improve the underlying capability over edits that fit individual dev items.

TARGET benchmarks: MMLU, ARC-Challenge, HellaSwag, CommonsenseQA. PROTECTED: ARC-Easy, OpenBookQA, PIQA, Winogrande, SciQ, plus WikiText language-modeling NLL. All tasks are zero-shot lettered multiple choice: the prompt ends with "Answer:" and the model is scored on the next-token distribution over " A", " B", ... Target fitness = mean over tasks of the mean log-probability of the correct letter (normalized over the valid letters). The study's final outcome is held-out ACCURACY; dev fitness is only a smooth proxy for it, so pure confidence rescaling (uniformly sharpening or flattening the letter distribution) is not a real improvement.

## Tensor names
L<i>.attn.q (2048x1024), L<i>.attn.k (1024x1024), L<i>.attn.v (1024x1024), L<i>.attn.o (1024x2048), L<i>.attn.qnorm (128), L<i>.attn.knorm (128), L<i>.mlp.gate (3072x1024), L<i>.mlp.up (3072x1024), L<i>.mlp.down (1024x3072), L<i>.ln1 (1024, pre-attention RMSNorm gain), L<i>.ln2 (1024, pre-MLP RMSNorm gain), final_norm (1024). i = 0..27. Matrices are (out, in): y = W x. The embedding matrix is not editable.

## Edit language (each mutation = 1 to 8 ops, applied in order)
- {"op":"scale","tensor":T,"axis":"row"|"col"|"all","index":[ints],"factor":f}   multiply rows/cols (or the whole tensor) by f in [0,3]. Use "head":h instead of axis/index for one attention head (rows of q/k/v, columns of o). For 1-D tensors give "index".
- {"op":"ablate","tensor":T,"head":h} or {"op":"ablate","tensor":T,"axis":...,"index":[...]}   same as scale with factor 0. MLP neuron j is column j of L<i>.mlp.down.
- {"op":"rank1","tensor":T,"u":REF,"v":REF,"alpha":a}   W += a * sigma1(W) * u v^T with u, v unit-normalized, 0<|a|<=0.5. u must have the tensor's OUT dimension, v its IN dimension.
- {"op":"set","tensor":T,"entries":[[row,col,value],...]}   write raw values ([index,value] for 1-D); at most 256 entries per mutation.
- {"op":"revert","tensor":T}   restore one tensor to the original checkpoint's values.
Vector REFs: err:L<i>, mean:L<i>, err:final (dim 1024); attnerr:L<i> (dim 2048); mlperr:L<i> (dim 3072); sv:<tensor>:<k>:left|right (k=0..7 singular vectors of the current tensor); unit:<i> (basis vector); tok:<text> and letter:<A-E> (embedding row, dim 1024).
Each mutation may change any touched tensor by at most 50% relative Frobenius norm. Invalid mutations score as failures.

Useful facts: the residual stream at layer i is x_i; attention adds o(heads), the MLP adds down(silu(gate x) * up x), each reading an RMSNorm of x (ln1, ln2) and the output logits read final_norm(x_28). Adding a rank-1 term u v^T to mlp.down writes direction u when the neuron pattern v fires; to attn.o it writes u when head outputs align with v; to ln gains it rescales features. Subtracting a projection onto an error direction from what a late component writes is a direct way to steer.

## How to work
Diagnose from the dossier, form hypotheses about WHY the model fails (e.g. letter-selection bias, components that push toward wrong letters, under-used knowledge), and make the 6 mutations DIVERSE: different hypotheses, layers, operations and magnitudes, including at least one conservative small edit and at least one bolder one. Learn from the history: do not repeat edits that failed; refine edits that helped; combine compatible successes. Keep the analysis concise.

Return JSON only, matching the schema: an "analysis" string (<= 120 words) and "mutations": exactly 6 objects, each with a short "rationale" and "ops"."""

_OP = {
    "type": "object",
    "properties": {
        "op": {"type": "string", "enum": ["scale", "ablate", "rank1", "set", "revert"]},
        "tensor": {"type": "string"},
        "axis": {"type": "string", "enum": ["row", "col", "all"]},
        "index": {"type": "array", "items": {"type": "integer"}},
        "head": {"type": "integer"},
        "factor": {"type": "number"},
        "u": {"type": "string"}, "v": {"type": "string"},
        "alpha": {"type": "number"},
        "entries": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
    },
    "required": ["op", "tensor"],
    "additionalProperties": False,
}
SCHEMA = {
    "type": "object",
    "properties": {
        "analysis": {"type": "string"},
        "mutations": {"type": "array", "items": {
            "type": "object",
            "properties": {"rationale": {"type": "string"}, "ops": {"type": "array", "items": _OP}},
            "required": ["rationale", "ops"], "additionalProperties": False}},
    },
    "required": ["analysis", "mutations"],
    "additionalProperties": False,
}


def call(user_text, label, archive_dir, system=SYSTEM, schema=None):
    """One mutation call. Returns (parsed_json_or_None, record). Never retries a completed call."""
    import anthropic
    archive_dir = Path(archive_dir); archive_dir.mkdir(parents=True, exist_ok=True)
    est_in = (len(system) + len(user_text)) // 3          # conservative chars->tokens
    rid = reserve(label, worst_case(est_in))
    client = anthropic.Anthropic(max_retries=3)
    t0 = time.time(); usage = None; text = None; err = None; stop = None
    try:
        with client.messages.stream(
            model=MODEL, max_tokens=MAX_TOKENS,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_text}],
            output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": schema or SCHEMA}},
        ) as stream:
            msg = stream.get_final_message()
        usage = msg.usage.model_dump() if hasattr(msg.usage, "model_dump") else dict(msg.usage)
        stop = msg.stop_reason
        text = next((b.text for b in msg.content if b.type == "text"), None)
        settle(rid, usage)
    except Exception as e:  # keep reservation counted on unknown failures
        err = f"{type(e).__name__}: {e}"
        settle(rid, None, status="error_reservation_kept")
    parsed = None
    if text:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            err = f"json: {e}"
    rec = dict(label=label, model=MODEL, effort=EFFORT, ledger_id=rid, seconds=round(time.time() - t0, 1),
               stop_reason=stop, usage=usage, cost_usd=cost(usage) if usage else None, error=err,
               user_text=user_text, response_text=text, utc=_now())
    (archive_dir / f"{label}.json").write_text(json.dumps(rec, indent=1))
    return parsed, rec
