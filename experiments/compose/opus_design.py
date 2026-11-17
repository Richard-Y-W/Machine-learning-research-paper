"""Can a reasoning model design the weight edit itself?  (Opus 5.5 as designer; a few API calls.)

Setting: Qwen3-1.7B-Base, lookup -> successor. The designer is given the diagnostics that were available before
the human/assistant-designed construction (lookup3.py) was written, the architecture of the attention layer, and
activations from lookups whose values are ordinary words. It is NOT given that construction. It must write Python
that computes replacement weight matrices for the layer-21 self-attention (any of q/k/v/o projections), using
linear algebra only. Its code is executed, the result is scored on development data (ordinary-word lookups,
WikiText loss, and ONE ordinal development domain), and it may revise twice. The best attempt by the development
successor rate is frozen and then evaluated on the same fresh confirmation prompts as the frozen procedure.
"""
import json
import os
import random
import time
import traceback
from pathlib import Path

import torch

from ..evolution import backend, editor
from .bridge import A, HD, OUT
from . import lookup2
from .lookup import KEYS, LK, ORD
from .lookup3_confirm import NEW_KEYS

DEV_DOM = "ordinals"; MAX_ATTEMPTS = 3; CAP = 2.00; LABEL = "design1_"

SYSTEM = """You are a weight-space programmer. You modify a pretrained language model by computing new weight matrices directly. No gradients, optimizers or training loops are allowed; linear algebra (torch.linalg) and statistics of activations are.

## Model and layer
Qwen3-1.7B-Base: 28 decoder layers, residual width d = 2048, pre-norm (RMSNorm) blocks. You may replace matrices of ONE module: the self-attention of layer 21. That module computes, for an input x (the RMS-normalised residual stream, shape T x d):
- q = q_proj(x), shape T x 2048, split into 16 heads of 128 dims; k = k_proj(x), shape T x 1024, split into 8 key/value groups of 128 dims; v = v_proj(x), T x 1024, 8 groups. Head h (0..15) uses key/value group h // 2. No biases.
- Each head's 128-dim query and each group's 128-dim key pass through an RMSNorm over those 128 dims (q_norm, k_norm: y = g * y / rms(y), one learned gain vector g per layer, shared by all heads; provided to you).
- Rotary position embedding is then applied to queries and keys: dims (j, j + 64) for j = 0..63 form a pair rotated by angle pos * theta^(-j/64), theta = 1e6. So pairs with large j rotate very slowly with position.
- Attention logit(t, s) = <q_t, k_s> / sqrt(128), causal mask, softmax over s. Head output coordinates z_h(t) = sum_s attn(t, s) v_group(s). The module output is o_proj applied to the concatenation of the 16 heads' z (o_proj: 2048 x 2048; columns 128h .. 128h+127 belong to head h).

## Task
Prompts are key-value tables followed by a query, e.g. "oak -> h\\npine -> c\\nelm -> q\\npine ->". The unedited model answers with the looked-up value (" c"). Goal: after your edit the model should answer with the SUCCESSOR of the looked-up value (" d"; for weekdays the next day; for "third", " fourth"; and so on), for value types and key words it has never been shown in this role. The edit is permanent and unconditional (a global rewrite of lookup behaviour is acceptable), but damage to ordinary text prediction should be kept moderate.

## What is known (diagnostics)
- Head 6 of layer 21 is a successor head. On plain sequence prompts ("e f g h" -> " i") removing it lowers the next-item margin more than any other head, and replacing its output z_6 at the final position by its own value vector taken from a prompt ending one item later makes the model answer two steps ahead in 97% of cases. So z_6 := (value vector of a position holding item X) makes the rest of the network produce the successor of X.
- On lookup prompts head 6 natively puts 0.04 of its attention on the queried value token. Heads elsewhere that retrieve the value attend to it diffusely (at most about 0.4).
- In the smaller sibling model (same family, same head index), forcing head 6 at the final position to attend ONLY to the queried value token gave successor answers on 12-56% of lookups; additionally multiplying head 6's output by 3 gave 62-79%. The copy pathway (other components that write the looked-up value itself) competes with head 6's vote.
- Head 6 shares its key/value group (group 3) with head 7.

## Your interface
Write Python source defining `build(data)`. `data` is a dict:
- "examples": list of 1,000 tuples (x, pos). x is a float64 tensor T x 2048: the layer-21 attention input at every position of a lookup prompt whose VALUES ARE ORDINARY WORDS (no ordinal items). The final position is index T - 1 (the query, after "->"). pos is the index of the queried value token (the token that the lookup should retrieve).
- "weights": dict with float64 clones of "q_proj" (2048 x 2048), "k_proj" (1024 x 2048), "v_proj" (1024 x 2048), "o_proj" (2048 x 2048), "q_norm" (128), "k_norm" (128).
- "torch": the torch module (already imported for you as `torch` too). Tensors are on the GPU; use data["examples"][0][0].device.
`build` returns a dict mapping any subset of {"q_proj", "k_proj", "v_proj", "o_proj"} to a full replacement matrix of the same shape. Matrices you do not return are left unchanged. Must run in under two minutes. No file or network access, no imports other than math.

## Feedback
After each attempt you receive measurements on development data: head attention from the final position to the queried value on held-out ordinary-word lookups, ordinary-word lookup accuracy, WikiText loss, and the successor rate on ONE ordinal development domain. You get up to three attempts in total; the best one by development successor rate is frozen and tested once on fresh prompts with other value types, unseen key words and unseen formats.

Return JSON only: "analysis" (<= 150 words: your mechanism and why it should work), "code" (the Python source)."""

SCHEMA = {"type": "object", "properties": {"analysis": {"type": "string"}, "code": {"type": "string"}}, "required": ["analysis", "code"], "additionalProperties": False}


def spent():
    led = json.loads(Path("results/evolution/ledger.json").read_text())
    return sum(e["charged_or_reserved_usd"] for e in led["entries"] if str(e.get("label", "")).startswith(LABEL))


def main():
    OUT.mkdir(parents=True, exist_ok=True); run_dir = OUT / "opus_design"; run_dir.mkdir(exist_ok=True)
    lk = LK(); tok = lk.tok; P = lk.P; res = dict(model=os.environ.get("EVO_MODEL"), attempts=[])
    try:
        lk.model.set_attn_implementation("eager")
    except Exception:
        lk.model.config._attn_implementation = "eager"
    pre = f"model.layers.{A[0]}.self_attn."; names = {k: pre + k + ".weight" for k in ("q_proj", "k_proj", "v_proj", "o_proj")}
    orig = {k: P[n].detach().clone() for k, n in names.items()}
    banned = {w.lower() for items in ORD.values() for w in items} | set(KEYS) | set(NEW_KEYS); words = []
    for i in range(tok.vocab_size):
        s = tok.decode([i])
        if s.startswith(" ") and s[1:].isalpha() and s[1:].islower() and 4 <= len(s) <= 9 and s[1:] not in banned:
            words.append(s[1:])
        if len(words) >= 3000:
            break
    ordv = {dm: [x for x in ORD[dm] if len(tok(" " + x, add_special_tokens=False)["input_ids"]) == 1] for dm in ORD}
    ordv["letters"] = list("bcdefghijklmnopqrstuvwx")
    rr = random.Random("opus-design-v1")
    fit = [lookup2.make(rr, words, None, "arrow") for _ in range(1000)]; wval = [lookup2.make(rr, words, None, "arrow") for _ in range(200)]
    dev = [lookup2.make(rr, ordv[DEV_DOM], DEV_DOM, "arrow") for _ in range(300)]
    wiki = json.loads(Path("results/evolution/splits.json").read_text())["wiki_dev"]["items"]
    T = lambda cs: [c["text"] for c in cs]

    @torch.inference_mode()
    def acts(c):
        ids = torch.tensor([tok(c["text"], add_special_tokens=False)["input_ids"]]); pos = len(tok(c["prefix"], add_special_tokens=False)["input_ids"]) - 1; st = {}

        def prehook(mod, args, kwargs):
            st["n"] = (args[0] if args else kwargs["hidden_states"])[0].double().clone()
        h = lk.layers[A[0]].self_attn.register_forward_pre_hook(prehook, with_kwargs=True)
        try:
            lk.model(input_ids=ids, use_cache=False)
        finally:
            h.remove()
        return st["n"], pos

    att_mod = lk.layers[A[0]].self_attn
    data = dict(examples=[acts(c) for c in fit], torch=torch,
                weights={**{k: orig[k].double().clone() for k in orig}, "q_norm": att_mod.q_norm.weight.detach().double().clone(), "k_norm": att_mod.k_norm.weight.detach().double().clone()})

    def restore():
        with torch.no_grad():
            for k, n in names.items():
                P[n].copy_(orig[k])

    def install(new):
        with torch.no_grad():
            for k, w in new.items():
                P[names[k]].copy_(w.to(P[names[k]].dtype))

    def rate(cs, key):
        top = lk.fwd(T(cs)).argmax(-1).tolist()
        return round(sum(t == lk.first(c[key]) for t, c in zip(top, cs)) / len(cs), 3)

    @torch.inference_mode()
    def attn_all(cs):
        acc = torch.zeros(lk.H)
        for c in cs:
            ids = torch.tensor([tok(c["text"], add_special_tokens=False)["input_ids"]]); pos = len(tok(c["prefix"], add_special_tokens=False)["input_ids"]) - 1
            acc += lk.model(input_ids=ids, output_attentions=True, use_cache=False).attentions[A[0]][0, :, -1, pos].float()
        return [round(float(x), 2) for x in acc / len(cs)]

    def feedback():
        return dict(attention_to_queried_value_by_head=attn_all(wval[:60]), word_lookup_accuracy=rate(wval, "v"), wikitext_loss=round(backend.perplexity_nll(lk.model, wiki), 3),
                    dev_successor_rate=rate(dev, "s"), dev_lookup_rate=rate(dev, "v"))

    base = feedback(); res["base"] = base
    print(f"[design] unedited model: {base}", flush=True)
    history = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if spent() + 0.6 > CAP:
            print("[design] budget cap for this experiment reached", flush=True); break
        user = (f"Unedited model on development data ({DEV_DOM} is the ordinal development domain): {json.dumps(base)}\n\n"
                + (history or "This is your first attempt.") + f"\n\nAttempt {attempt} of {MAX_ATTEMPTS}. Return JSON with \"analysis\" and \"code\".")
        (run_dir / f"prompt_{attempt}.txt").write_text(user, encoding="utf-8")
        parsed, rec = editor.call(user, f"{LABEL}a{attempt}", run_dir / "calls", system=SYSTEM, schema=SCHEMA)
        row = dict(attempt=attempt, cost_usd=rec["cost_usd"], error=rec["error"], analysis=(parsed or {}).get("analysis"))
        code = (parsed or {}).get("code") or ""; (run_dir / f"code_{attempt}.py").write_text(code, encoding="utf-8")
        fb = None
        try:
            ns = {"torch": torch, "math": __import__("math")}; exec(code, ns); t0 = time.time()
            new = ns["build"](data); row["build_seconds"] = round(time.time() - t0, 1)
            assert isinstance(new, dict) and new and set(new) <= set(names), f"bad keys {list(new) if isinstance(new, dict) else type(new)}"
            for k, w in new.items():
                assert tuple(w.shape) == tuple(orig[k].shape) and bool(torch.isfinite(w).all()), f"{k}: bad shape or non-finite values"
            row["changed"] = {k: round(float((w.to(orig[k].dtype) - orig[k]).norm() / orig[k].norm()), 3) for k, w in new.items()}
            install(new); fb = feedback(); restore()
            torch.save({k: w.float().cpu() for k, w in new.items()}, run_dir / f"weights_{attempt}.pt")
        except Exception as e:
            restore(); row["exec_error"] = f"{type(e).__name__}: {e}"; row["trace"] = traceback.format_exc()[-600:]
        row["feedback"] = fb; res["attempts"].append(row)
        print(f"[design] attempt {attempt}: ${rec['cost_usd']:.3f} | {row.get('exec_error') or fb}", flush=True)
        print(f"[design]   analysis: {(row['analysis'] or '')[:400]}", flush=True)
        history += (f"\n## Your attempt {attempt}\nAnalysis: {row['analysis']}\nCode:\n```python\n{code}\n```\n"
                    + (f"Execution error: {row['exec_error']}\n{row.get('trace', '')}" if fb is None else f"Measured after installing it: {json.dumps(fb)}\nRelative change of each matrix you returned: {row['changed']}") + "\n")
        (run_dir / "result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    ok = [r for r in res["attempts"] if r["feedback"]]
    if ok:
        best = max(ok, key=lambda r: r["feedback"]["dev_successor_rate"]); res["frozen_attempt"] = best["attempt"]
        print(f"[design] FROZEN: attempt {best['attempt']} (development successor rate {best['feedback']['dev_successor_rate']:.2f}; unedited {base['dev_successor_rate']:.2f})", flush=True)
        new = torch.load(run_dir / f"weights_{best['attempt']}.pt"); res["confirm"] = {}
        for label, keys, fmt in (("arrow_new_keys", NEW_KEYS, "arrow"), ("colon_new_keys", NEW_KEYS, "colon"), ("equals_new_keys", NEW_KEYS, "equals")):
            lookup2.KEYS = keys; r2 = random.Random("confirm-v1|" + label)                 # the same prompts as lookup3_confirm.py
            ev = {dm: [lookup2.make(r2, ordv[dm], dm, fmt) for _ in range(400)] for dm in ORD}; lookup2.KEYS = KEYS
            nat = {dm: rate(ev[dm], "s") for dm in ORD}; install(new); ed = {dm: rate(ev[dm], "s") for dm in ORD}; restore()
            res["confirm"][label] = dict(native=nat, edited=ed)
            print(f"[design] confirm {label}: " + " ".join(f"{dm} {nat[dm]:.2f}->{ed[dm]:.2f}" for dm in ORD), flush=True)
        install(new); res["confirm_wikitext_loss"] = round(backend.perplexity_nll(lk.model, wiki), 3); restore()
        print(f"[design] WikiText loss with the frozen edit: {res['confirm_wikitext_loss']} (unedited {base['wikitext_loss']})", flush=True)
    res["spent_usd"] = round(spent(), 3)
    (run_dir / "result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[design] DONE; editor spend for this experiment ${res['spent_usd']:.2f}", flush=True)


if __name__ == "__main__":
    main()
