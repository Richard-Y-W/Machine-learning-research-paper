"""A reasoning model as the weight programmer (Opus 5.5), using a small library of weight-writing primitives.

The designer receives diagnostics about the model (which heads matter for succession, where heads attend in
lookup prompts, what each head of the layer costs when removed), the attention architecture, and a library of
primitives. The attention_program primitive documents the construction of lookup3.py; no completed hand-designed
programme or its measured results are supplied. Each round it returns up to three programmes (lists
of primitive calls); each is executed, scored on development data, and the measurements are returned. After
the last round the best programme by a rule fixed in advance is frozen and tested once on the confirmation
prompts used for the hand-designed edit.

Primitives (executed by this harness; no code is written by the designer):
  attention_program  write a head's query rows and its key group's rows from activation statistics:
                     whitened cross-covariance between the input at `query_role` and at `key_role`
                     (minus `contrast`), over lookup prompts with ORDINARY-WORD values; ridge, rank, RoPE band
  copy_qk            copy another head's query rows and key-group rows (same layer)
  copy_ov            copy another head's value-group rows and output-projection columns (same layer)
  scale_output       multiply a head's output-projection columns by a gain (0 removes the head)
Selection rule (fixed before the run): highest development successor rate among programmes whose WikiText loss
is at most 15% above the unedited model.

    EVO_MODEL=Qwen/Qwen3-0.6B-Base python -m experiments.compose.opus_program
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

DEV_DOM = "ordinals"; ROUNDS = int(os.environ.get("WSP_ROUNDS", "5")); N_CAND = 3; CAP = 4.00; LABEL = os.environ.get("WSP_LABEL", "program1_"); WIKI_LIMIT = 1.15
ROLES = ["final", "queried_value", "queried_arrow", "queried_key"]; CONTRASTS = ["other_positions", "other_values", "none"]

SYSTEM = """You are a weight-space programmer. You change a pretrained language model by writing weights directly, using a small library of primitives. There is no gradient training anywhere.

## Model
{model}: 28 decoder layers, residual width d = {d}, pre-norm blocks. Each self-attention layer has 16 query heads of 128 dims and 8 key/value groups (head h uses group h // 2, so heads 2g and 2g+1 share key rows and value rows). For input x (normalised residual stream): q = q_proj(x) per head, k = k_proj(x) per group, each passed through an RMS normalisation over its 128 dims (learned gain shared across the layer's heads), then rotary position embedding (dimension pairs (j, j+64), angle = position * 1e6^(-j/64): pairs with j >= 32 barely rotate, so scores placed there do not depend on distance). Attention logit = <q_t, k_s> / sqrt(128) with a causal mask. Head output coordinates z_h(t) = sum_s attn(t, s) * v_group(s), v = v_proj(x). The layer output is o_proj applied to the 16 heads' z (columns 128h..128h+127 belong to head h).

## Task
Prompts are key-value tables followed by a query, one pair per line, e.g. "oak -> h", "pine -> c", "elm -> q", then "pine ->". The unedited model answers with the looked-up value (" c"). Goal: the edited model answers with the SUCCESSOR of the looked-up value (" d"; the next weekday; "third" -> " fourth"), for value types and key words it has not been shown in this role. A global rewrite of lookup behaviour is acceptable. Damage to ordinary text prediction (WikiText loss) should stay moderate: programmes more than 15% above the unedited loss are not eligible.

## Primitives
A programme is a list of ops applied in order to the current weights.
- {{"op":"attention_program","layer":L,"head":h,"query_role":R1,"key_role":R2,"contrast":C,"ridge":lam,"rank":r,"rope_band":"low"|"all"}}: REPLACES head h's query rows and its key group's rows so that, from the position with role R1, head h attends to the position with role R2. Computed in closed form from activations of layer L's input over lookup prompts whose VALUES ARE ORDINARY WORDS: B = (C_qq + lam I)^-1 E[x_R1 (x_R2 - c)^T] (C_kk + lam I)^-1, truncated SVD of rank r, split into query and key rows. c is the contrast: "other_positions" (mean of the prompt's other positions), "other_values" (mean of the other lines' value tokens) or "none". "low" places the rows in the 64 slowly rotating dims (rank <= 64), "all" uses all 128 (rank <= 128, position dependent). lam is relative to the mean eigenvalue (useful range 1e-3 .. 10). Roles: "final" (the query position, after the last "->"), "queried_value" (the value token of the queried key's line), "queried_arrow" (the "->" token just before that value), "queried_key" (the key token of that line). Side effect: the other head of the same key group gets the new keys too.
- {{"op":"copy_qk","layer":L,"from_head":a,"to_head":b}}: head b gets head a's query rows; b's key group gets a's key group's rows.
- {{"op":"copy_ov","layer":L,"from_head":a,"to_head":b}}: b's value group gets a's value group's rows; b's output columns get a's.
- {{"op":"scale_output","layer":L,"head":h,"gain":g}}: multiply head h's output columns by g (g = 0 removes the head).
Any layer 0..27 may be used. At most 8 ops per programme.

## Diagnostics of the unedited model
{diagnostics}

## Protocol
{rounds} rounds. Each round you return up to 3 programmes; each is applied to the UNEDITED model, measured, and reported back: successor rate and lookup rate on ONE ordinal development value type, lookup accuracy on tables with ordinary-word values, sequence-continuation accuracy, WikiText loss, and the attention that each head you programmed puts on the queried value. After the last round, the eligible programme with the highest development successor rate is frozen and tested once on fresh prompts with other value types, unseen key words and unseen formats. Use the rounds to test hypotheses about the mechanism, not only to tune numbers.

Return JSON only: "analysis" (<= 150 words) and "candidates" (1 to 3 objects, each with "rationale" and "ops")."""

OP = {"type": "object", "properties": {
    "op": {"type": "string", "enum": ["attention_program", "copy_qk", "copy_ov", "scale_output"]}, "layer": {"type": "integer"}, "head": {"type": "integer"},
    "from_head": {"type": "integer"}, "to_head": {"type": "integer"}, "query_role": {"type": "string", "enum": ROLES}, "key_role": {"type": "string", "enum": ROLES},
    "contrast": {"type": "string", "enum": CONTRASTS}, "ridge": {"type": "number"}, "rank": {"type": "integer"}, "rope_band": {"type": "string", "enum": ["low", "all"]}, "gain": {"type": "number"}},
    "required": ["op", "layer"], "additionalProperties": False}
SCHEMA = {"type": "object", "properties": {"analysis": {"type": "string"}, "candidates": {"type": "array", "items": {
    "type": "object", "properties": {"rationale": {"type": "string"}, "ops": {"type": "array", "items": OP}}, "required": ["rationale", "ops"], "additionalProperties": False}}},
    "required": ["analysis", "candidates"], "additionalProperties": False}


def spent():
    led = json.loads(Path("results/evolution/ledger.json").read_text())
    return sum(e["charged_or_reserved_usd"] for e in led["entries"] if str(e.get("label", "")).startswith(LABEL))


class Bench:
    def __init__(self):
        self.lk = lk = LK(); self.tok = tok = lk.tok; self.P = lk.P
        try:
            lk.model.set_attn_implementation("eager")
        except Exception:
            lk.model.config._attn_implementation = "eager"
        self.L, self.H, self.kvdiv = lk.L, lk.H, lk.lab.kvdiv
        self.orig = {n: p.detach().clone() for n, p in self.P.items() if ".self_attn." in n and n.endswith("_proj.weight")}
        banned = {w.lower() for items in ORD.values() for w in items} | set(KEYS) | set(NEW_KEYS); words = []
        for i in range(tok.vocab_size):
            s = tok.decode([i])
            if s.startswith(" ") and s[1:].isalpha() and s[1:].islower() and 4 <= len(s) <= 9 and s[1:] not in banned:
                words.append(s[1:])
            if len(words) >= 3000:
                break
        self.ordv = {dm: [x for x in ORD[dm] if len(tok(" " + x, add_special_tokens=False)["input_ids"]) == 1] for dm in ORD}
        self.ordv["letters"] = list("bcdefghijklmnopqrstuvwx")
        rr = random.Random("opus-program-v1"); n_fit = int(os.environ.get("WSP_NFIT", "800"))
        self.fit = [lookup2.make(rr, words, None, "arrow") for _ in range(n_fit)]; self.wval = [lookup2.make(rr, words, None, "arrow") for _ in range(150)]
        self.dev = [lookup2.make(rr, self.ordv[DEV_DOM], DEV_DOM, "arrow") for _ in range(200)]
        self.wiki = json.loads(Path("results/evolution/splits.json").read_text())["wiki_dev"]["items"]
        self.seq = [(" ".join(x), y) for x, y in [(list("defg"), "h"), (list("klmn"), "o"), (list("pqrs"), "t"), (list("hijk"), "l"), (["Monday", "Tuesday", "Wednesday"], "Thursday"),
                                                  (["March", "April", "May"], "June"), (["one", "two", "three"], "four"), (["first", "second", "third"], "fourth"),
                                                  (["July", "August", "September"], "October"), (["five", "six", "seven"], "eight")]]
        self.cache = {}

    def name(self, layer, which):
        return f"model.layers.{layer}.self_attn.{which}_proj.weight"

    def positions(self, c):
        tk = lambda s: len(self.tok(s, add_special_tokens=False)["input_ids"]) - 1
        lines = c["text"].split("\n")[:-1]; vals = [tk("\n".join(lines[:i + 1])) for i in range(len(lines))]; pos = tk(c["prefix"])
        return dict(queried_value=pos, queried_arrow=tk(c["prefix"].rsplit(" ", 1)[0]), queried_key=tk(c["prefix"].rsplit(" -> ", 1)[0]), other_values=[v for v in vals if v != pos])

    @torch.inference_mode()
    def acts(self, layer):
        if layer not in self.cache:
            out = []; st = {}

            def pre(mod, args, kwargs):
                st["n"] = (args[0] if args else kwargs["hidden_states"])[0].float().clone()
            h = self.lk.layers[layer].self_attn.register_forward_pre_hook(pre, with_kwargs=True)
            try:
                for c in self.fit:
                    self.lk.model(input_ids=torch.tensor([self.tok(c["text"], add_special_tokens=False)["input_ids"]]), use_cache=False); out.append((st["n"], self.positions(c)))
            finally:
                h.remove()
            self.cache[layer] = out
        return self.cache[layer]

    def restore(self):
        with torch.no_grad():
            for n, w in self.orig.items():
                self.P[n].copy_(w)

    def apply(self, ops):
        assert isinstance(ops, list) and 1 <= len(ops) <= 8, "a programme has 1 to 8 ops"
        lk = self.lk; d = self.P[self.name(0, "q")].shape[1]
        for o in ops:
            L = int(o["layer"]); assert 0 <= L < self.L, "layer out of range"
            q, k, v, w = (self.P[self.name(L, x)] for x in "qkvo")
            def hd(key, o=o):
                h = int(o[key]); assert 0 <= h < self.H, f"{key} out of range"
                return h
            with torch.no_grad():
                if o["op"] == "scale_output":
                    h = hd("head"); w[:, lk.hsl(h)] *= float(o["gain"])
                elif o["op"] == "copy_qk":
                    a, b = hd("from_head"), hd("to_head"); q[lk.hsl(b)] = q[lk.hsl(a)].clone(); k[lk.lab.kvsl(b)] = k[lk.lab.kvsl(a)].clone()
                elif o["op"] == "copy_ov":
                    a, b = hd("from_head"), hd("to_head"); v[lk.lab.kvsl(b)] = v[lk.lab.kvsl(a)].clone(); w[:, lk.hsl(b)] = w[:, lk.hsl(a)].clone()
                elif o["op"] == "attention_program":
                    h = hd("head"); r1, r2 = o["query_role"], o["key_role"]; con = o.get("contrast", "other_positions"); lam = float(o.get("ridge", 1.0))
                    band = list(range(32, 64)) + list(range(96, 128)) if o.get("rope_band", "low") == "low" else list(range(128))
                    r = max(1, min(int(o.get("rank", len(band))), len(band)))
                    Cqq = torch.zeros(d, d, dtype=torch.float64); Ckk = torch.zeros(d, d, dtype=torch.float64); X = torch.zeros(d, d, dtype=torch.float64); nk = 0
                    idx = lambda n, p, role: n.shape[0] - 1 if role == "final" else p[role]
                    for n, p in self.acts(L):
                        n = n.double(); f = n[idx(n, p, r1)]; kp = idx(n, p, r2); kv = n[kp]
                        if con == "other_positions":
                            c = (n.sum(0) - kv) / max(1, n.shape[0] - 1)
                        elif con == "other_values":
                            c = n[torch.tensor(p["other_values"])].mean(0) if p["other_values"] else torch.zeros_like(kv)
                        else:
                            c = torch.zeros_like(kv)
                        Cqq += torch.outer(f, f); Ckk += n.T @ n; nk += n.shape[0]; X += torch.outer(f, kv - c)
                    N = len(self.acts(L)); Cqq /= N; Ckk /= nk; X /= N; I = torch.eye(d, dtype=torch.float64)
                    B = torch.linalg.solve(Cqq + lam * torch.trace(Cqq) / d * I, X) @ torch.linalg.inv(Ckk + lam * torch.trace(Ckk) / d * I)
                    U, S, Vh = torch.linalg.svd(B); bt = torch.tensor(band[:r])
                    qn = torch.zeros(HD, d, dtype=q.dtype); kn = torch.zeros(HD, d, dtype=k.dtype)
                    qn[bt] = ((U[:, :r] * S[:r].sqrt()).T).to(q.dtype); kn[bt] = (S[:r].sqrt()[:, None] * Vh[:r]).to(k.dtype)
                    q[lk.hsl(h)] = qn; k[lk.lab.kvsl(h)] = kn
                else:
                    raise AssertionError(f"unknown op {o['op']}")

    def rate(self, cs, key):
        top = self.lk.fwd([c["text"] for c in cs]).argmax(-1).tolist()
        return round(sum(t == self.lk.first(c[key]) for t, c in zip(top, cs)) / len(cs), 3)

    @torch.inference_mode()
    def attn(self, cs, layer):
        acc = torch.zeros(self.H)
        for c in cs:
            ids = torch.tensor([self.tok(c["text"], add_special_tokens=False)["input_ids"]])
            acc += self.lk.model(input_ids=ids, output_attentions=True, use_cache=False).attentions[layer][0, :, -1, self.positions(c)["queried_value"]].float()
        return [round(float(x), 2) for x in acc / len(cs)]

    def measure(self, ops=None):
        lg = self.lk.fwd([s for s, _ in self.seq]); seq = sum(int(t) == self.lk.first(a) for t, (_, a) in zip(lg.argmax(-1).tolist(), self.seq)) / len(self.seq)
        m = dict(dev_successor_rate=self.rate(self.dev, "s"), dev_lookup_rate=self.rate(self.dev, "v"), word_lookup_accuracy=self.rate(self.wval, "v"), sequence_continuation=round(seq, 2),
                 wikitext_loss=round(backend.perplexity_nll(self.lk.model, self.wiki), 3))
        if ops:
            m["attention_on_queried_value"] = {f"L{o['layer']}h{o['head']}": self.attn(self.wval[:30], int(o["layer"]))[int(o["head"])] for o in ops if o["op"] == "attention_program"}
        return m


def diagnostics(b):
    lk = b.lk; P = b.P; base = b.measure(); lines = [f"Unedited model on development data (ordinal development value type: {DEV_DOM}): {json.dumps(base)}"]
    fh = [OUT / "find_head.json", Path("results/compose_0p6b_check/find_head.json")]
    f = next((json.loads(p.read_text()) for p in fh if p.exists()), None)
    if f:
        lines.append("Sequence prompts (\"e f g h\" -> \" i\"), drop in next-item logit margin when a head's output at the final position is removed (largest first): "
                     + ", ".join(f"L{r['layer']}h{r['head']} {r['d_margin']:+.2f}" for r in f["scan"][:8]))
        lines.append("Oracle on sequence prompts: the head's output is replaced by its own value vector from the prompt ending one item later; fraction of two-steps-ahead answers: "
                     + ", ".join(f"L{r['layer']}h{r['head']} {r['two_step']:.2f}" for r in f["oracle"]))
    att = torch.zeros(lk.L, lk.H)
    for c in b.wval[:40]:
        ids = torch.tensor([b.tok(c["text"], add_special_tokens=False)["input_ids"]]); pos = b.positions(c)["queried_value"]
        with torch.inference_mode():
            a = lk.model(input_ids=ids, output_attentions=True, use_cache=False).attentions
        att += torch.stack([x[0, :, -1, pos].float() for x in a])
    att /= 40; flat = sorted(((float(att[l, h]), l, h) for l in range(lk.L) for h in range(lk.H)), reverse=True)[:12]
    lines.append("Lookup prompts, mean attention from the final position to the queried value token (largest): " + ", ".join(f"L{l}h{h} {a:.2f}" for a, l, h in flat))
    for layer in sorted({A[0]} | {l for _, l, _ in flat[:3]}):
        lines.append(f"  all heads of layer {layer}: " + " ".join(f"h{h}:{float(att[layer, h]):.2f}" for h in range(lk.H)))
    w = P[b.name(A[0], "o")]; cost = []
    for h in range(lk.H):
        with torch.no_grad():
            w[:, lk.hsl(h)] = 0
        cost.append(round(backend.perplexity_nll(lk.model, b.wiki) - base["wikitext_loss"], 3)); b.restore()
    lines.append(f"WikiText loss increase when each head of layer {A[0]} is removed entirely: " + " ".join(f"h{h}:{c:+.3f}" for h, c in enumerate(cost)))
    return base, "\n".join("- " + x for x in lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True); run_dir = OUT / "opus_program"; run_dir.mkdir(exist_ok=True); b = Bench(); t0 = time.time()
    if os.environ.get("WSP_DRY"):            # offline check of the primitives: the hand-designed edit expressed as a programme
        ops = [dict(op="attention_program", layer=A[0], head=A[1], query_role="final", key_role="queried_value", contrast="other_positions", ridge=1.0, rank=64, rope_band="low"),
               dict(op="scale_output", layer=A[0], head=A[1], gain=3.0)]
        print("[program] DRY unedited:", b.measure(), flush=True); b.apply(ops); print("[program] DRY hand-designed programme:", b.measure(ops), flush=True); b.restore()
        print("[program] DRY restored:", b.measure(), f"({time.time() - t0:.0f}s)", flush=True); return
    lp = run_dir / "log.json"; log = json.loads(lp.read_text()) if lp.exists() else dict(model=os.environ.get("EVO_MODEL"), rounds=[])
    if "base" not in log:
        log["base"], log["diagnostics"] = diagnostics(b); lp.write_text(json.dumps(log, indent=1), encoding="utf-8")
    base = log["base"]; print(f"[program] unedited: {base} ({time.time() - t0:.0f}s)", flush=True)
    system = SYSTEM.format(model=os.environ.get("EVO_MODEL", "Qwen/Qwen3-0.6B-Base").split("/")[-1], d=b.P[b.name(0, "q")].shape[1], diagnostics=log["diagnostics"], rounds=ROUNDS)
    (run_dir / "system_prompt.txt").write_text(system, encoding="utf-8")
    for rnd in range(len(log["rounds"]) + 1, ROUNDS + 1):
        if spent() + 0.8 > CAP:
            log["stop"] = "budget cap for this experiment"; break
        hist = []
        for g in log["rounds"]:
            hist.append(f"## Round {g['round']}\nYour analysis: {g['analysis']}")
            for c in g["candidates"]:
                hist.append(f"Programme {c['k']} ({c['rationale']}): {json.dumps(c['ops'])}\n  -> " + (f"ERROR {c['error']}" if c.get("error") else json.dumps(c["measured"])))
        user = ("\n".join(hist) if hist else "No programmes have been tried yet.") + f"\n\nRound {rnd} of {ROUNDS}. Return up to 3 programmes."
        (run_dir / f"prompt_r{rnd}.txt").write_text(user, encoding="utf-8")
        parsed, rec = editor.call(user, f"{LABEL}r{rnd}", run_dir / "calls", system=system, schema=SCHEMA)
        g = dict(round=rnd, cost_usd=rec["cost_usd"], stop_reason=rec.get("stop_reason"), analysis=(parsed or {}).get("analysis"), candidates=[])
        if not parsed:
            log["rounds"].append(g); log["stop"] = f"no usable reply (stop_reason {rec.get('stop_reason')}, error {rec.get('error')})"; lp.write_text(json.dumps(log, indent=1), encoding="utf-8")
            print(f"[program] round {rnd}: {log['stop']}", flush=True); break
        for k, c in enumerate(parsed["candidates"][:N_CAND]):
            row = dict(k=k, rationale=c.get("rationale", "")[:400], ops=c.get("ops"))
            try:
                b.apply(c["ops"]); row["measured"] = b.measure(c["ops"])
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"[:300]; row["trace"] = traceback.format_exc()[-400:]
            finally:
                b.restore()
            g["candidates"].append(row)
            print(f"[program] round {rnd} programme {k}: {row.get('error') or row['measured']}\n[program]    ops {json.dumps(row['ops'])[:500]}", flush=True)
        log["rounds"].append(g); lp.write_text(json.dumps(log, indent=1), encoding="utf-8")
        print(f"[program] round {rnd} analysis: {(g['analysis'] or '')[:500]} | ${rec['cost_usd']:.3f} | {time.time() - t0:.0f}s", flush=True)
    ok = [(c, g["round"]) for g in log["rounds"] for c in g["candidates"] if c.get("measured") and c["measured"]["wikitext_loss"] <= WIKI_LIMIT * base["wikitext_loss"]]
    if ok:
        best, rnd = max(ok, key=lambda t: t[0]["measured"]["dev_successor_rate"]); log["frozen"] = dict(round=rnd, k=best["k"], ops=best["ops"], measured=best["measured"], rationale=best["rationale"])
        print(f"[program] FROZEN (round {rnd}, programme {best['k']}): {json.dumps(best['ops'])}\n[program]   development: {best['measured']}", flush=True)
        b.apply(best["ops"]); conf = {}
        for label, keys, fmt in (("arrow_new_keys", NEW_KEYS, "arrow"), ("colon_new_keys", NEW_KEYS, "colon"), ("equals_new_keys", NEW_KEYS, "equals")):
            lookup2.KEYS = keys; r2 = random.Random("confirm-v1|" + label)                  # identical prompts to lookup3_confirm.py
            ev = {dm: [lookup2.make(r2, b.ordv[dm], dm, fmt) for _ in range(400)] for dm in ORD}; lookup2.KEYS = KEYS
            conf[label] = {dm: dict(successor=b.rate(ev[dm], "s"), lookup=b.rate(ev[dm], "v")) for dm in ORD}
            print(f"[program] confirm {label}: " + " ".join(f"{dm} {conf[label][dm]['successor']:.2f}" for dm in ORD), flush=True)
        conf["wikitext_loss"] = round(backend.perplexity_nll(b.lk.model, b.wiki), 3); conf["word_lookup_accuracy"] = b.rate(b.wval, "v"); b.restore(); log["confirm"] = conf
        print(f"[program] frozen programme: WikiText loss {conf['wikitext_loss']} (unedited {base['wikitext_loss']}), word lookup {conf['word_lookup_accuracy']}", flush=True)
    log["spent_usd"] = round(spent(), 3); lp.write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"[program] DONE; editor spend ${log['spent_usd']:.2f}; {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
