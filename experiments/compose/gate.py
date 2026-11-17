"""Composition gate.

Question: can successor components that Qwen3-0.6B-Base already has be chained, so that in a native
alphabet-continuation context the model advances by two letters? Forward passes and activation
patches only: no weight edits, no gradients, no API. Diagnostic interventions, not deployment results.

Steps
 0  native competence: next-letter accuracy (full vocabulary) and unpatched "+2" rate
 1  head scan: zero each head's output at the final position; effect on margin(next - current)
 2  forced self-attention on single heads (head output := its own value vector at the final position)
 3  routing only: forced self-attention on an early head A and a later head B; "+2" rate
 4  oracle interface patch: at layer l, replace the final position's normalised attention input with
    the one from the paired context that ends one letter later; "+2" rate, per layer and per head
 5  interchange: same patch with donors ending in an unrelated letter y; predicted output succ(y);
    controls: wrong-position patch, and the donor's own letter appearing instead of its successor

    EVO_MODEL=Qwen/Qwen3-0.6B-Base python -m experiments.compose.gate
"""
import json
import os
import random
import string
import time
from pathlib import Path

import torch

from ..evolution import backend

OUT = Path("results/compose"); HD = 128
AB = string.ascii_lowercase
LAYOUTS = {
    "alphabet": lambda w: "Alphabet: " + " ".join(AB) + ".\nContinue: " + " ".join(w[-2:]),
    "bare": lambda w: " ".join(w),
    "prose": lambda w: "Here are some letters in alphabetical order: " + " ".join(w),
    "seq": lambda w: "Sequence: " + " ".join(w[-3:]),
}
XS = list(AB[3:23])                                  # d..w : x-3 >= a and x+2 <= y
_r = random.Random("compose-gate-v1"); DEV = sorted(_r.sample(XS, 12)); RES = [x for x in XS if x not in DEV]
LOCATED = [(21, 6), (22, 8), (25, 8), (25, 9)]       # from the weights-only successor-head search


def window(x):
    i = AB.index(x)
    return list(AB[i - 3:i + 1])


class Lab:
    def __init__(self):
        backend.set_threads()
        self.model, self.tok = backend.load_model(os.environ.get("EVO_MODEL", "Qwen/Qwen3-0.6B-Base"))
        self.layers = self.model.model.layers; self.L = len(self.layers); self.H = self.model.config.num_attention_heads
        self.kvdiv = self.H // self.model.config.num_key_value_heads
        self.lt = torch.tensor([self.tid(" " + c) for c in AB])

    def tid(self, s):
        ids = self.tok(s, add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, (s, ids)
        return ids[0]

    def ids(self, layout, xs):
        enc = [self.tok(LAYOUTS[layout](window(x)), add_special_tokens=False)["input_ids"] for x in xs]
        assert len({len(e) for e in enc}) == 1, (layout, [len(e) for e in enc])
        return torch.tensor(enc)

    @torch.inference_mode()
    def run(self, ids, hooks=()):
        hs = [f() for f in hooks]
        try:
            return self.model(input_ids=ids, use_cache=False).logits[:, -1].float()
        finally:
            for h in hs:
                h.remove()

    # ---- hook factories (each returns a zero-argument function that registers and returns the handle)
    def grab_attn_in(self, layer, store):
        def pre(mod, args, kwargs):
            store[layer] = (args[0] if args else kwargs["hidden_states"])[:, -1].clone()
        return lambda: self.layers[layer].self_attn.register_forward_pre_hook(pre, with_kwargs=True)

    def patch_attn_in(self, layer, vec, pos=-1):
        def pre(mod, args, kwargs):
            h = (args[0] if args else kwargs["hidden_states"]).clone(); h[:, pos] = vec
            if args:
                return (h,) + tuple(args[1:]), kwargs
            kwargs = dict(kwargs); kwargs["hidden_states"] = h
            return args, kwargs
        return lambda: self.layers[layer].self_attn.register_forward_pre_hook(pre, with_kwargs=True)

    def grab_v(self, layer, store):
        def hook(mod, inp, out):
            store[layer] = out[:, -1].clone()
        return lambda: self.layers[layer].self_attn.v_proj.register_forward_hook(hook)

    def set_head(self, layer, head, mode, vec=None):
        """mode 'zero': head output at the final position := 0; 'self': := its own value vector there
        (forced self-attention); 'vec': := the given value vector (kv-head slice of another run)."""
        st = {}
        kv = head // self.kvdiv

        def vhook(mod, inp, out):
            st["v"] = out[:, -1, kv * HD:(kv + 1) * HD]

        def pre(mod, args):
            z = args[0].clone(); sl = slice(head * HD, (head + 1) * HD)
            z[:, -1, sl] = 0 if mode == "zero" else st["v"] if mode == "self" else vec
            return (z,)

        def reg():
            a = self.layers[layer].self_attn.v_proj.register_forward_hook(vhook)
            b = self.layers[layer].self_attn.o_proj.register_forward_pre_hook(pre)

            class Both:
                def remove(_):
                    a.remove(); b.remove()
            return Both()
        return reg


def stats(lab, lg, xs, shift_names=("x", "x+1", "x+2")):
    """Full-vocabulary top-1 rates for the current letter, its successor and its double successor."""
    top = lg.argmax(-1); out = {}
    for k, name in enumerate(shift_names):
        tgt = torch.tensor([lab.tid(" " + AB[AB.index(x) + k]) for x in xs])
        out[name] = float((top == tgt).float().mean())
    i = torch.arange(len(xs)); t = lambda k: torch.tensor([lab.tid(" " + AB[AB.index(x) + k]) for x in xs])
    out["margin_next_vs_cur"] = float((lg[i, t(1)] - lg[i, t(0)]).mean()); out["margin_plus2_vs_next"] = float((lg[i, t(2)] - lg[i, t(1)]).mean())
    return out


def pooled(lab, fn, xs):
    """Average a per-layout statistic over the four layouts."""
    rows = [fn(layout, xs) for layout in LAYOUTS]
    return {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}


def main():
    OUT.mkdir(parents=True, exist_ok=True); lab = Lab(); t0 = time.time()
    res = dict(dev=DEV, reserved=RES, layouts=list(LAYOUTS))
    nxt = lambda xs: [AB[AB.index(x) + 1] for x in xs]

    # ---- 0 native competence
    for name, xs in (("dev", DEV), ("reserved", RES)):
        res[f"native_{name}"] = pooled(lab, lambda lay, xs: stats(lab, lab.run(lab.ids(lay, xs)), xs), xs)
        res[f"native_{name}_by_layout"] = {lay: stats(lab, lab.run(lab.ids(lay, xs)), xs) for lay in LAYOUTS}
    print(f"[gate] 0 native dev {res['native_dev']}\n[gate] 0 native reserved {res['native_reserved']}", flush=True)
    for lay, s in res["native_dev_by_layout"].items():
        print(f"[gate] 0   {lay:9s} next {s['x+1']:.2f} +2 {s['x+2']:.2f}", flush=True)
    base_m = res["native_dev"]["margin_next_vs_cur"]

    # ---- 1 head scan (dev): zero-ablate each head at the final position
    scan = []
    for l in range(lab.L):
        for h in range(lab.H):
            s = pooled(lab, lambda lay, xs: stats(lab, lab.run(lab.ids(lay, xs), [lab.set_head(l, h, "zero")]), xs), DEV)
            scan.append(dict(layer=l, head=h, d_margin=s["margin_next_vs_cur"] - base_m, next=s["x+1"]))
    scan.sort(key=lambda r: r["d_margin"]); res["head_scan"] = scan
    print("[gate] 1 heads whose removal hurts next-letter margin most (d_margin, next-letter acc after removal):", flush=True)
    for r in scan[:10]:
        print(f"[gate] 1   L{r['layer']}h{r['head']}: {r['d_margin']:+.2f}  acc {r['next']:.2f}", flush=True)
    rank = {(r["layer"], r["head"]): i for i, r in enumerate(scan)}
    print("[gate] 1 weights-located heads: " + ", ".join(f"L{l}h{h} rank {rank[(l, h)] + 1}/{len(scan)} d_margin {scan[rank[(l, h)]]['d_margin']:+.2f}" for l, h in LOCATED), flush=True)
    causal = [(r["layer"], r["head"]) for r in scan[:8]]
    cand = list(dict.fromkeys(LOCATED + causal))

    # ---- 2 forced self-attention on single heads
    res["forced_self"] = {}
    for l, h in cand:
        s = pooled(lab, lambda lay, xs: stats(lab, lab.run(lab.ids(lay, xs), [lab.set_head(l, h, "self")]), xs), DEV)
        res["forced_self"][f"L{l}h{h}"] = s
        print(f"[gate] 2 forced self-attention L{l}h{h}: next {s['x+1']:.2f} +2 {s['x+2']:.2f} margin(next-cur) {s['margin_next_vs_cur']:+.2f} (base {base_m:+.2f}) "
              f"margin(+2-next) {s['margin_plus2_vs_next']:+.2f}", flush=True)

    # ---- 3 routing only: early head A and later head B both forced to self-attention
    res["routing_pairs"] = {}
    for la, ha in cand:
        for lb, hb in cand:
            if lb > la:
                s = pooled(lab, lambda lay, xs: stats(lab, lab.run(lab.ids(lay, xs), [lab.set_head(la, ha, "self"), lab.set_head(lb, hb, "self")]), xs), DEV)
                res["routing_pairs"][f"L{la}h{ha}->L{lb}h{hb}"] = s
    best = sorted(res["routing_pairs"].items(), key=lambda kv: -kv[1]["x+2"])[:5]
    for k, s in best:
        print(f"[gate] 3 routing {k}: +2 {s['x+2']:.2f} next {s['x+1']:.2f} margin(+2-next) {s['margin_plus2_vs_next']:+.2f}", flush=True)

    # ---- 4 oracle interface patch, per layer: attention input at the final position taken from the context ending in x+1
    def layer_patch(lay, xs, layer, donors, pos=-1):
        st = {}; lab.run(lab.ids(lay, donors), [lab.grab_attn_in(layer, st)])
        return lab.run(lab.ids(lay, xs), [lab.patch_attn_in(layer, st[layer], pos)])

    res["oracle_layer"] = {}
    for l in range(lab.L):
        res["oracle_layer"][l] = pooled(lab, lambda lay, xs: stats(lab, layer_patch(lay, xs, l, nxt(xs)), xs), DEV)
    print("[gate] 4 oracle patch of one layer's attention input (+2 rate by layer): " +
          " ".join(f"{l}:{res['oracle_layer'][l]['x+2']:.2f}" for l in range(lab.L)), flush=True)

    # cumulative: patch the attention input of every layer from l0 upward
    def multi_patch(lay, xs, l0, donors):
        st = {}; lab.run(lab.ids(lay, donors), [lab.grab_attn_in(l, st) for l in range(l0, lab.L)])
        return lab.run(lab.ids(lay, xs), [lab.patch_attn_in(l, st[l]) for l in range(l0, lab.L)])

    res["oracle_from_layer"] = {}
    for l0 in range(0, lab.L, 2):
        res["oracle_from_layer"][l0] = pooled(lab, lambda lay, xs: stats(lab, multi_patch(lay, xs, l0, nxt(xs)), xs), DEV)
    print("[gate] 4 oracle patch of attention inputs of all layers >= l (+2 rate): " +
          " ".join(f"{l}:{s['x+2']:.2f}" for l, s in res["oracle_from_layer"].items()), flush=True)

    # per head: the head's output at the final position := its value vector in the donor context
    def head_patch(lay, xs, heads, donors):
        st = {}; lab.run(lab.ids(lay, donors), [lab.grab_v(l, st) for l in {l for l, _ in heads}])
        hooks = [lab.set_head(l, h, "vec", st[l][:, (h // lab.kvdiv) * HD:(h // lab.kvdiv + 1) * HD]) for l, h in heads]
        return lab.run(lab.ids(lay, xs), hooks)

    res["oracle_head"] = {}
    for l, h in cand:
        s = pooled(lab, lambda lay, xs: stats(lab, head_patch(lay, xs, [(l, h)], nxt(xs)), xs), DEV)
        res["oracle_head"][f"L{l}h{h}"] = s
        print(f"[gate] 4 oracle value patch L{l}h{h}: +2 {s['x+2']:.2f} next {s['x+1']:.2f} margin(+2-next) {s['margin_plus2_vs_next']:+.2f}", flush=True)
    s = pooled(lab, lambda lay, xs: stats(lab, head_patch(lay, xs, cand, nxt(xs)), xs), DEV); res["oracle_head_all_candidates"] = s
    print(f"[gate] 4 oracle value patch, all {len(cand)} candidate heads together: +2 {s['x+2']:.2f} next {s['x+1']:.2f}", flush=True)

    # ---- 5 interchange at the best single layer and best from-layer: unrelated donors
    best_l = max(res["oracle_layer"], key=lambda l: res["oracle_layer"][l]["x+2"])
    best_from = max(res["oracle_from_layer"], key=lambda l: res["oracle_from_layer"][l]["x+2"])
    rr = random.Random("interchange")

    def interchange(xs, patch_fn):
        hit = cur = don = n = 0
        for lay in LAYOUTS:
            ys = [rr.choice([y for y in XS if abs(AB.index(y) - AB.index(x)) > 2]) for x in xs]
            top = patch_fn(lay, xs, ys).argmax(-1).tolist()
            for x, y, t in zip(xs, ys, top):
                n += 1; hit += t == lab.tid(" " + AB[AB.index(y) + 1]); don += t == lab.tid(" " + y); cur += t == lab.tid(" " + AB[AB.index(x) + 1])
        return dict(donor_successor=hit / n, donor_itself=don / n, original_next=cur / n)

    for name, xs in (("dev", DEV), ("reserved", RES)):
        res[f"interchange_layer{best_l}_{name}"] = interchange(xs, lambda lay, xs, ys: layer_patch(lay, xs, best_l, ys))
        res[f"interchange_from{best_from}_{name}"] = interchange(xs, lambda lay, xs, ys: multi_patch(lay, xs, best_from, ys))
        res[f"interchange_wrongpos_layer{best_l}_{name}"] = interchange(xs, lambda lay, xs, ys: layer_patch(lay, xs, best_l, ys, pos=-2))
        print(f"[gate] 5 interchange ({name}) single layer {best_l}: {res[f'interchange_layer{best_l}_{name}']}", flush=True)
        print(f"[gate] 5 interchange ({name}) layers >= {best_from}: {res[f'interchange_from{best_from}_{name}']}", flush=True)
        print(f"[gate] 5 control, wrong position ({name}) layer {best_l}: {res[f'interchange_wrongpos_layer{best_l}_{name}']}", flush=True)
    res["oracle_reserved_best_layer"] = pooled(lab, lambda lay, xs: stats(lab, layer_patch(lay, xs, best_l, nxt(xs)), xs), RES)
    res["oracle_reserved_best_from"] = pooled(lab, lambda lay, xs: stats(lab, multi_patch(lay, xs, best_from, nxt(xs)), xs), RES)
    print(f"[gate] 4 reserved letters: single layer {best_l} +2 {res['oracle_reserved_best_layer']['x+2']:.2f}; layers >= {best_from} +2 {res['oracle_reserved_best_from']['x+2']:.2f}", flush=True)

    # ---- verdict (engineering thresholds from the consultation)
    unp = res["native_dev"]["x+2"]; o1 = res["oracle_layer"][best_l]["x+2"]; o2 = res["oracle_from_layer"][best_from]["x+2"]
    res["verdict"] = dict(native_ok=res["native_dev"]["x+1"] >= 0.90, unpatched_plus2=unp, best_single_layer=best_l, single_layer_plus2=o1,
                          best_from_layer=best_from, from_layer_plus2=o2,
                          go_single_layer=res['native_dev']['x+1'] >= 0.90 and o1 >= 0.80 and o1 - unp >= 0.30,
                          reserved_single_layer_plus2=res['oracle_reserved_best_layer']['x+2'],
                          note='from-layer (cumulative) patches import donor computation and are diagnostic only')
    (OUT / "gate.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[gate] VERDICT {res['verdict']} ({time.time() - t0:.0f}s)", flush=True)
    print("[gate] DONE", flush=True)


if __name__ == "__main__":
    main()
