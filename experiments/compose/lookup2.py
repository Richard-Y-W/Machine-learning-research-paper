"""Lookup -> successor, second design: retrieval's WHERE joined to the successor head's WHAT.

lookup.py showed: (1) handing the successor head A = L21h6 the retrieved item's representation makes the
model output its successor on 0.61-0.67 of day/month/ordinal lookups (native 0.07-0.20), so A works outside
sequence contexts; (2) rewriting retrieval heads' output projections fails (no better than shuffled control),
and retrieval is not concentrated in heads found by final-position ablation.

Here the join is made on the attention side. A's OV circuit is left untouched; A is made to LOOK where a
retrieval head looks, so it reads the queried value's own token position (the format A reads natively).
  a  attention scan: which heads attend from the final position to the queried value?
  b  oracle: A's output := sum_s attn_H(s) * A's value vector at s   (H's attention pattern, A's values)
  c  weight edit, exact and fit-free, for retrieval heads in A's own layer: copy H's query rows into A's
     query rows and H's key rows into A's key group. Optional gain on A's output projection.
     Controls: same transplant from a non-retrieval head of the layer; gain alone without the transplant.
"""
import json
import random
import time
from pathlib import Path

import torch

from ..evolution import backend
from .bridge import A, HD, OUT
from .lookup import KEYS, LK, ORD, succ, score

FORMATS = {"colon": ("{k}: {v}\n", "{k}:"), "equals": ("{k} = {v}\n", "{k} ="), "arrow": ("{k} -> {v}\n", "{k} ->")}


def make(rr, pool, dom, fmt, n_pairs=None):
    line, query = FORMATS[fmt]
    for _ in range(300):
        n = n_pairs or rr.choice([3, 4, 5]); keys = rr.sample(KEYS, n); vals = rr.sample(pool, n); q = rr.randrange(n)
        s = None
        if dom:
            idx = [ORD[dom].index(v) for v in vals]; s = succ(dom, vals[q])
            if s is None or s in vals or any(abs(a - b) == 1 for i, a in enumerate(idx) for b in idx[i + 1:]):
                continue
        pre = "".join(line.format(k=k, v=v) for k, v in zip(keys[:q + 1], vals[:q + 1])).rstrip("\n")
        text = "".join(line.format(k=k, v=v) for k, v in zip(keys, vals)) + query.format(k=keys[q])
        return dict(text=text, prefix=pre, v=vals[q], s=s)
    raise RuntimeError("could not build a prompt")


def main():
    lk = LK(); tok = lk.tok; t0 = time.time(); res = {}
    try:
        lk.model.set_attn_implementation("eager")
    except Exception:
        lk.model.config._attn_implementation = "eager"
    banned = {w.lower() for items in ORD.values() for w in items} | set(KEYS); words = []
    for i in range(tok.vocab_size):
        s = tok.decode([i])
        if s.startswith(" ") and s[1:].isalpha() and s[1:].islower() and 4 <= len(s) <= 9 and s[1:] not in banned:
            words.append(s[1:])
        if len(words) >= 3000:
            break
    ordv = {d: [x for x in ORD[d] if len(tok(" " + x, add_special_tokens=False)["input_ids"]) == 1] for d in ORD}
    ordv["letters"] = list("bcdefghijklmnopqrstuvwx")
    rr = random.Random("lookup-v2")
    sc = lambda cases, lg: score(lk, lg, [(c["text"], c["v"], c["s"]) for c in cases])
    T = lambda cases: [c["text"] for c in cases]

    # ---- format selection on ordinary-word values only (no ordinal items)
    res["format_native_word_lookup"] = {}
    for f in FORMATS:
        cs = [make(rr, words, None, f) for _ in range(200)]
        res["format_native_word_lookup"][f] = sc(cs, lk.fwd(T(cs)))["lookup"]
    fmt = max(res["format_native_word_lookup"], key=res["format_native_word_lookup"].get); res["format"] = fmt
    print(f"[lookup2] native lookup with ordinary-word values by format: {res['format_native_word_lookup']} -> using '{fmt}'", flush=True)
    wcases = [make(rr, words, None, fmt) for _ in range(120)]
    ev = {d: [make(rr, ordv[d], d, fmt) for _ in range(200)] for d in ORD}
    res["native"] = {d: sc(ev[d], lk.fwd(T(ev[d]))) for d in ORD}
    for d in ORD:
        print(f"[lookup2] native {d:9s}: lookup {res['native'][d]['lookup']:.2f} successor {res['native'][d]['successor']:.2f}", flush=True)

    # ---- a attention scan (ordinary-word lookups): attention from the final position to the queried value token
    @torch.inference_mode()
    def attn_and_values(c):
        ids = torch.tensor([tok(c["text"], add_special_tokens=False)["input_ids"]]); pos = len(tok(c["prefix"], add_special_tokens=False)["input_ids"]) - 1
        st = {}
        h = lk.layers[A[0]].self_attn.v_proj.register_forward_hook(lambda m, i, o: st.__setitem__("v", o[0, :, lk.lab.kvsl(A[1])].clone()))
        try:
            out = lk.model(input_ids=ids, output_attentions=True, use_cache=False)
        finally:
            h.remove()
        att = torch.stack([a[0, :, -1, :] for a in out.attentions])            # layers x heads x T
        return att.float(), st["v"].float(), pos, int(out.logits[0, -1].argmax())

    acc = torch.zeros(lk.L, lk.H)
    for c in wcases:
        att, _, pos, _ = attn_and_values(c); acc += att[:, :, pos]
    acc /= len(wcases)
    flat = sorted(((float(acc[l, h]), l, h) for l in range(lk.L) for h in range(lk.H)), reverse=True)
    res["attention_to_queried_value"] = [dict(layer=l, head=h, attn=round(a, 3)) for a, l, h in flat[:20]]
    print("[lookup2] a heads attending from the final position to the queried value (mean attention): " +
          ", ".join(f"L{l}h{h} {a:.2f}" for a, l, h in flat[:12]), flush=True)
    same = [(a, l, h) for a, l, h in flat if l == A[0] and h != A[1]]
    print(f"[lookup2] a in A's own layer {A[0]}: " + ", ".join(f"h{h} {a:.2f}" for a, l, h in same[:6]) + f"; A itself (h{A[1]}) {float(acc[A[0], A[1]]):.2f}", flush=True)
    res["same_layer_attention"] = [dict(head=h, attn=round(a, 3)) for a, l, h in same]

    # ---- b oracle: H's attention pattern with A's values
    cand = [(l, h) for a, l, h in flat[:6]] + [(l, h) for a, l, h in same[:2]]; cand = list(dict.fromkeys(cand))
    res["oracle_attention_join"] = {}
    for (l, h) in cand:
        row = {}
        for d in ORD:
            cases = ev[d][:100]; zs = []
            for c in cases:
                att, v, pos, _ = attn_and_values(c); zs.append(att[l, h] @ v)
            lg = lk.fwd(T(cases), [lk.set_z(A[0], A[1], torch.stack(zs))]); row[d] = sc(cases, lg)
        res["oracle_attention_join"][f"L{l}h{h}"] = row
        print(f"[lookup2] b oracle, A looks where L{l}h{h} looks: " + " ".join(f"{d} succ {row[d]['successor']:.2f}/look {row[d]['lookup']:.2f}" for d in ORD), flush=True)
    # ideal routing: A attends only to the queried value token
    row = {}
    for d in ORD:
        cases = ev[d][:100]; zs = []
        for c in cases:
            _, v, pos, _ = attn_and_values(c); zs.append(v[pos])
        row[d] = sc(cases, lk.fwd(T(cases), [lk.set_z(A[0], A[1], torch.stack(zs))]))
        for g in (2.0, 3.0):
            row[f"{d}_gain{g}"] = sc(cases, lk.fwd(T(cases), [lk.set_z(A[0], A[1], g * torch.stack(zs))]))
    res["oracle_ideal_routing"] = row
    print("[lookup2] b oracle, A attends only to the queried value: " + " ".join(f"{d} succ {row[d]['successor']:.2f}/look {row[d]['lookup']:.2f}" for d in ORD), flush=True)
    for g in (2.0, 3.0):
        print(f"[lookup2] b   same with A's output x{g}: " + " ".join(f"{d} succ {row[f'{d}_gain{g}']['successor']:.2f}/look {row[f'{d}_gain{g}']['lookup']:.2f}" for d in ORD), flush=True)

    # ---- c weight edit: copy a same-layer retrieval head's query rows and key group into A's (exact, no fitting)
    P = lk.P; pre = f"model.layers.{A[0]}.self_attn."; kvdiv = lk.lab.kvdiv
    names = [pre + "q_proj.weight", pre + "k_proj.weight", pre + "o_proj.weight"] + [n for n in (pre + "q_proj.bias", pre + "k_proj.bias") if n in P]
    orig = {n: P[n].detach().clone() for n in names}
    wiki = json.loads(Path("results/evolution/splits.json").read_text())["wiki_dev"]["items"]; nll0 = backend.perplexity_nll(lk.model, wiki)
    seq = [(" ".join(x), y, None) for x, y in [((list("defg")), "h"), (list("klmn"), "o"), (list("pqrs"), "t"), (["Monday", "Tuesday", "Wednesday"], "Thursday"),
                                               (["March", "April", "May"], "June"), (["one", "two", "three"], "four"), (["first", "second", "third"], "fourth")]]

    def transplant(src, gain):
        with torch.no_grad():
            if src is not None:
                P[pre + "q_proj.weight"][lk.hsl(A[1])] = orig[pre + "q_proj.weight"][lk.hsl(src)]
                ka, ks = lk.lab.kvsl(A[1]), lk.lab.kvsl(src)
                P[pre + "k_proj.weight"][ka] = orig[pre + "k_proj.weight"][ks]
                if pre + "q_proj.bias" in P:
                    P[pre + "q_proj.bias"][lk.hsl(A[1])] = orig[pre + "q_proj.bias"][lk.hsl(src)]; P[pre + "k_proj.bias"][ka] = orig[pre + "k_proj.bias"][ks]
            P[pre + "o_proj.weight"][:, lk.hsl(A[1])] = gain * orig[pre + "o_proj.weight"][:, lk.hsl(A[1])]

    def restore():
        with torch.no_grad():
            for n, w in orig.items():
                P[n].copy_(w)

    def measure():
        out = {d: sc(ev[d], lk.fwd(T(ev[d]))) for d in ORD}
        out["word_lookup"] = sc(wcases, lk.fwd(T(wcases)))["lookup"]; out["wiki_nll"] = backend.perplexity_nll(lk.model, wiki)
        lg = lk.fwd([s[0] for s in seq]); out["sequence_next"] = sum(int(t) == lk.first(s[1]) for t, s in zip(lg.argmax(-1).tolist(), seq)) / len(seq)
        return out

    res["edit"] = {}; res["wiki_nll_base"] = nll0
    srcs = [h for a, l, h in same if h // kvdiv != A[1] // kvdiv]            # a source in A's own key group would be a no-op on keys
    plan = [("retrieval:h%d" % h, h) for h in srcs[:3]] + [("control_nonretrieval:h%d" % srcs[-1], srcs[-1]), ("control_gain_only", None)]
    for label, src in plan:
        for gain in (1.0, 2.0, 3.0):
            transplant(src, gain); m = measure(); restore(); res["edit"][f"{label}_gain{gain}"] = m
            print(f"[lookup2] c EDIT {label} gain {gain}: " + " ".join(f"{d} succ {m[d]['successor']:.2f}/look {m[d]['lookup']:.2f}" for d in ORD)
                  + f" | word lookup {m['word_lookup']:.2f} | sequences {m['sequence_next']:.2f} | WikiText loss {m['wiki_nll']:.3f} (base {nll0:.3f})", flush=True)
    (OUT / "lookup2.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[lookup2] DONE ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
