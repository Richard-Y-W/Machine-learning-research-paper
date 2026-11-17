"""Step 1 of the frozen procedure: locate the successor head of a model (forward passes only).

Rule (fixed before it was applied to any model other than Qwen3-0.6B-Base):
  1. On native sequence-continuation prompts (letters, number words, weekdays, months, ordinals; four layouts),
     zero each head's output at the final position and rank heads by the drop in next-item logit margin.
  2. For the six highest-ranked heads, replace the head's output by its own value vector taken from the context
     that ends one item later (oracle value patch) and measure how often the model then answers two steps ahead.
  3. The successor head is the one with the highest two-step rate.
Lookup prompts are not used here.

    EVO_MODEL=<model> python -m experiments.compose.find_head
"""
import json
import os

import torch

from .bridge import DOMAINS, LAY, Lab, OUT, contexts
from .gate import AB, XS


def main():
    OUT.mkdir(parents=True, exist_ok=True); lab = Lab(); L = len(lab.layers); H = lab.model.config.num_attention_heads
    cases = []                       # (text, text one later, next, nextnext)
    for x in XS:
        i = AB.index(x)
        for lay in LAY:
            cases.append((LAY[lay](list(AB[i - 3:i + 1])), LAY[lay](list(AB[i - 2:i + 2])), AB[i + 1], AB[i + 2]))
    for dom, items in DOMAINS.items():
        if dom == "digits":
            continue
        for w, a, b in contexts(items, dom in ("days", "months")):
            if lab.first(a) != lab.first(b):
                cases += [(LAY[lay](w), LAY[lay](w[1:] + [a]), a, b) for lay in LAY]
    scan_cases = cases[::3]

    def margin_and_acc(cs, edit_for=None):
        m = acc = 0.0
        for text, _, a, b in cs:
            lg, _ = lab.forward(text, grab=edit_for["grab"] if edit_for else (), edit=edit_for["edit"] if edit_for else None)
            m += float(lg[lab.first(a)] - lg[lab.first(b)]); acc += int(lg.argmax()) == lab.first(a)
        return m / len(cs), acc / len(cs)

    base_m, base_acc = margin_and_acc(scan_cases)
    print(f"[find] {os.environ.get('EVO_MODEL')}: {L} layers x {H} heads; native next-item accuracy {base_acc:.2f} on {len(scan_cases)} prompts", flush=True)
    scan = []
    for l in range(L):
        for h in range(H):
            e = dict(grab=[("z", l, h)], edit=dict(targets=[(l, h)], fn=lambda st, l=l, h=h: {(l, h): -st[("z", l, h)]}))
            m, acc = margin_and_acc(scan_cases, e); scan.append(dict(layer=l, head=h, d_margin=m - base_m, acc=acc))
    scan.sort(key=lambda r: r["d_margin"])
    print("[find] largest drops in next-item margin when a head is removed: " + ", ".join(f"L{r['layer']}h{r['head']} {r['d_margin']:+.2f}" for r in scan[:8]), flush=True)
    oracle = []
    for r in scan[:6]:
        l, h = r["layer"], r["head"]; n2 = 0
        for text, text2, a, b in cases:
            _, s2 = lab.forward(text2, grab=[("v", l, h)]); v = s2[("v", l, h)]
            lg, _ = lab.forward(text, grab=[("z", l, h)], edit=dict(targets=[(l, h)], fn=lambda st, v=v, l=l, h=h: {(l, h): v - st[("z", l, h)]}))
            n2 += int(lg.argmax()) == lab.first(b)
        oracle.append(dict(layer=l, head=h, two_step=n2 / len(cases)))
        print(f"[find] oracle value patch L{l}h{h}: two-step rate {n2 / len(cases):.2f}", flush=True)
    best = max(oracle, key=lambda r: r["two_step"])
    json.dump(dict(model=os.environ.get("EVO_MODEL"), native_acc=base_acc, scan=scan[:30], oracle=oracle, successor_head=[best["layer"], best["head"]]),
              open(OUT / "find_head.json", "w"), indent=1)
    print(f"[find] SUCCESSOR HEAD: L{best['layer']}h{best['head']} (two-step {best['two_step']:.2f})", flush=True)


if __name__ == "__main__":
    main()
