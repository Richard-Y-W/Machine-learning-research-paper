"""Search for a second native operation that the frozen procedure could route into (forward passes only).

The successor head was found by: ablation ranking on native prompts where the item is the final token, then an
oracle value patch (the head's output := its value vector from a prompt ending in a different item) and a check
that the answer becomes the operation applied to the DONOR item. The same rule is applied here to four other
operations that the model performs natively in few-shot form with the item as the final token:
  capitalise (paris -> Paris), plural (cat -> cats), past tense (walk -> walked), antonym (hot -> cold).
A candidate operation head must reach a high donor-operation rate; otherwise the branch stops.

    EVO_MODEL=<model> python -m experiments.compose.second_op
"""
import json
import os
import random

import torch

from .bridge import Lab, OUT

OPS = {
    "capitalise": [(w, w.capitalize()) for w in "paris london berlin madrid rome tokyo china india france spain italy japan texas ohio monday friday april june david peter".split()],
    "plural": [(w, w + "s") for w in "cat dog car book tree house road river bird hand door window table chair girl boy king queen ship game".split()],
    "past": [(w, w + "ed") for w in "walk talk jump play call look want need help start open turn pull push kick pick cook clean paint watch".split()],
    "antonym": [("hot", "cold"), ("big", "small"), ("up", "down"), ("good", "bad"), ("fast", "slow"), ("high", "low"), ("old", "young"), ("rich", "poor"),
                ("light", "dark"), ("early", "late"), ("hard", "soft"), ("open", "closed"), ("wet", "dry"), ("happy", "sad"), ("long", "short"), ("full", "empty")],
}


def main():
    OUT.mkdir(parents=True, exist_ok=True); lab = Lab(); tok = lab.tok; L = len(lab.layers); H = lab.model.config.num_attention_heads; res = {}
    one = lambda s: len(tok(" " + s, add_special_tokens=False)["input_ids"]) == 1
    for op, pairs in OPS.items():
        pairs = [(a, b) for a, b in pairs if one(a) and one(b) and lab.first(a) != lab.first(b)]
        rr = random.Random("second-op|" + op); cases = []
        for a, b in pairs:
            for _ in range(2):
                shots = rr.sample([p for p in pairs if p[0] != a], int(os.environ.get("WSP_SHOTS", "3")))
                cases.append((" ".join(f"{x} {y}" for x, y in shots) + f" {a}", a, b, shots))
        def acc_margin(cs, edit_of=None):
            hit = m = 0.0
            for text, a, b, _ in cs:
                e = edit_of(text) if edit_of else None
                lg, _ = lab.forward(text, grab=e["grab"] if e else (), edit=e["edit"] if e else None)
                hit += int(lg.argmax()) == lab.first(b); m += float(lg[lab.first(b)] - lg[lab.first(a)])
            return hit / len(cs), m / len(cs)
        acc, base_m = acc_margin(cases)
        print(f"[op2] {op}: {len(pairs)} single-token pairs, {len(cases)} prompts, native accuracy {acc:.2f}", flush=True)
        if acc < 0.8:
            res[op] = dict(native_acc=acc, verdict="native competence below .80; stop"); print(f"[op2] {op}: stop (native competence too low)", flush=True); continue
        scan = []
        for l in range(L):
            for h in range(H):
                e = lambda text, l=l, h=h: dict(grab=[("z", l, h)], edit=dict(targets=[(l, h)], fn=lambda st: {(l, h): -st[("z", l, h)]}))
                a_, m = acc_margin(cases[::2], e); scan.append(dict(layer=l, head=h, d_margin=m - base_m, acc=a_))
        scan.sort(key=lambda r: r["d_margin"])
        print(f"[op2] {op}: largest margin drops: " + ", ".join(f"L{r['layer']}h{r['head']} {r['d_margin']:+.2f}" for r in scan[:6]), flush=True)
        oracle = []
        for r in scan[:6]:
            l, h = r["layer"], r["head"]; hit = own = n = 0
            for text, a, b, shots in cases:
                da, db = rr.choice([p for p in pairs if p[0] != a and p not in shots])
                donor = " ".join(f"{x} {y}" for x, y in shots) + f" {da}"
                _, s2 = lab.forward(donor, grab=[("v", l, h)]); v = s2[("v", l, h)]
                lg, _ = lab.forward(text, grab=[("z", l, h)], edit=dict(targets=[(l, h)], fn=lambda st, v=v: {(l, h): v - st[("z", l, h)]}))
                top = int(lg.argmax()); hit += top == lab.first(db); own += top == lab.first(b); n += 1
            oracle.append(dict(layer=l, head=h, donor_operation=hit / n, own_operation=own / n))
            print(f"[op2] {op}: oracle value patch L{l}h{h}: answer is op(donor) {hit / n:.2f}, op(own) {own / n:.2f}", flush=True)
        best = max(oracle, key=lambda r: r["donor_operation"])
        res[op] = dict(native_acc=acc, scan=scan[:20], oracle=oracle, best=best, verdict="candidate" if best["donor_operation"] >= 0.5 else "no single head carries the operation; stop")
        print(f"[op2] {op}: best L{best['layer']}h{best['head']} donor-operation {best['donor_operation']:.2f} -> {res[op]['verdict']}", flush=True)
    json.dump(dict(model=os.environ.get("EVO_MODEL"), ops=res), open(OUT / "second_op.json", "w"), indent=1)
    print("[op2] DONE", flush=True)


if __name__ == "__main__":
    main()
