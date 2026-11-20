"""Second-operation screen, expanded (reviewer item 3). Forward/backward passes only; no fitting, no routing yet.

second_op.py found no single head whose value vector carries capitalise/plural/past/antonym (donor-operation 0.00
everywhere). Here the search is widened along the two axes the routing edit actually needs, staged so that no
routing fit is attempted unless an oracle intervention already works:

  Stage 1  native competence: 8-shot 'x -> f(x)' prompts, f(item) as the target. Continue if accuracy >= 0.80.
  Stage 2  head importance: attribution patching (gradient x activation of every head's final-position output on the
           margin logit f(x) - logit x), top 16 verified by real ablation.
  Stage 3  oracle interventions with donor prompts (same shots, a different query item d):
             value patch    heads' outputs := their value vectors at the donor's final position (what a routing edit
                            could deliver: the head reads the item token through its own OV)
             output patch   heads' outputs := their outputs on the donor prompt (upper bound: is f(d) carried at all)
           for the top 6 heads singly and the top 2/4/8/16 jointly. Report the rate of answering f(d).
  Verdict  'routable candidate' if a value patch of <= 8 heads gives f(d) on >= 0.50 of prompts.

Operations: the four earlier ones plus seven non-succession mappings with single-token items on both sides.

    EVO_MODEL=Qwen/Qwen3-0.6B-Base XF_OUT=results/second_op2_0p6b python -m experiments.compose.second_op2
"""
import json
import os
import random
import time

import torch

from .second_op import OPS as OPS0
from .xfamily import XO, Arch

OPS = dict(OPS0)
OPS.update({
    "upper_letter": [(c, c.upper()) for c in "abcdefghijklmnopqrstuvwxyz"],
    "card_to_ord": list(zip("one two three four five six seven eight nine ten eleven twelve".split(),
                            "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth".split())),
    "word_to_digit": list(zip("one two three four five six seven eight nine".split(), "1 2 3 4 5 6 7 8 9".split())),
    "digit_to_word": list(zip("1 2 3 4 5 6 7 8 9".split(), "one two three four five six seven eight nine".split())),
    "male_female": [("king", "queen"), ("man", "woman"), ("boy", "girl"), ("father", "mother"), ("brother", "sister"), ("son", "daughter"), ("uncle", "aunt"),
                    ("husband", "wife"), ("actor", "actress"), ("prince", "princess"), ("nephew", "niece"), ("he", "she"), ("his", "her"), ("lord", "lady"),
                    ("waiter", "waitress"), ("god", "goddess"), ("male", "female"), ("men", "women"), ("boys", "girls"), ("dad", "mom")],
    "country_capital": [("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"), ("Japan", "Tokyo"), ("China", "Beijing"), ("Russia", "Moscow"),
                        ("Egypt", "Cairo"), ("England", "London"), ("Greece", "Athens"), ("Austria", "Vienna"), ("Poland", "Warsaw"), ("Ireland", "Dublin"),
                        ("Norway", "Oslo"), ("Sweden", "Stockholm"), ("Kenya", "Nairobi"), ("Peru", "Lima"), ("Cuba", "Havana"), ("Thailand", "Bangkok"), ("Portugal", "Lisbon")],
    "english_french": [("dog", "chien"), ("cat", "chat"), ("house", "maison"), ("water", "eau"), ("red", "rouge"), ("black", "noir"), ("white", "blanc"), ("book", "livre"),
                       ("sun", "soleil"), ("moon", "lune"), ("man", "homme"), ("woman", "femme"), ("city", "ville"), ("bread", "pain"), ("wine", "vin"), ("king", "roi"),
                       ("sea", "mer"), ("night", "nuit"), ("day", "jour"), ("friend", "ami"), ("green", "vert"), ("tree", "arbre"), ("car", "voiture"), ("milk", "lait")],
})
SHOTS = 8; PER_PAIR = 3


def main():
    XO.mkdir(parents=True, exist_ok=True); ar = Arch(os.environ.get("EVO_MODEL", "Qwen/Qwen3-0.6B-Base")); t0 = time.time()
    resf = XO / "second_op2.json"; res = json.loads(resf.read_text()) if resf.exists() else {}; res["model"] = ar.name
    one = lambda s: len(ar.ids(" " + s)) == 1
    only = os.environ.get("XF_OPS")
    for op, pairs in OPS.items():
        if (only and op not in only.split(",")) or op in res:
            continue
        pairs = [(a, b) for a, b in pairs if one(a) and one(b) and ar.first(a) != ar.first(b)]
        if len(pairs) < SHOTS + 3:
            res[op] = dict(pairs=len(pairs), verdict="too few single-token pairs"); print(f"[op2b] {op}: only {len(pairs)} single-token pairs; skip", flush=True); continue
        rr = random.Random(f"second-op2|{op}"); cases = []
        for a, b in pairs:
            for _ in range(PER_PAIR):
                shots = rr.sample([p for p in pairs if p[0] != a], SHOTS); d = rr.choice([p for p in pairs if p[0] != a and p not in shots])
                pre = "".join(f"{x} -> {y}\n" for x, y in shots)
                cases.append(dict(text=pre + f"{a} ->", donor=pre + f"{d[0]} ->", a=a, b=b, da=d[0], db=d[1]))
        T = [c["text"] for c in cases]; Dn = [c["donor"] for c in cases]
        tb = torch.tensor([ar.first(c["b"]) for c in cases]); ta = torch.tensor([ar.first(c["a"]) for c in cases]); tdb = [ar.first(c["db"]) for c in cases]
        acc = float((ar.last(T).argmax(-1) == tb).float().mean())
        print(f"[op2b] {op}: {len(pairs)} pairs, {len(cases)} prompts, native 8-shot accuracy {acc:.2f}", flush=True)
        if acc < 0.8:
            res[op] = dict(pairs=len(pairs), native_acc=acc, verdict="native competence below .80; stop"); resf.write_text(json.dumps(res, indent=1)); continue
        # ---- Stage 2: attribution patching over all heads (final position), batch-wise
        attr = torch.zeros(ar.L, ar.H)
        for i in range(0, len(cases), 16):
            ids, mask = ar.batch(T[i:i + 16]); zs = {}; hk = []
            for l in range(ar.L):
                def pre(mod, args, l=l):
                    z = args[0]; z.retain_grad(); zs[l] = z
                hk.append(ar.oproj(l).register_forward_pre_hook(pre))
            try:
                with torch.enable_grad():
                    emb = ar.body.get_input_embeddings()(ids).detach().requires_grad_(True)
                    h = ar.body(inputs_embeds=emb, attention_mask=mask, use_cache=False).last_hidden_state[:, -1]
                    lg = ar.head_out(h).float(); r = torch.arange(len(ids))
                    (lg[r, tb[i:i + 16]] - lg[r, ta[i:i + 16]]).sum().backward()
            finally:
                for x in hk:
                    x.remove()
            for l, z in zs.items():
                g = (z.grad[:, -1] * z.detach()[:, -1]).view(len(ids), ar.H, ar.hd).sum(-1)         # first-order effect of zeroing = -g
                attr[l] += g.sum(0)
        attr /= len(cases); flat = sorted(((float(attr[l, h]), l, h) for l in range(ar.L) for h in range(ar.H)), reverse=True)[:16]
        base = ar.last(T); r = torch.arange(len(cases)); base_m = float((base[r, tb] - base[r, ta]).mean()); ver = []
        for a_, l, h in flat:
            lg = ar.last(T, [ar.zhook(l, h)]); ver.append(dict(layer=l, head=h, attribution=a_, d_margin=float((lg[r, tb] - lg[r, ta]).mean()) - base_m))
        ver.sort(key=lambda x: x["d_margin"]); top = [(x["layer"], x["head"]) for x in ver]
        print(f"[op2b] {op}: verified ablation drops: " + ", ".join(f"L{x['layer']}h{x['head']} {x['d_margin']:+.2f}" for x in ver[:8]), flush=True)

        # ---- Stage 3: oracle value / output patches
        def donor_acts(heads, kind):
            out = {}
            for l, h in heads:
                st = []; ar.last(Dn, [ar.vhook(l, h, st) if kind == "value" else ar.zhook(l, h, store=st)]); out[(l, h)] = torch.cat(st)
            return out

        def patched(heads, kind):
            acts = donor_acts(heads, kind); toks = ar.last(T, [ar.zhook(l, h, acts[(l, h)]) for l, h in heads]).argmax(-1).tolist()
            return dict(donor_op=sum(t == d for t, d in zip(toks, tdb)) / len(cases), own_op=sum(t == int(b) for t, b in zip(toks, tb)) / len(cases))

        groups = {f"L{l}h{h}": [(l, h)] for l, h in top[:6]}
        groups.update({f"top{k}": top[:k] for k in (2, 4, 8, 16)})
        orc = {}
        for gname, heads in groups.items():
            orc[gname] = dict(value=patched(heads, "value"), output=patched(heads, "output"))
            print(f"[op2b] {op}: {gname:9s} value patch f(donor) {orc[gname]['value']['donor_op']:.2f} (own {orc[gname]['value']['own_op']:.2f}) | "
                  f"output patch f(donor) {orc[gname]['output']['donor_op']:.2f} (own {orc[gname]['output']['own_op']:.2f})", flush=True)
        small = [g for g in groups if len(groups[g]) <= 8]; best = max(small, key=lambda g: orc[g]["value"]["donor_op"])
        verdict = "routable candidate" if orc[best]["value"]["donor_op"] >= 0.5 else "no head group of <= 8 carries the operation through its value path"
        res[op] = dict(pairs=len(pairs), prompts=len(cases), native_acc=acc, verified=ver, oracle=orc, best_value_group=best, verdict=verdict)
        resf.write_text(json.dumps(res, indent=1)); print(f"[op2b] {op}: best value group {best} -> {verdict}", flush=True)
    print(f"[op2b] DONE ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
