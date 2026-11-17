"""Self-composition of the successor head through a GENERIC re-embedding map (forward passes only).

diag.py showed: (1) head A = L21h6 is the successor computation across domains (oracle value patch gives
two steps on 89-94% of cases); (2) a linear "+1" map fitted in A's output space does not transfer to
items it was not fitted on. So the bridge must not be fitted on ordinal items at all.

Here the bridge R is item-agnostic: it maps the output-format direction of ANY token t (its unembedding
row) to the layer-21 attention-input representation of a position holding token t. R is fitted on
several thousand ordinary word tokens; letters and every ordinal item are excluded from the fit.

Composition tested:  z_A  ->  output direction  g_f * W_O[A] z_A  ->  R  ->  W_V[A]  ->  A's output slot.
That is "apply A, re-embed its answer, apply A again", with no donor prompt and no task-specific fitting.
"""
import json
import random
from pathlib import Path

import torch

from .bridge import A, DOMAINS, LAY, Lab, contexts, OUT
from .gate import AB, XS

N_TOK = 6000


def main():
    lab = Lab(); tok = lab.tok; P = lab.P; res = {}
    WU = (P["lm_head.weight"] if "lm_head.weight" in P else P["model.embed_tokens.weight"]).double()
    gf = P["model.norm.weight"].double()
    WO = P[f"model.layers.{A[0]}.self_attn.o_proj.weight"].double()[:, lab.sl(A[1])]
    WV = P[f"model.layers.{A[0]}.self_attn.v_proj.weight"].double()[lab.kvsl(A[1])]
    banned = {w.lower() for items in DOMAINS.values() for w in items} | set(AB)
    pool = []
    for i in range(tok.vocab_size):
        s = tok.decode([i])
        if s.startswith(" ") and s[1:].isalpha() and s[1:].islower() and len(s) >= 4 and s[1:] not in banned:
            pool.append(i)
        if len(pool) >= N_TOK:
            break
    rr = random.Random("reembed"); prefix = tok("Sequence:", add_special_tokens=False)["input_ids"]
    ids = torch.tensor([prefix + [rr.choice(pool), rr.choice(pool), t] for t in pool])
    st = {}

    def pre(mod, args, kwargs):
        st["n"] = (args[0] if args else kwargs["hidden_states"])[:, -1].clone()
    feats = []
    with torch.inference_mode():
        for i in range(0, len(ids), 256):
            h = lab.layers[A[0]].self_attn.register_forward_pre_hook(pre, with_kwargs=True)
            try:
                lab.model(input_ids=ids[i:i + 256], use_cache=False)
            finally:
                h.remove()
            feats.append(st["n"].double())
    Y = torch.cat(feats); E = WU[torch.tensor(pool)]; E = E / E.norm(dim=-1, keepdim=True)
    ntr = int(0.9 * len(pool))
    print(f"[reembed] fitting R on {ntr} word tokens (letters and ordinal items excluded); {len(pool) - ntr} held out", flush=True)

    def fit(lam):
        X = E[:ntr]; G = X.T @ X; reg = lam * torch.trace(G) / G.shape[0]
        return torch.linalg.solve(G + reg * torch.eye(G.shape[0], dtype=X.dtype), X.T @ Y[:ntr]).T        # n21 ~ R e

    cases = {"letters": [], "calibration": []}
    for x in XS:
        i = AB.index(x)
        for lay in LAY:
            cases["letters"].append((LAY[lay](list(AB[i - 3:i + 1])), AB[i + 1], AB[i + 2]))
    for dom, items in DOMAINS.items():
        if dom == "digits":
            continue
        for w, a, b in contexts(items, dom in ("days", "months")):
            if lab.first(a) != lab.first(b):
                cases["calibration"] += [(LAY[lay](w), a, b) for lay in LAY]

    def rates(cs, fn=None, add=False):
        n1 = n2 = 0
        for text, a, b in cs:
            edit = None
            if fn:
                edit = dict(targets=[A], fn=lambda st: {A: fn(st[("z",) + A]) - (0 if add else st[("z",) + A])})
            lg, _ = lab.forward(text, grab=[("z",) + A], edit=edit); top = int(lg.argmax())
            n1 += top == lab.first(a); n2 += top == lab.first(b)
        return dict(next=round(n1 / len(cs), 3), plus2=round(n2 / len(cs), 3), n=len(cs))

    for k, cs in cases.items():
        res[f"native_{k}"] = rates(cs); print(f"[reembed] native {k}: next {res[f'native_{k}']['next']:.2f} +2 {res[f'native_{k}']['plus2']:.2f}", flush=True)
    for lam in (1e-3, 1e-2, 1e-1):
        R = fit(lam)
        pred = E[ntr:] @ R.T; r2 = 1 - float(((pred - Y[ntr:]) ** 2).sum() / ((Y[ntr:] - Y[:ntr].mean(0)) ** 2).sum())
        # sanity: does R's image of the letter x+1's own unembedding row act like "position holding x+1" for head A?
        def from_token(item):
            e = WU[lab.first(item)]; return (WV @ (R @ (e / e.norm()))).float()

        def comp(z, gamma=1.0):
            u = gf * (WO @ z.double()); return (gamma * (WV @ (R @ (u / u.norm())))).float()
        row = dict(heldout_r2=r2)
        for k, cs in cases.items():
            n2 = 0
            for text, a, b in cs:                      # upper bound for this bridge: feed A the re-embedded TRUE next item
                lg, _ = lab.forward(text, grab=[("z",) + A], edit=dict(targets=[A], fn=lambda st, a=a: {A: from_token(a) - st[("z",) + A]}))
                n2 += int(lg.argmax()) == lab.first(b)
            row[f"{k}_true_next_token_through_R"] = round(n2 / len(cs), 3)
            for gamma in (1.0, 2.0, 4.0):
                row[f"{k}_replace_g{gamma}"] = rates(cs, lambda z, g=gamma: comp(z, g))
                row[f"{k}_add_g{gamma}"] = rates(cs, lambda z, g=gamma: z + comp(z, g), add=False)
        res[f"lam{lam}"] = row
        print(f"[reembed] lambda {lam}: held-out R^2 {r2:.2f}", flush=True)
        for k in cases:
            print(f"[reembed]   {k}: re-embedded TRUE next token fed to A -> +2 {row[f'{k}_true_next_token_through_R']:.2f}", flush=True)
            print(f"[reembed]   {k}: A o R o A replacing A's output (+2 / next): " + " ".join(f"g{g}:{row[f'{k}_replace_g{g}']['plus2']:.2f}/{row[f'{k}_replace_g{g}']['next']:.2f}" for g in (1.0, 2.0, 4.0)), flush=True)
            print(f"[reembed]   {k}: A o R o A added to A's output   (+2 / next): " + " ".join(f"g{g}:{row[f'{k}_add_g{g}']['plus2']:.2f}/{row[f'{k}_add_g{g}']['next']:.2f}" for g in (1.0, 2.0, 4.0)), flush=True)
    (OUT / "reembed.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("[reembed] DONE", flush=True)


if __name__ == "__main__":
    main()
