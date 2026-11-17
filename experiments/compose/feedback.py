"""Linearised autoregressive feedback as the bridge for self-composition (forward passes only).

reembed.py: a bridge fitted on unembedding rows of ordinary words is a valid input interface for the
successor head A (two steps on 91% of letters when it is handed the true next token), but A's own output
direction is a mixture, and A o R o A reached only 29% (letters) / 46% (other ordinal domains).

Here the bridge F is fitted on what the network actually produces: on ordinary text (WikiText), wherever
the model predicts the next token correctly and confidently,
      input : normalised final hidden state at position p        (output format, mixtures included)
      target: layer-21 attention input at position p + 1          (a position that holds the predicted token)
F is the linear map a decoding step implements (sample the prediction, embed it, run 21 layers), so
A o F o A is "one more decoding step inside the forward pass". Letters and all ordinal items are excluded.

Also reported: headroom with a non-linear decode (argmax of A's direct logits, then re-embed), which no
linear weight edit can implement exactly.
"""
import json
from pathlib import Path

import torch

from .bridge import A, DOMAINS, LAY, Lab, contexts, OUT
from .gate import AB, XS

CHUNK, MAX_CHUNKS, MAX_PAIRS = 128, 1500, 30000


def main():
    from datasets import load_dataset
    lab = Lab(); tok = lab.tok; P = lab.P; res = {}
    WU = (P["lm_head.weight"] if "lm_head.weight" in P else P["model.embed_tokens.weight"]).double()
    gf = P["model.norm.weight"].double()
    WO = P[f"model.layers.{A[0]}.self_attn.o_proj.weight"].double()[:, lab.sl(A[1])]
    WV = P[f"model.layers.{A[0]}.self_attn.v_proj.weight"].double()[lab.kvsl(A[1])]
    banned_words = {w.lower() for items in DOMAINS.values() for w in items} | set(AB) | set(AB.upper())
    banned = {i for w in banned_words for i in (tok(" " + w, add_special_tokens=False)["input_ids"][:1] + tok(w, add_special_tokens=False)["input_ids"][:1]
                                                 + tok(" " + w.capitalize(), add_special_tokens=False)["input_ids"][:1])}
    text = "\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"] if len(t) > 200)
    ids = tok(text[:4_000_000], add_special_tokens=False)["input_ids"]
    chunks = torch.tensor([ids[i:i + CHUNK] for i in range(0, min(len(ids) - CHUNK, MAX_CHUNKS * CHUNK), CHUNK)])
    st = {}

    def pre(mod, args, kwargs):
        st["n"] = (args[0] if args else kwargs["hidden_states"]).clone()
    Xs, Ys, toks = [], [], []
    ban = torch.zeros(WU.shape[0], dtype=torch.bool); ban[torch.tensor(sorted(banned))] = True
    with torch.inference_mode():
        for i in range(0, len(chunks), 32):
            b = chunks[i:i + 32]
            h = lab.layers[A[0]].self_attn.register_forward_pre_hook(pre, with_kwargs=True)
            try:
                out = lab.model(input_ids=b, output_hidden_states=True, use_cache=False)
            finally:
                h.remove()
            pr = out.logits[:, :-1].float().softmax(-1); top = pr.argmax(-1); conf = pr.max(-1).values
            ok = (top == b[:, 1:]) & (conf > 0.5) & ~ban[top]
            ok[:, :8] = False                                    # skip the start of each chunk (little context)
            hf = out.hidden_states[-1][:, :-1].double()          # final hidden state before the final norm
            x = gf * hf / hf.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
            Xs.append((x / x.norm(dim=-1, keepdim=True))[ok]); Ys.append(st["n"][:, 1:].double()[ok]); toks.append(top[ok])
            if sum(len(t) for t in toks) >= MAX_PAIRS:
                break
    X = torch.cat(Xs); Y = torch.cat(Ys); T = torch.cat(toks); n = len(X); ntr = int(0.9 * n)
    res["n_pairs"] = n; res["n_distinct_tokens"] = int(T.unique().numel())
    print(f"[feedback] {n} confident correct predictions from ordinary text ({res['n_distinct_tokens']} distinct tokens; letters and ordinal items excluded)", flush=True)

    def fit(lam, center):
        Xt, Yt = X[:ntr], Y[:ntr]; mx = Xt.mean(0) if center else torch.zeros_like(Xt[0]); my = Yt.mean(0) if center else torch.zeros_like(Yt[0])
        Xc, Yc = Xt - mx, Yt - my; G = Xc.T @ Xc; reg = lam * torch.trace(G) / G.shape[0]
        F = torch.linalg.solve(G + reg * torch.eye(G.shape[0], dtype=G.dtype), Xc.T @ Yc).T
        pred = (X[ntr:] - mx) @ F.T + my; r2 = 1 - float(((pred - Y[ntr:]) ** 2).sum() / ((Y[ntr:] - Yt.mean(0)) ** 2).sum())
        return F, mx, my, r2

    cases = {"letters": [], "other_ordinal": []}
    for x in XS:
        i = AB.index(x)
        for lay in LAY:
            cases["letters"].append((LAY[lay](list(AB[i - 3:i + 1])), AB[i + 1], AB[i + 2]))
    for dom, items in DOMAINS.items():
        if dom == "digits":
            continue
        for w, a, b in contexts(items, dom in ("days", "months")):
            if lab.first(a) != lab.first(b):
                cases["other_ordinal"] += [(LAY[lay](w), a, b) for lay in LAY]

    def rates(cs, fn, add=False):
        n1 = n2 = 0
        for text_, a, b in cs:
            edit = dict(targets=[A], fn=lambda st_: {A: fn(st_[("z",) + A]) - (0 if add else st_[("z",) + A])}) if fn else None
            lg, _ = lab.forward(text_, grab=[("z",) + A], edit=edit); top = int(lg.argmax())
            n1 += top == lab.first(a); n2 += top == lab.first(b)
        return dict(next=round(n1 / len(cs), 3), plus2=round(n2 / len(cs), 3), n=len(cs))

    for k, cs in cases.items():
        res[f"native_{k}"] = rates(cs, None); print(f"[feedback] native {k}: next {res[f'native_{k}']['next']:.2f} +2 {res[f'native_{k}']['plus2']:.2f}", flush=True)

    def unit_out(z):
        u = gf * (WO @ z.double()); return u / u.norm()

    best = None
    for center in (False, True):
        for lam in (1e-3, 1e-2, 1e-1):
            F, mx, my, r2 = fit(lam, center)
            row = dict(heldout_r2=r2)
            for k, cs in cases.items():
                for gamma in (1.0, 2.0, 4.0):
                    comp = lambda z, g=gamma: (g * (WV @ (F @ (unit_out(z) - mx) + my))).float()
                    row[f"{k}_replace_g{gamma}"] = rates(cs, comp)
                    row[f"{k}_add_g{gamma}"] = rates(cs, lambda z, c=comp: z + c(z))
            res[f"center{int(center)}_lam{lam}"] = row
            print(f"[feedback] centred={center} lambda {lam}: held-out R^2 {r2:.2f}", flush=True)
            for k in cases:
                print(f"[feedback]   {k}: A o F o A replacing A (+2/next): " + " ".join(f"g{g}:{row[f'{k}_replace_g{g}']['plus2']:.2f}/{row[f'{k}_replace_g{g}']['next']:.2f}" for g in (1.0, 2.0, 4.0))
                      + " | added to A: " + " ".join(f"g{g}:{row[f'{k}_add_g{g}']['plus2']:.2f}/{row[f'{k}_add_g{g}']['next']:.2f}" for g in (1.0, 2.0, 4.0)), flush=True)
            score = max(row[f"other_ordinal_replace_g{g}"]["plus2"] for g in (1.0, 2.0, 4.0))      # selection uses NON-letter domains only
            if best is None or score > best[0]:
                best = (score, center, lam, F, mx, my)
    _, center, lam, F, mx, my = best
    gsel = max((1.0, 2.0, 4.0), key=lambda g: res[f"center{int(center)}_lam{lam}"][f"other_ordinal_replace_g{g}"]["plus2"])
    res["selected_on_other_ordinal"] = dict(center=center, lam=lam, gamma=gsel, letters=res[f"center{int(center)}_lam{lam}"][f"letters_replace_g{gsel}"],
                                            other_ordinal=res[f"center{int(center)}_lam{lam}"][f"other_ordinal_replace_g{gsel}"])
    print(f"[feedback] SELECTED on non-letter domains: centred={center} lambda {lam} gamma {gsel}: LETTERS +2 {res['selected_on_other_ordinal']['letters']['plus2']:.2f} "
          f"next {res['selected_on_other_ordinal']['letters']['next']:.2f}; other ordinal +2 {res['selected_on_other_ordinal']['other_ordinal']['plus2']:.2f}", flush=True)

    # headroom: non-linear decode of A's direct logits, then the true unembedding row through F is not available; use the feedback map on a one-hot output state
    def hard(z):
        t = int((WU @ unit_out(z)).argmax()); e = WU[t] / WU[t].norm()
        return (gsel * (WV @ (F @ (e - mx) + my))).float(), t
    for k, cs in cases.items():
        n2 = dec = 0
        for text_, a, b in cs:
            _, s0 = lab.forward(text_, grab=[("z",) + A]); v, t = hard(s0[("z",) + A]); dec += t == lab.first(a)
            lg, _ = lab.forward(text_, grab=[("z",) + A], edit=dict(targets=[A], fn=lambda st_, v=v: {A: v - st_[("z",) + A]}))
            n2 += int(lg.argmax()) == lab.first(b)
        res[f"hard_decode_{k}"] = dict(a_direct_top1_is_next=round(dec / len(cs), 3), plus2=round(n2 / len(cs), 3))
        print(f"[feedback] {k}: A's own direct top-1 is the next item in {dec / len(cs):.2f}; hard decode then feedback -> +2 {n2 / len(cs):.2f}", flush=True)
    torch.save(dict(F=F.float().cpu(), mx=mx.float().cpu(), my=my.float().cpu(), center=center, lam=lam, gamma=gsel), OUT / "feedback_map.pt")
    (OUT / "feedback.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("[feedback] DONE", flush=True)


if __name__ == "__main__":
    main()
