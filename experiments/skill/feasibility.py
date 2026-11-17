"""Closed-form feasibility certificate for the ENC skill (no gradients, no API).

For each layer l: h_l(x) = residual entering layer l at the last position of "ENC: x ->".
Fit an affine map (ridge / reduced-rank) from h_l(x) to the output embedding e(succ x) on PRACTICE
letters; score held-out letters by retrieval among the 26 letter embeddings. A shuffled-target
control gives the memorization-only floor. Also: interference cost of a trigger-keyed rank-1 write
(ROME algebra): E||dW k||^2 = ||delta||^2 / (k*^T C^-1 k*).
"""
import json, string
import numpy as np, torch
from .pilot import Skill, PRACTICE, HELD, LETTERS, succ

def main():
    sk = Skill(); m = sk.model; L = sk.n_layers
    E = (m.lm_head.weight if hasattr(m, "lm_head") else m.model.embed_tokens.weight).detach().float()
    lid = {c: sk.ans[c] for c in string.ascii_lowercase}
    U = torch.stack([E[lid[c]] for c in string.ascii_lowercase]).numpy()          # 26 x d letter output embeddings
    texts = [f"ENC: {c} ->" for c in LETTERS]; ids, mask, last = sk._batch(texts)
    with torch.inference_mode():
        out = m(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
    H = [out.hidden_states[l][torch.arange(len(texts)), last].float().numpy() for l in range(L + 1)]
    pi = [LETTERS.index(c) for c in PRACTICE]; hi = [LETTERS.index(c) for c in HELD]
    Y = np.stack([U[string.ascii_lowercase.index(succ(c))] for c in LETTERS])
    rng = np.random.default_rng(0)
    def fit_eval(X, Yt, rank=None, lam=1e-2):
        mu_x, mu_y = X[pi].mean(0), Yt[pi].mean(0)
        Xc, Yc = X[pi] - mu_x, Yt[pi] - mu_y
        s = np.linalg.norm(Xc, axis=1).mean() + 1e-8; Xc = Xc / s
        A = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ Yc) if X.shape[1] <= 64 else Xc.T @ np.linalg.solve(Xc @ Xc.T + lam * np.eye(len(pi)), Yc)
        if rank is not None:                       # reduced-rank regression: project fitted values onto top-r directions
            Uf, S, Vt = np.linalg.svd(Xc @ A, full_matrices=False); A = A @ Vt[:rank].T @ Vt[:rank]
        pred = ((X - mu_x) / s) @ A + mu_y
        tgt = [string.ascii_lowercase.index(succ(LETTERS[i])) for i in range(len(LETTERS))]
        sc = pred @ U.T
        return (float(np.mean([sc[i].argmax() == tgt[i] for i in pi])), float(np.mean([sc[i].argmax() == tgt[i] for i in hi])))
    res = {"layers": {}}
    for l in range(L + 1):
        X = H[l]
        r = {"full": fit_eval(X, Y)}
        for k in (1, 2, 4, 8): r[f"rank{k}"] = fit_eval(X, Y, rank=k)
        sh = []
        for _ in range(20):
            perm = rng.permutation(len(LETTERS)); sh.append(fit_eval(X, Y[perm])[1])
        r["shuffled_held_mean"] = float(np.mean(sh))
        # alphabet position decodability from top PCs (practice fit, held-out R^2)
        Xc = X - X[pi].mean(0); Uu, S, Vt = np.linalg.svd(Xc[pi], full_matrices=False); Z = Xc @ Vt[:6].T
        pos = np.arange(len(LETTERS), dtype=float)
        w = np.linalg.lstsq(np.c_[Z[pi], np.ones(len(pi))], pos[pi], rcond=None)[0]
        ph = np.c_[Z[hi], np.ones(len(hi))] @ w
        r["pos_R2_held"] = float(1 - ((ph - pos[hi]) ** 2).sum() / ((pos[hi] - pos[hi].mean()) ** 2).sum())
        res["layers"][l] = r
    # interference cost of binding the trigger at an MLP (ROME algebra), layer 14 and 20
    S2 = __import__("json").loads(open("results/evolution/splits.json").read())
    wiki = torch.tensor(S2["wiki_dev"]["items"])
    for l in (8, 14, 20):
        cap = {}
        h = m.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda mod, a: cap.__setitem__("k", a[0]))
        with torch.inference_mode():
            m(input_ids=wiki, use_cache=False); Kw = cap["k"].reshape(-1, cap["k"].shape[-1]).float().numpy()
            m(input_ids=ids, attention_mask=mask, use_cache=False); Kt = cap["k"][torch.arange(len(texts)), last].float().numpy()
        h.remove()
        C = Kw.T @ Kw / len(Kw); lam = 1e-3 * np.trace(C) / C.shape[0]; Ci = np.linalg.inv(C + lam * np.eye(C.shape[0]))
        kt = Kt[pi].mean(0); maha_t = float(kt @ Ci @ kt)
        maha_w = float(np.mean(np.einsum("ij,jk,ik->i", Kw[:200], Ci, Kw[:200])))
        res.setdefault("interference", {})[l] = dict(maha_trigger=maha_t, maha_typical_token=maha_w,
                                                      relative_interference=maha_w / maha_t)
    open("results/skill/theory/feasibility.json", "w").write(json.dumps(res, indent=1))
    print("layer | full(prac,held) | rank1 | rank2 | rank4 | rank8 | shuffled-held | pos R2 held")
    for l, r in res["layers"].items():
        print(l, r["full"], r["rank1"][1], r["rank2"][1], r["rank4"][1], r["rank8"][1], round(r["shuffled_held_mean"], 3), round(r["pos_R2_held"], 2))
    print(json.dumps(res["interference"], indent=1))

if __name__ == "__main__":
    main()
