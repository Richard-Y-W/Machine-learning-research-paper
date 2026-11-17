"""Lookup -> successor: joining two native computations by a closed-form weight edit (no gradients, no API).

Prompt (Base model, plain completion):      oak: h / pine: c / elm: q / pine:
Native behaviour: " c" (lookup). Target after the edit: " d" (successor of the looked-up value).

Components: A = L21h6, the successor head (gate.py); H = retrieval heads found here by ablation on lookups
with ORDINARY-WORD values. The edit rewrites H's output projection so that, instead of copying the value it
retrieved, it writes A's successor code for it:      W_O[H]  <-  gamma * W_O[A] C_H
C_H is a ridge map from H's retrieved-value coordinates to A's value vector for the same token, fitted ONLY
on lookups whose values are ordinary words. No ordinal item and no composed example is used in fitting.

Steps: 0 native rates; 1 oracle: A handed the retrieved item; 2 retrieval-head scan; 3 oracle: H's output
replaced by A's code; 4 weight edit, with controls (shuffled pairing, wrong output projection, swapped
retrieval) and collateral measures.
"""
import json
import random
import time
from pathlib import Path

import torch

from ..evolution import backend
from .bridge import A, DOMAINS, Lab, HD, OUT
from .gate import AB

KEYS = ("oak pine elm maple cedar birch willow river stone cloud tiger eagle horse piano violin castle bridge garden window candle "
        "silver copper marble velvet cotton pepper butter lemon cherry walnut").split()
ORD = {"letters": list(AB), "numwords": DOMAINS["numwords"], "days": DOMAINS["days"], "months": DOMAINS["months"], "ordinals": DOMAINS["ordinals"]}
CYCLIC = {"days", "months"}
BATCH = 64


class LK:
    def __init__(self):
        self.lab = Lab(); self.tok = self.lab.tok; self.model = self.lab.model; self.layers = self.lab.layers; self.P = self.lab.P
        self.L = len(self.layers); self.H = self.model.config.num_attention_heads
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0

    def first(self, item):
        return self.tok(" " + item, add_special_tokens=False)["input_ids"][0]

    def batch(self, texts):
        enc = [self.tok(t, add_special_tokens=False)["input_ids"] for t in texts]; T = max(map(len, enc))
        ids = torch.full((len(enc), T), self.pad); mask = torch.zeros((len(enc), T), dtype=torch.long)
        for r, e in enumerate(enc):
            ids[r, T - len(e):] = torch.tensor(e); mask[r, T - len(e):] = 1          # left padding: the final position is always -1
        return ids, mask

    @torch.inference_mode()
    def fwd(self, texts, hooks=()):
        """hooks: list of functions(batch_slice) -> handle. Returns final-position logits for all texts."""
        outs = []
        for i in range(0, len(texts), BATCH):
            sl = slice(i, i + BATCH); ids, mask = self.batch(texts[sl]); hs = [h(sl) for h in hooks]
            try:
                outs.append(self.model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, -1].float())
            finally:
                for h in hs:
                    h.remove()
        return torch.cat(outs)

    def hsl(self, h):
        return slice(h * HD, (h + 1) * HD)

    # hooks (all act on the final position)
    def grab_z(self, layer, head, store):
        def reg(sl):
            def pre(mod, args):
                store.append(args[0][:, -1, self.hsl(head)].clone())
            return self.layers[layer].self_attn.o_proj.register_forward_pre_hook(pre)
        return reg

    def set_z(self, layer, head, vecs=None):
        """vecs None: zero the head; else a (N,128) tensor aligned with the texts."""
        def reg(sl):
            def pre(mod, args):
                z = args[0].clone(); z[:, -1, self.hsl(head)] = 0 if vecs is None else vecs[sl].to(z.dtype)
                return (z,)
            return self.layers[layer].self_attn.o_proj.register_forward_pre_hook(pre)
        return reg

    def add_out(self, layer, delta):
        """Adds delta (N,1024) to the attention output of `layer` at the final position."""
        def reg(sl):
            def hook(mod, inp, out):
                out = out.clone(); out[:, -1] += delta[sl].to(out.dtype)
                return out
            return self.layers[layer].self_attn.o_proj.register_forward_hook(hook)
        return reg

    @torch.inference_mode()
    def carrier_value(self, items, seed="carrier"):
        """A's value vector at a position holding the item, in a neutral carrier context."""
        rr = random.Random(seed); st = []
        kv = self.lab.kvsl(A[1])

        def reg(sl):
            def hook(mod, inp, out):
                st.append(out[:, -1, kv].clone())
            return self.layers[A[0]].self_attn.v_proj.register_forward_hook(hook)
        texts = ["Sequence: " + " ".join(rr.sample(KEYS, 2)) + " " + it for it in items]
        self.fwd(texts, [reg])
        return torch.cat(st)


def succ(dom, item):
    items = ORD[dom]; i = items.index(item)
    return items[(i + 1) % len(items)] if dom in CYCLIC else (items[i + 1] if i + 1 < len(items) else None)


def make(rr, values_pool, dom=None, n_pairs=None):
    """One lookup prompt. Returns (text, retrieved value, successor or None)."""
    for _ in range(200):
        n = n_pairs or rr.choice([3, 4]); keys = rr.sample(KEYS, n); vals = rr.sample(values_pool, n); q = rr.randrange(n)
        if dom:
            idx = [ORD[dom].index(v) for v in vals]; s = succ(dom, vals[q])
            if s is None or s in vals or any(abs(a - b) == 1 for i, a in enumerate(idx) for b in idx[i + 1:]):
                continue                                   # no accidental sequences; the target successor is not on the page
        else:
            s = None
        text = "".join(f"{k}: {v}\n" for k, v in zip(keys, vals)) + f"{keys[q]}:"
        return text, vals[q], s
    raise RuntimeError("could not build a prompt")


def score(lk, logits, cases):
    top = logits.argmax(-1).tolist(); n = len(cases)
    look = sum(t == lk.first(v) for t, (_, v, s) in zip(top, cases)); su = sum(s is not None and t == lk.first(s) for t, (_, v, s) in zip(top, cases))
    return dict(lookup=round(look / n, 3), successor=round(su / n, 3), n=n)


def main():
    OUT.mkdir(parents=True, exist_ok=True); lk = LK(); tok = lk.tok; t0 = time.time(); res = {}
    banned = {w.lower() for items in ORD.values() for w in items} | set(KEYS)
    words = []
    for i in range(tok.vocab_size):
        s = tok.decode([i])
        if s.startswith(" ") and s[1:].isalpha() and s[1:].islower() and 4 <= len(s) <= 9 and s[1:] not in banned:
            words.append(s[1:])
        if len(words) >= 3000:
            break
    rr = random.Random("lookup-v1")
    ordv = {d: [x for x in ORD[d] if len(tok(" " + x, add_special_tokens=False)["input_ids"]) == 1] for d in ORD}
    ordv["letters"] = list(AB[1:25])
    cal = [make(rr, words) for _ in range(1200)]; cal_fit, cal_val = cal[:1000], cal[1000:]
    ev = {d: [make(rr, ordv[d], d) for _ in range(200)] for d in ORD}
    T = lambda cases: [c[0] for c in cases]

    # ---- 0 native
    res["native_words"] = score(lk, lk.fwd(T(cal_val)), cal_val)
    res["native"] = {d: score(lk, lk.fwd(T(ev[d])), ev[d]) for d in ORD}
    print(f"[lookup] 0 native lookup, ordinary-word values: {res['native_words']['lookup']:.2f}", flush=True)
    for d in ORD:
        print(f"[lookup] 0 native {d:9s}: lookup {res['native'][d]['lookup']:.2f} successor {res['native'][d]['successor']:.2f}", flush=True)

    WOA = lk.P[f"model.layers.{A[0]}.self_attn.o_proj.weight"][:, lk.hsl(A[1])].float()

    # ---- 1 oracle: A's output := its value vector for the retrieved item
    res["oracle_A"] = {}
    for d in ORD:
        va = lk.carrier_value([c[1] for c in ev[d]])
        res["oracle_A"][d] = score(lk, lk.fwd(T(ev[d]), [lk.set_z(A[0], A[1], va)]), ev[d])
        print(f"[lookup] 1 oracle, A handed the retrieved item ({d}): lookup {res['oracle_A'][d]['lookup']:.2f} successor {res['oracle_A'][d]['successor']:.2f}", flush=True)

    # ---- 2 retrieval-head scan on ordinary-word lookups (ablate at the final position)
    scan_cases = cal_fit[:96]; base_lg = lk.fwd(T(scan_cases)); tgt = torch.tensor([lk.first(c[1]) for c in scan_cases]); ar = torch.arange(len(scan_cases))
    base_m = float(base_lg[ar, tgt].mean()); scan = []
    for l in range(lk.L):
        for h in range(lk.H):
            lg = lk.fwd(T(scan_cases), [lk.set_z(l, h)])
            scan.append(dict(layer=l, head=h, d_logit=float(lg[ar, tgt].mean()) - base_m, acc=float((lg.argmax(-1) == tgt).float().mean())))
    scan.sort(key=lambda r: r["d_logit"]); res["retrieval_scan"] = scan[:40]
    print("[lookup] 2 heads whose removal hurts lookup most (change in correct-value logit, accuracy after removal):", flush=True)
    for r in scan[:8]:
        print(f"[lookup] 2   L{r['layer']}h{r['head']}: {r['d_logit']:+.2f}  acc {r['acc']:.2f}", flush=True)
    HS = [(r["layer"], r["head"]) for r in scan if (r["layer"], r["head"]) != A][:4]
    res["retrieval_heads"] = [f"L{l}h{h}" for l, h in HS]

    def z_of(cases, heads):
        st = {hd: [] for hd in heads}
        lk.fwd(T(cases), [lk.grab_z(l, h, st[(l, h)]) for l, h in heads])
        return {hd: torch.cat(v) for hd, v in st.items()}

    # ---- 3 oracle: retrieval heads' outputs replaced by A's code for the retrieved item, at the heads' own layers
    res["oracle_H"] = {}
    for k in (1, 2, 4):
        heads = HS[:k]
        for d in ORD:
            va = lk.carrier_value([c[1] for c in ev[d]]); code = va.float() @ WOA.T                 # (N,1024): what A would write
            hooks = [lk.set_z(l, h) for l, h in heads] + [lk.add_out(l, code) for l in sorted({l for l, _ in heads})]
            res["oracle_H"][f"top{k}_{d}"] = score(lk, lk.fwd(T(ev[d]), hooks), ev[d])
        print(f"[lookup] 3 oracle, top-{k} retrieval heads write A's code: " + " ".join(f"{d} succ {res['oracle_H'][f'top{k}_{d}']['successor']:.2f}/look {res['oracle_H'][f'top{k}_{d}']['lookup']:.2f}" for d in ORD), flush=True)

    # ---- 4 weight edit: W_O[H] <- gamma * W_O[A] C_H, C_H fitted on ordinary-word lookups only
    Zf = z_of(cal_fit, HS); Zv = z_of(cal_val, HS)
    Vf = lk.carrier_value([c[1] for c in cal_fit]).double(); Vv = lk.carrier_value([c[1] for c in cal_val]).double()

    def ridge(X, Y, lam):
        G = X.T @ X; reg = lam * torch.trace(G) / G.shape[0]
        return torch.linalg.solve(G + reg * torch.eye(G.shape[0], dtype=X.dtype), X.T @ Y).T

    C = {}; res["interface_fit"] = {}
    for hd in HS:
        best = None
        for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
            c = ridge(Zf[hd].double(), Vf, lam); pred = Zv[hd].double() @ c.T
            r2 = 1 - float(((pred - Vv) ** 2).sum() / ((Vv - Vf.mean(0)) ** 2).sum())
            if best is None or r2 > best[0]:
                best = (r2, lam, c)
        C[hd] = best[2].float(); res["interface_fit"][f"L{hd[0]}h{hd[1]}"] = dict(heldout_r2=best[0], lam=best[1])
        print(f"[lookup] 4 interface map for L{hd[0]}h{hd[1]}: held-out R^2 {best[0]:.2f} (lambda {best[1]}; selected on ordinary-word lookups)", flush=True)
    wiki = json.loads(Path("results/evolution/splits.json").read_text())["wiki_dev"]["items"]; nll0 = backend.perplexity_nll(lk.model, wiki)
    orig = {}

    def apply(heads, gamma, cmap, wo=WOA, add=False):
        for l, h in heads:
            name = f"model.layers.{l}.self_attn.o_proj.weight"
            if name not in orig:
                orig[name] = lk.P[name].detach().clone()
            with torch.no_grad():
                new = gamma * (wo @ cmap[(l, h)]).to(lk.P[name].dtype)
                lk.P[name][:, lk.hsl(h)] = new + (lk.P[name][:, lk.hsl(h)] if add else 0)

    def restore():
        with torch.no_grad():
            for name, w in orig.items():
                lk.P[name].copy_(w)

    def measure():
        out = {d: score(lk, lk.fwd(T(ev[d])), ev[d]) for d in ORD}
        out["words_lookup"] = score(lk, lk.fwd(T(cal_val)), cal_val)["lookup"]; out["wiki_nll"] = backend.perplexity_nll(lk.model, wiki)
        return out

    res["edit"] = {}; res["wiki_nll_base"] = nll0
    for k in (1, 2, 4):
        for gamma in (1.0, 2.0):
            for add in (False, True):
                apply(HS[:k], gamma, C, add=add); m = measure(); restore()
                name = f"top{k}_gamma{gamma}_{'add' if add else 'replace'}"; res["edit"][name] = m
                print(f"[lookup] 4 EDIT {name}: " + " ".join(f"{d} succ {m[d]['successor']:.2f}/look {m[d]['lookup']:.2f}" for d in ORD)
                      + f" | word lookup {m['words_lookup']:.2f} | WikiText loss {m['wiki_nll']:.3f} (base {nll0:.3f})", flush=True)
    # choose the configuration WITHOUT ordinal data is not possible here (no generic composed task), so all pre-declared
    # configurations are reported above; controls use the configuration with the best mean successor rate, stated as post hoc.
    key = max(res["edit"], key=lambda n: sum(res["edit"][n][d]["successor"] for d in ORD)); res["posthoc_best"] = key
    k = int(key.split("_")[0][3:]); gamma = float(key.split("_")[1][5:]); add = key.endswith("add"); heads = HS[:k]
    print(f"[lookup] 4 controls use {key} (post hoc choice; all configurations are reported)", flush=True)
    perm = list(range(len(cal_fit))); random.Random(1).shuffle(perm); perm = torch.tensor(perm)
    Csh = {hd: ridge(Zf[hd].double(), Vf[perm], res["interface_fit"][f"L{hd[0]}h{hd[1]}"]["lam"]).float() for hd in heads}
    apply(heads, gamma, Csh, add=add); res["control_shuffled_pairing"] = measure(); restore()
    other = lk.P[f"model.layers.{A[0]}.self_attn.o_proj.weight"][:, lk.hsl((A[1] + 5) % lk.H)].float()
    apply(heads, gamma, C, wo=other, add=add); res["control_wrong_output_projection"] = measure(); restore()
    Wp = WOA[:, torch.tensor(random.Random(2).sample(range(HD), HD))]
    apply(heads, gamma, C, wo=Wp, add=add); res["control_permuted_A_columns"] = measure(); restore()
    for n in ("control_shuffled_pairing", "control_wrong_output_projection", "control_permuted_A_columns"):
        print(f"[lookup] 4 {n}: " + " ".join(f"{d} succ {res[n][d]['successor']:.2f}/look {res[n][d]['lookup']:.2f}" for d in ORD), flush=True)
    # swapped retrieval: with the edit in place, H's retrieved-value coordinates come from a different prompt
    apply(heads, gamma, C, add=add); res["swap"] = {}
    for d in ORD:
        cases = ev[d]; n = len(cases); sh = [(i + 37) % n for i in range(n)]; zs = z_of(cases, heads)
        lg = lk.fwd(T(cases), [lk.set_z(l, h, zs[(l, h)][torch.tensor(sh)]) for l, h in heads]); top = lg.argmax(-1).tolist()
        don = sum(cases[j][2] is not None and t == lk.first(cases[j][2]) for t, j in zip(top, sh)); own = sum(t == lk.first(c[2]) for t, c in zip(top, cases))
        res["swap"][d] = dict(donor_successor=round(don / n, 3), own_successor=round(own / n, 3))
    restore()
    print("[lookup] 4 swapped retrieval (H's coordinates from another prompt): " + " ".join(f"{d} donor-succ {v['donor_successor']:.2f}/own-succ {v['own_successor']:.2f}" for d, v in res["swap"].items()), flush=True)
    (OUT / "lookup.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[lookup] DONE ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
