"""Composition by a closed-form bridge between two successor heads (no gradients, no API).

Gate result (gate.py): head A = L21h6 computes "next item" from whatever item its value vector encodes;
routing alone never yields two steps, because A's output is not in the format a later head reads.

Here: fit a linear bridge T from A's output coordinates to the value vector that a later successor
head B would have seen if the current item were the NEXT item. T is calibrated ONLY on non-letter
sequences (digits, number words, weekdays, months, ordinals). All letters are held out.

 Stage D (diagnostic, activations): during the forward pass, B's output at the final position gets
          gamma * T z_A added, where z_A is A's own output in the same pass. No donor prompt.
 Stage W (weights): the same bridge written into B's value weights, reading A's output from the residual:
          W_V[B] += gamma * T_w * pinv(g * W_O[A]). Persistent; no hooks at test time.

Reported: full-vocabulary top-1 rates for "next" and "next-next" on letters (never used for fitting),
controls (bridge with shuffled targets, bridge with A removed, mean-vector bridge), and collateral
change in WikiText loss for the weight edit.

    EVO_MODEL=Qwen/Qwen3-0.6B-Base python -m experiments.compose.bridge
"""
import json
import os
import string
import time
from pathlib import Path

import torch

from ..evolution import backend
from .gate import AB, HD, XS, DEV, RES

OUT = Path(os.environ.get("WSP_OUT", "results/compose"))
A = tuple(int(x) for x in os.environ.get("WSP_HEAD", "21,6").split(","))      # the successor head (find_head.py)
BS = [(22, 8), (23, 10), (24, 4), (25, 4)]          # later heads that showed a two-step effect under oracle value patches
DOMAINS = {
    "digits": [str(i) for i in range(1, 10)],
    "numwords": "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split(),
    "days": "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split(),
    "months": "January February March April May June July August September October November December".split(),
    "ordinals": "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth".split(),
}
LAY = {
    "bare": lambda w: " ".join(w),
    "seq": lambda w: "Sequence: " + " ".join(w[-3:]),
    "prose": lambda w: "Here are some items in order: " + " ".join(w),
    "list": lambda w: "List: " + " ".join(w[-2:]),
}


class Lab:
    def __init__(self):
        backend.set_threads()
        self.model, self.tok = backend.load_model(os.environ.get("EVO_MODEL", "Qwen/Qwen3-0.6B-Base"))
        self.layers = self.model.model.layers; self.P = dict(self.model.named_parameters())
        self.kvdiv = self.model.config.num_attention_heads // self.model.config.num_key_value_heads

    def first(self, item):
        return self.tok(" " + item, add_special_tokens=False)["input_ids"][0]

    def enc(self, text):
        return torch.tensor([self.tok(text, add_special_tokens=False)["input_ids"]])

    def sl(self, head):
        return slice(head * HD, (head + 1) * HD)

    def kvsl(self, head):
        k = head // self.kvdiv
        return slice(k * HD, (k + 1) * HD)

    @torch.inference_mode()
    def forward(self, text, grab=(), edit=None):
        """Returns final-position logits and a dict of grabbed activations at the final position.
        grab entries: ('z', layer, head) head output coordinates; ('v', layer, head) value vector;
        ('n', layer) normalised attention input. edit: function(store) -> list of (layer, head, vector to ADD to z)."""
        st = {}; hs = []
        for g in grab:
            if g[0] == "z":
                def pre(mod, args, g=g):
                    st[g] = args[0][0, -1, self.sl(g[2])].clone()
                hs.append(self.layers[g[1]].self_attn.o_proj.register_forward_pre_hook(pre))
            elif g[0] == "v":
                def hook(mod, inp, out, g=g):
                    st[g] = out[0, -1, self.kvsl(g[2])].clone()
                hs.append(self.layers[g[1]].self_attn.v_proj.register_forward_hook(hook))
            else:
                def pre2(mod, args, kwargs, g=g):
                    st[g] = (args[0] if args else kwargs["hidden_states"])[0, -1].clone()
                hs.append(self.layers[g[1]].self_attn.register_forward_pre_hook(pre2, with_kwargs=True))
        if edit is not None:
            for layer in sorted({l for l, _ in edit["targets"]}):
                def pre3(mod, args, layer=layer):
                    z = args[0].clone()
                    for (l, h), vec in edit["fn"](st).items():
                        if l == layer:
                            z[0, -1, self.sl(h)] += vec
                    return (z,)
                hs.append(self.layers[layer].self_attn.o_proj.register_forward_pre_hook(pre3))
        try:
            lg = self.model(input_ids=self.enc(text), use_cache=False).logits[0, -1].float()
        finally:
            for h in hs:
                h.remove()
        return lg, st


def contexts(items, cyclic=False):
    """(window ending at item i, index i) for every i that has two successors and at least two predecessors."""
    n = len(items); out = []
    for i in range(2, n if cyclic else n - 2):
        w = [items[(i - k) % n] for k in (3, 2, 1, 0)] if (cyclic or i >= 3) else [items[i - 2], items[i - 1], items[i]]
        out.append((w, items[(i + 1) % n], items[(i + 2) % n]))
    return out


def evaluate(lab, cases, edit=None, remove_a=False):
    """cases: list of (text, next item, next-next item). Returns top-1 rates and mean margin(next-next minus next).
    remove_a: A's own write to the residual is removed (the bridge, if any, still reads A's real output)."""
    n1 = n2 = 0; m = 0.0
    for text, a, b in cases:
        e = edit
        if remove_a:
            f = edit["fn"] if edit else (lambda st: {})
            e = dict(targets=list((edit or {}).get("targets", [])) + [A], fn=lambda st, f=f: {**f(st), A: -st[("z",) + A]})
        lg, _ = lab.forward(text, grab=[("z",) + A], edit=e)
        ta, tb = lab.first(a), lab.first(b); top = int(lg.argmax())
        n1 += top == ta; n2 += top == tb; m += float(lg[tb] - lg[ta])
    k = len(cases)
    return dict(next=n1 / max(k, 1), plus2=n2 / max(k, 1), margin=m / max(k, 1), n=k)


def interchange(lab, T, heads, gamma, seed="bridge-interchange"):
    """The bridge input is taken from a context ending in an unrelated letter y (A's own write is untouched).
    If the bridge carries A's intermediate result, the answer should move to y+2."""
    import random
    rr = random.Random(seed); hit = keep2 = keep1 = n = 0
    for x in XS:
        for lay in LAY:
            y = rr.choice([c for c in XS if abs(AB.index(c) - AB.index(x)) > 3])
            win = lambda c: LAY[lay]([AB[AB.index(c) - 3 + k] for k in range(4)])
            _, sd = lab.forward(win(y), grab=[("z",) + A]); zd = sd[("z",) + A]
            lg, _ = lab.forward(win(x), grab=[("z",) + A], edit=dict(targets=list(heads), fn=lambda st: {b: gamma * (T[b] @ zd) for b in heads}))
            top = int(lg.argmax()); n += 1
            hit += top == lab.first(AB[AB.index(y) + 2]); keep2 += top == lab.first(AB[AB.index(x) + 2]); keep1 += top == lab.first(AB[AB.index(x) + 1])
    return dict(donor_plus2=hit / n, own_plus2=keep2 / n, own_next=keep1 / n, n=n)


def main():
    OUT.mkdir(parents=True, exist_ok=True); lab = Lab(); t0 = time.time(); res = {}
    # ---------------- cases
    cal = []       # (domain, text_i, text_i+1, next, nextnext)
    for dom, items in DOMAINS.items():
        cyc = dom in ("days", "months")
        for w, a, b in contexts(items, cyc):
            w2 = w[1:] + [a]
            for lay in LAY:
                cal.append((dom, LAY[lay](w), LAY[lay](w2), a, b))
    letters = {name: [(LAY[lay]([AB[AB.index(x) - 3 + k] for k in range(4)]), AB[AB.index(x) + 1], AB[AB.index(x) + 2]) for x in xs for lay in LAY]
               for name, xs in (("all", XS), ("gate_dev", DEV), ("gate_reserved", RES))}
    cal_cases = [(t, a, b) for _, t, _, a, b in cal if lab.first(a) != lab.first(b)]
    res["n_cal"] = len(cal); res["native_cal"] = evaluate(lab, cal_cases); res["native_letters"] = evaluate(lab, letters["all"])
    by_dom = {d: evaluate(lab, [(t, a, b) for dd, t, _, a, b in cal if dd == d and lab.first(a) != lab.first(b)]) for d in DOMAINS}
    res["native_cal_by_domain"] = by_dom
    print(f"[bridge] native: calibration domains next {res['native_cal']['next']:.2f} +2 {res['native_cal']['plus2']:.2f} (n={res['native_cal']['n']}); "
          f"letters next {res['native_letters']['next']:.2f} +2 {res['native_letters']['plus2']:.2f}", flush=True)
    print("[bridge] native by domain: " + ", ".join(f"{d} {s['next']:.2f}" for d, s in by_dom.items()), flush=True)
    good_doms = [d for d, s in by_dom.items() if s["next"] >= 0.8]; res["calibration_domains_used"] = good_doms
    cal = [c for c in cal if c[0] in good_doms]; cal_cases = [(t, a, b) for _, t, _, a, b in cal if lab.first(a) != lab.first(b)]
    print(f"[bridge] calibration uses domains with native next-item accuracy >= .80: {good_doms} ({len(cal)} contexts, no letters)", flush=True)

    # ---------------- calibration activations
    X = []; Y = {b: [] for b in BS}; N = {b: [] for b in BS}
    for _, t1, t2, _, _ in cal:
        _, s1 = lab.forward(t1, grab=[("z",) + A] + [("n", l) for l, _ in BS])
        _, s2 = lab.forward(t2, grab=[("v",) + b for b in BS])
        X.append(s1[("z",) + A])
        for b in BS:
            Y[b].append(s2[("v",) + b]); N[b].append(s1[("n", b[0])])
    X = torch.stack(X).double()

    def ridge(Xm, Ym, lam):
        G = Xm.T @ Xm; reg = lam * torch.trace(G) / G.shape[0]
        return (torch.linalg.solve(G + reg * torch.eye(G.shape[0], dtype=Xm.dtype), Xm.T @ Ym)).T      # Y ~ T X

    def bridges(lam, perm=None):
        out = {}
        for b in BS:
            Yb = torch.stack(Y[b]).double()
            out[b] = ridge(X, Yb if perm is None else Yb[perm], lam).float()
        return out

    def edit_for(T, heads, gamma, mean_only=False):
        mx = X.mean(0).float()
        return dict(targets=list(heads), fn=lambda st: {b: gamma * (T[b] @ (mx if mean_only else st[("z",) + A])) for b in heads})

    # ---------------- stage D: choose lambda, gamma, head set on the CALIBRATION domains' own two-step task
    grid = []
    for lam in (1e-3, 1e-2, 1e-1):
        T = bridges(lam)
        for heads in ([BS[0]], BS[:2], BS):
            for gamma in (1.0, 2.0, 4.0, 8.0):
                s = evaluate(lab, cal_cases, edit_for(T, heads, gamma))
                grid.append(dict(lam=lam, heads=[f"L{l}h{h}" for l, h in heads], gamma=gamma, cal=s, _T=T, _heads=heads))
    grid.sort(key=lambda g: (-g["cal"]["plus2"], -g["cal"]["margin"])); best = grid[0]
    res["stageD_grid"] = [{k: v for k, v in g.items() if not k.startswith("_")} for g in grid]
    print(f"[bridge] D: best on calibration domains: lambda {best['lam']} heads {best['heads']} gamma {best['gamma']} -> +2 {best['cal']['plus2']:.2f} next {best['cal']['next']:.2f}", flush=True)
    T, heads, gamma = best["_T"], best["_heads"], best["gamma"]
    res["stageD_letters"] = {k: evaluate(lab, v, edit_for(T, heads, gamma)) for k, v in letters.items()}
    print(f"[bridge] D: LETTERS (never used in fitting or selection): +2 {res['stageD_letters']['all']['plus2']:.2f} next {res['stageD_letters']['all']['next']:.2f} "
          f"margin {res['stageD_letters']['all']['margin']:+.2f} (native +2 {res['native_letters']['plus2']:.2f}, margin {res['native_letters']['margin']:+.2f})", flush=True)
    res["stageD_letters_gamma_sweep"] = {g: evaluate(lab, letters["all"], edit_for(T, heads, g)) for g in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)}
    print("[bridge] D: letters by gamma (+2 / next): " + " ".join(f"{g}:{s['plus2']:.2f}/{s['next']:.2f}" for g, s in res["stageD_letters_gamma_sweep"].items()), flush=True)
    # controls
    import random
    pl = list(range(len(cal))); random.Random(0).shuffle(pl); perm = torch.tensor(pl)
    Tsh = bridges(best["lam"], perm)
    res["control_shuffled_targets"] = evaluate(lab, letters["all"], edit_for(Tsh, heads, gamma))
    res["control_mean_vector"] = evaluate(lab, letters["all"], edit_for(T, heads, gamma, mean_only=True))
    res["bridge_with_A_direct_write_removed"] = evaluate(lab, letters["all"], edit_for(T, heads, gamma), remove_a=True)
    res["A_removed_no_bridge"] = evaluate(lab, letters["all"], None, remove_a=True)
    for k in ("control_shuffled_targets", "control_mean_vector", "bridge_with_A_direct_write_removed", "A_removed_no_bridge"):
        print(f"[bridge] D: {k}: +2 {res[k]['plus2']:.2f} next {res[k]['next']:.2f} margin {res[k]['margin']:+.2f}", flush=True)
    res["interchange"] = interchange(lab, T, heads, gamma)
    print(f"[bridge] D: interchange (bridge fed A's output from an unrelated letter y): answer y+2 {res['interchange']['donor_plus2']:.2f}, "
          f"own +2 {res['interchange']['own_plus2']:.2f}, own next {res['interchange']['own_next']:.2f}", flush=True)

    # ---------------- stage W: write the bridge into value weights (reads A's output from the residual)
    WO = lab.P[f"model.layers.{A[0]}.self_attn.o_proj.weight"].double()[:, lab.sl(A[1])]            # 1024 x 128
    orig = {}; wiki = json.loads(Path("results/evolution/splits.json").read_text())["wiki_dev"]["items"]
    nll0 = backend.perplexity_nll(lab.model, wiki)
    resW = {}
    for lam in (1e-2, 1e-1):
        Tw = {}
        for b in heads:
            gain = lab.P[f"model.layers.{b[0]}.input_layernorm.weight"].double()
            R = torch.linalg.pinv(gain[:, None] * WO)                                                 # 128 x 1024: reads A's coordinates from the normalised input
            F = torch.stack(N[b]).double() @ R.T                                                      # calibration features as the weights will see them
            Tw[b] = (ridge(F, torch.stack(Y[b]).double(), lam) @ R).float()                           # 128 x 1024 update for B's value rows
        for gm in (1.0, 2.0, 4.0, 8.0):
            for b in heads:
                name = f"model.layers.{b[0]}.self_attn.v_proj.weight"
                if name not in orig:
                    orig[name] = lab.P[name].detach().clone()
                with torch.no_grad():
                    lab.P[name][lab.kvsl(b[1])] += gm * Tw[b].to(lab.P[name].dtype)
            s = dict(cal=evaluate(lab, cal_cases), letters=evaluate(lab, letters["all"]), letters_reserved=evaluate(lab, letters["gate_reserved"]),
                     wiki_nll=backend.perplexity_nll(lab.model, wiki))
            with torch.no_grad():
                for name, w in orig.items():
                    lab.P[name].copy_(w)
            resW[f"lam{lam}_gamma{gm}"] = s
            print(f"[bridge] W: lambda {lam} gamma {gm}: calibration +2 {s['cal']['plus2']:.2f}; LETTERS +2 {s['letters']['plus2']:.2f} next {s['letters']['next']:.2f} "
                  f"margin {s['letters']['margin']:+.2f}; WikiText loss {s['wiki_nll']:.4f} (base {nll0:.4f})", flush=True)
    res["stageW"] = resW; res["wiki_nll_base"] = nll0
    res["chosen"] = dict(lam=best["lam"], heads=best["heads"], gamma=gamma)
    (OUT / "bridge.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[bridge] DONE ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
