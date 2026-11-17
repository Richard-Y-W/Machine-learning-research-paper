"""Diagnostics after the first bridge attempt failed (bridge.py). Forward passes only.

 1  oracle on the calibration domains: replace A's (and B's) output with the value from the next context
 2  bridge as REPLACEMENT of B's output (bridge.py added it on top)
 3  shift operator on A: fit S with z_A(context i) -> z_A(context i+1) on non-letter domains, replace
    A's output by S z_A; test on letters. (A replacement map in A's output space, not a composition.)
 4  same, calibrated on the gate's development letters and tested on reserved letters
 5  what the model outputs when a two-step attempt fails (offset histogram)
"""
import json
import os
from pathlib import Path

import torch

from .bridge import A, BS, DOMAINS, LAY, Lab, contexts, OUT
from .gate import AB, XS, DEV, RES

B = BS[0]


def run(lab, text, repl=None):
    """repl: dict head -> function(store) giving the vector that REPLACES that head's output at the final position."""
    edit = None
    if repl:
        edit = dict(targets=list(repl), fn=lambda st: {h: f(st) - st[("z",) + h] for h, f in repl.items() if ("z",) + h in st})
    lg, st = lab.forward(text, grab=[("z",) + A, ("z",) + B, ("v",) + A, ("v",) + B], edit=edit)
    return lg, st


def rates(lab, cases, repl_fn=None, seq=None):
    """cases: (text, text_next, next, nextnext[, ordered items, index]). Returns top-1 rates and offset histogram."""
    n1 = n2 = 0; hist = {}
    for c in cases:
        text, text2, a, b = c[:4]
        repl = repl_fn(c) if repl_fn else None
        lg, _ = run(lab, text, repl); top = int(lg.argmax())
        n1 += top == lab.first(a); n2 += top == lab.first(b)
        if len(c) > 4:
            items, i = c[4], c[5]
            off = next((k for k in range(-2, 5) if 0 <= i + k < len(items) and lab.first(items[i + k]) == top), "other")
            hist[off] = hist.get(off, 0) + 1
    k = len(cases)
    return dict(next=round(n1 / k, 3), plus2=round(n2 / k, 3), n=k, offsets={str(o): v for o, v in sorted(hist.items(), key=lambda kv: str(kv[0]))})


def main():
    lab = Lab(); res = {}
    cal = []
    for dom, items in DOMAINS.items():
        if dom == "digits":
            continue
        cyc = dom in ("days", "months")
        for w, a, b in contexts(items, cyc):
            if lab.first(a) == lab.first(b):
                continue
            for lay in LAY:
                cal.append((LAY[lay](w), LAY[lay](w[1:] + [a]), a, b))

    def letter_cases(xs):
        out = []
        for x in xs:
            i = AB.index(x)
            for lay in LAY:
                out.append((LAY[lay](list(AB[i - 3:i + 1])), LAY[lay](list(AB[i - 2:i + 2])), AB[i + 1], AB[i + 2], list(AB), i))
        return out
    let, let_dev, let_res = letter_cases(XS), letter_cases(DEV), letter_cases(RES)

    def fixed(c, *specs):
        """Donor vectors are computed BEFORE the patched forward pass (specs: (head, kind) pairs)."""
        _, s2 = run(lab, c[1])
        return {head: (lambda st, v=s2[(kind,) + head]: v) for head, kind in specs}

    # ---- 1 oracle replacements
    for name, cases in (("calibration", cal), ("letters", let)):
        res[f"native_{name}"] = rates(lab, cases)
        res[f"oracle_A_value_{name}"] = rates(lab, cases, lambda c: fixed(c, (A, "v")))
        res[f"oracle_A_output_{name}"] = rates(lab, cases, lambda c: fixed(c, (A, "z")))
        res[f"oracle_B_value_{name}"] = rates(lab, cases, lambda c: fixed(c, (B, "v")))
        res[f"oracle_B_output_{name}"] = rates(lab, cases, lambda c: fixed(c, (B, "z")))
        res[f"oracle_AB_output_{name}"] = rates(lab, cases, lambda c: fixed(c, (A, "z"), (B, "z")))
        for k in ("native", "oracle_A_value", "oracle_A_output", "oracle_B_value", "oracle_B_output", "oracle_AB_output"):
            r = res[f"{k}_{name}"]; print(f"[diag] 1 {name:11s} {k:18s} next {r['next']:.2f} +2 {r['plus2']:.2f} offsets {r['offsets']}", flush=True)

    # ---- activations for fitting
    def collect(cases):
        ZA, ZA2, VA2, ZB2, VB2 = [], [], [], [], []
        for c in cases:
            _, s1 = run(lab, c[0]); _, s2 = run(lab, c[1])
            ZA.append(s1[("z",) + A]); ZA2.append(s2[("z",) + A]); VA2.append(s2[("v",) + A]); ZB2.append(s2[("z",) + B]); VB2.append(s2[("v",) + B])
        return [torch.stack(t).double() for t in (ZA, ZA2, VA2, ZB2, VB2)]

    def ridge(X, Y, lam):
        G = X.T @ X; reg = lam * torch.trace(G) / G.shape[0]
        return torch.linalg.solve(G + reg * torch.eye(G.shape[0], dtype=X.dtype), X.T @ Y).T.float()

    ZA, ZA2, VA2, ZB2, VB2 = collect(cal)
    cos = torch.nn.functional.cosine_similarity
    for lam in (1e-3, 1e-2, 1e-1, 1.0):
        # ---- 2 bridge into B as replacement (targets: B's own output, or B's value, in the next context)
        for tname, Y in (("B_output", ZB2), ("B_value", VB2)):
            T = ridge(ZA, Y, lam)
            r1 = rates(lab, cal, lambda c: {B: lambda st: T @ st[("z",) + A]}); r2 = rates(lab, let, lambda c: {B: lambda st: T @ st[("z",) + A]})
            res[f"bridge_replace_{tname}_lam{lam}"] = dict(cal=r1, letters=r2)
            print(f"[diag] 2 replace B by T z_A (target {tname}, lambda {lam}): calibration +2 {r1['plus2']:.2f} next {r1['next']:.2f} | letters +2 {r2['plus2']:.2f} next {r2['next']:.2f} {r2['offsets']}", flush=True)
        # ---- 3 shift operator on A (targets: A's output in the next context)
        S = ridge(ZA, ZA2, lam)
        r1 = rates(lab, cal, lambda c: {A: lambda st: S @ st[("z",) + A]}); r2 = rates(lab, let, lambda c: {A: lambda st: S @ st[("z",) + A]})
        res[f"shift_A_lam{lam}"] = dict(cal=r1, letters=r2)
        print(f"[diag] 3 shift operator on A (lambda {lam}): calibration +2 {r1['plus2']:.2f} next {r1['next']:.2f} | LETTERS +2 {r2['plus2']:.2f} next {r2['next']:.2f} {r2['offsets']}", flush=True)
        # both: A shifted and B bridged
        T = ridge(ZA, ZB2, lam)
        r2 = rates(lab, let, lambda c: {A: lambda st: S @ st[("z",) + A], B: lambda st: T @ st[("z",) + A]})
        res[f"shift_A_plus_bridge_B_lam{lam}"] = dict(letters=r2)
        print(f"[diag] 3 shift on A + bridge into B (lambda {lam}): LETTERS +2 {r2['plus2']:.2f} next {r2['next']:.2f} {r2['offsets']}", flush=True)

    # ---- 4 letters-only calibration: development letters -> reserved letters
    ZAl, ZA2l, _, ZB2l, _ = collect(let_dev)
    for lam in (1e-2, 1e-1, 1.0):
        S = ridge(ZAl, ZA2l, lam); T = ridge(ZAl, ZB2l, lam)
        r = rates(lab, let_res, lambda c: {A: lambda st: S @ st[("z",) + A]})
        rb = rates(lab, let_res, lambda c: {A: lambda st: S @ st[("z",) + A], B: lambda st: T @ st[("z",) + A]})
        res[f"shift_A_letters_dev_to_reserved_lam{lam}"] = dict(shift=r, shift_plus_bridge=rb)
        print(f"[diag] 4 fit on development letters, test on RESERVED letters (lambda {lam}): shift +2 {r['plus2']:.2f} next {r['next']:.2f} {r['offsets']} | "
              f"shift+bridge +2 {rb['plus2']:.2f}", flush=True)
    # how similar are A's outputs for consecutive items, and across domains?
    res["cos_zA_consecutive_cal"] = float(cos(ZA, ZA2).mean()); res["cos_zA_consecutive_letters"] = float(cos(ZAl, ZA2l).mean())
    print(f"[diag] cosine(z_A(i), z_A(i+1)): calibration {res['cos_zA_consecutive_cal']:.2f}, letters {res['cos_zA_consecutive_letters']:.2f}", flush=True)
    (OUT / "diag.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("[diag] DONE", flush=True)


if __name__ == "__main__":
    main()
