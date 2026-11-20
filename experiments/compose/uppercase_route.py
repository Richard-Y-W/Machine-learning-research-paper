"""Second operation, stage 2/3: lookup -> uppercase via a group of heads (declared before any result).

second_op2.py (Qwen3-0.6B): for 'a -> A' few-shot prompts, replacing the outputs of the top-8 attribution heads with
their own value vectors from a donor item makes the model answer the donor's uppercase letter on 0.68 of prompts
(single heads and groups of 2/4: 0.00). That oracle ran in a context that SIGNALS the task. A routing edit would
have to install uppercasing as the default answer to a plain lookup, so the decisive oracle is in plain lookups.

DECLARED PROTOCOL
  K   the top-8 heads from second_op2.json ('upper_letter' -> verified order). No reselection.
  A0  native, plain lookup tables with lowercase letter values (KEYS, arrow): rates of lowercase (lookup) and uppercase.
  A1  oracle in plain lookups: for every head in K, final-position output := gamma * its value vector at the QUERIED
      VALUE's position (what perfect routing would deliver), gamma in {1, 2, 3}. Also the same for 8 random other
      heads of the same layers (control). Gate: proceed to B only if some gamma gives uppercase >= 0.30
      (lower than the few-shot gate because no task cue is present; stated before running).
  B   (only if gated) routing fit per layer (lookup3 closed form on 1500 ordinary-word lookups, low RoPE band, ridge by
      held-out attention), installed in every head of K, common output gain chosen in {1,2,3} on development letter
      lookups (KEYS, arrow). Confirmation on TEST_KEYS, three formats, 120 prompts each; controls: programme in 8
      random other heads of the same layers; programme in K with output gain 0 (routing without the OV).
      Collateral: WikiText-103 validation (64k tokens) and native sequences, as in robustness.py.

    EVO_MODEL=Qwen/Qwen3-0.6B-Base XF_OUT=results/uppercase_0p6b python -m experiments.compose.uppercase_route
"""
import json
import os
import random
import time
from pathlib import Path

import torch

from . import lookup2
from .lookup import KEYS
from .robustness import TEST_KEYS, word_pool
from .xfamily import XO, Arch

SRC = Path(os.environ.get("XF_SRC", "results/second_op2_0p6b/second_op2.json"))
LOW = list("bcdefghijklmnopqrstuvwxy"); FMTS = ("arrow", "colon", "equals"); N = int(os.environ.get("XF_N", "120"))


def main():
    XO.mkdir(parents=True, exist_ok=True); ar = Arch(os.environ.get("EVO_MODEL", "Qwen/Qwen3-0.6B-Base")); t0 = time.time()
    K = [tuple(x[k] for k in ("layer", "head")) for x in json.loads(SRC.read_text())["upper_letter"]["verified"][:8]]
    resf = XO / "uppercase.json"; res = json.loads(resf.read_text()) if resf.exists() else {}; res["heads"] = K
    save = lambda: resf.write_text(json.dumps(res, indent=1), encoding="utf-8")
    up = lambda c: c["v"].upper()
    lk = [c for c in LOW if len(ar.ids(" " + c)) == 1 and len(ar.ids(" " + c.upper())) == 1]
    rr = random.Random("uppercase-v1"); lookup2.KEYS = KEYS
    dev = [lookup2.make(rr, lk, None, "arrow") for _ in range(200)]
    T = lambda cs: [c["text"] for c in cs]
    pos_in_batch = lambda cs: (lambda Tn: torch.tensor([Tn - len(ar.ids(c["text"])) + len(ar.ids(c["prefix"])) - 1 for c in cs]))(max(len(ar.ids(c["text"])) for c in cs))

    def rates(cs, hooks=()):
        top = ar.last(T(cs), list(hooks), bs=len(cs)).argmax(-1).tolist()
        return dict(upper=sum(t == ar.first(up(c)) for t, c in zip(top, cs)) / len(cs), lower=sum(t == ar.first(c["v"]) for t, c in zip(top, cs)) / len(cs))

    def value_at(cs, l, h):
        """Head h's value vector at each prompt's queried-value position (one batch)."""
        st = []; a = ar.attn(l); w, b, rs = ar.rows(l, h, "v"); p = pos_in_batch(cs)
        lin = a.query_key_value if ar.neox else a.v_proj
        hk = lin.register_forward_hook(lambda m, i, o: st.append(o[torch.arange(len(cs)), p][:, rs].clone()))
        try:
            ar.last(T(cs), bs=len(cs))
        finally:
            hk.remove()
        return st[0]

    # ---- A0 native
    if "A0" not in res:
        res["A0"] = rates(dev); save()
    print(f"[upper] A0 native plain lookups (letters): lowercase {res['A0']['lower']:.2f} uppercase {res['A0']['upper']:.2f}", flush=True)

    # ---- A1 oracle in plain lookups (batches of 50 so value positions align with one padded batch)
    if "A1" not in res:
        others = random.Random("uppercase-others").sample([(l, h) for l in sorted({l for l, _ in K}) for h in range(ar.H) if (l, h) not in K], 8)
        res["A1"] = {"control_heads": others}
        for name, heads in (("K", K), ("others", others)):
            for g in (1.0, 2.0, 3.0):
                agg = dict(upper=0.0, lower=0.0)
                for i in range(0, len(dev), 50):
                    cs = dev[i:i + 50]; vs = {hd: value_at(cs, *hd) for hd in heads}
                    r = rates(cs, [ar.zhook(l, h, g * vs[(l, h)]) for l, h in heads])
                    for k in agg:
                        agg[k] += r[k] * len(cs) / len(dev)
                res["A1"][f"{name}_g{g}"] = agg; save()
                print(f"[upper] A1 oracle in plain lookups, {name} heads, gamma {g}: uppercase {agg['upper']:.2f} lowercase {agg['lower']:.2f}", flush=True)
    best = max(v["upper"] for k, v in res["A1"].items() if k.startswith("K_"))
    res["gate_passed"] = best >= 0.30; save()
    if not res["gate_passed"]:
        print(f"[upper] gate failed (best oracle uppercase {best:.2f} < 0.30): the head group does not uppercase a routed lookup value without a task cue. STOP.", flush=True)
        print(f"[upper] DONE ({time.time() - t0:.0f}s)", flush=True); return
    print(f"[upper] gate passed (best {best:.2f}); stage B", flush=True)

    # ---- B routing fit per layer, install in K
    words = word_pool(ar.tok); rf = random.Random("uppercase-fit-v1")
    fit_cases = [lookup2.make(rf, words, None, "arrow") for _ in range(1500)]; wval = [lookup2.make(rf, words, None, "arrow") for _ in range(100)]
    pos = lambda c: len(ar.ids(c["prefix"])) - 1; progs = {}
    for l in sorted({l for l, _ in K}):
        d = ar.D; Cff = torch.zeros(d, d, dtype=torch.float64); Ckk = torch.zeros(d, d, dtype=torch.float64); X = torch.zeros(d, d, dtype=torch.float64); nk = 0
        for c in fit_cases:
            n = ar.layer_input(l, c["text"]); p = pos(c); f = n[-1]; neg = torch.cat([n[:p], n[p + 1:]]).mean(0)
            Cff += torch.outer(f, f); Ckk += n.T @ n; nk += n.shape[0]; X += torch.outer(f, n[p] - neg)
        Cff /= len(fit_cases); Ckk /= nk; X /= len(fit_cases); I = torch.eye(d, dtype=torch.float64); r = len(ar.band); best = None
        h0 = next(h for ll, h in K if ll == l)
        for lam in (1e-2, 1e-1, 1.0):
            B = torch.linalg.solve(Cff + lam * torch.trace(Cff) / d * I, X) @ torch.linalg.inv(Ckk + lam * torch.trace(Ckk) / d * I)
            U, S, Vh = torch.linalg.svd(B); wq, wk = (U[:, :r] * S[:r].sqrt()).T.float(), (S[:r].sqrt()[:, None] * Vh[:r]).float()
            ar.program(l, h0, wq, wk); m = sum(float(ar.attn_row(l, h0, c["text"])[pos(c)]) for c in wval) / len(wval); ar.restore()
            if best is None or m > best[0]:
                best = (m, lam, wq, wk)
        progs[l] = best[2:]; res.setdefault("fit", {})[str(l)] = dict(lam=best[1], heldout_attention=best[0]); save()
        print(f"[upper] B fit layer {l}: ridge {best[1]}, held-out attention {best[0]:.2f}", flush=True)

    def install(heads, g):
        for l, h in heads:
            ar.program(l, h, *progs[l]); ar.gain(l, h, g)
    dg = {}
    for g in (1.0, 2.0, 3.0):
        install(K, g); dg[str(g)] = rates(dev); ar.restore()
    G = max((1.0, 2.0, 3.0), key=lambda x: dg[str(x)]["upper"]); res["gain"] = dict(chosen=G, dev=dg); save()
    print(f"[upper] B gain on development letter lookups: {dg} -> {G}", flush=True)
    lookup2.KEYS = TEST_KEYS; res["confirm"] = {}; others = [tuple(x) for x in res["A1"]["control_heads"]]
    for fmt in FMTS:
        cs = [lookup2.make(random.Random(f"uppercase-conf|{fmt}|{i}"), lk, None, fmt) for i in range(N)]
        row = dict(native=rates(cs))
        install(K, G); row["edited"] = rates(cs); ar.restore()
        install(others, G); row["programme_in_other_heads"] = rates(cs); ar.restore()
        install(K, 0.0); row["routing_without_output"] = rates(cs); ar.restore()
        res["confirm"][fmt] = row; save()
        print(f"[upper] B confirm {fmt}: uppercase {row['native']['upper']:.2f} -> {row['edited']['upper']:.2f} (lowercase {row['native']['lower']:.2f} -> {row['edited']['lower']:.2f}); "
              f"other heads {row['programme_in_other_heads']['upper']:.2f}; routing without output {row['routing_without_output']['upper']:.2f}", flush=True)
    lookup2.KEYS = KEYS
    from datasets import load_dataset
    text = "\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")["text"] if t.strip())
    ids = ar.tok(text[:3_000_000])["input_ids"]; wiki = [ids[i * 256:(i + 1) * 256] for i in range(250)]
    un = sum(ar.chunk_nll(wiki)) / 250; install(K, G); ed = sum(ar.chunk_nll(wiki)) / 250; ar.restore()
    res["wiki_val_nll"] = dict(unedited=un, edited=ed); save()
    print(f"[upper] WikiText-val loss {un:.4f} -> {ed:.4f}", flush=True)
    print(f"[upper] DONE ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
