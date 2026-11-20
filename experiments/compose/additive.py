"""Replace vs add: does the closed form's damage come from overwriting the successor head's native query/key rows?
(declared before running; follow-up to learning_ind.py, where gradient training on the same routing labels and the
same parameters preserved native sequences, 0.62, while the closed form did not, 0.23)

Same archived programme (lookup3_attention_programme.pt) and the same independent sets as robustness.py.
  replace_g{g}   the published edit: A's query rows and key-group rows set to the programme in the low RoPE band, ZERO
                 elsewhere (already measured in robustness.json; recomputed here only for g=3 as a check)
  band_g{g}      programme written into the band dimensions only; the other 64 dimensions keep their native weights
  add_g{g}       programme ADDED to the native rows (band dimensions); native rows untouched otherwise
Output gain g in {2, 3}. Prediction if 'replace' is the cause: band/add keep native sequences near 0.59.

    EVO_MODEL=Qwen/Qwen3-0.6B-Base WSP_OUT=results/compose RB_OUT=results/robustness_0p6b python -m experiments.compose.additive
"""
import json
import time

import torch

from .bridge import A, HD, OUT
from .lookup3 import LOW
from .robustness import RB, Rig, build_sets, evaluate


def main():
    rig = Rig(); P = rig.P; S = build_sets(rig); t0 = time.time()
    resf = RB / "additive.json"; res = json.loads(resf.read_text()) if resf.exists() else {}
    prog = torch.load(OUT / "lookup3_attention_programme.pt"); low = torch.tensor(LOW)
    qn, kn = rig.name("q"), rig.name("k"); qa = rig.lk.hsl(A[1]); ka = rig.lk.lab.kvsl(A[1])

    def install(mode, g):
        with torch.no_grad():
            for n, sl, W in ((qn, qa, prog["WQ"]), (kn, ka, prog["WK"])):
                rows = rig.orig[n][sl].clone()
                if mode == "replace":
                    rows = torch.zeros_like(rows); rows[low] = W.to(rows.dtype)
                elif mode == "band":
                    rows[low] = W.to(rows.dtype)
                else:
                    rows[low] = rows[low] + W.to(rows.dtype)
                P[n][sl] = rows
        rig.scale(A[1], g)

    for mode, g in (("replace", 3.0), ("band", 2.0), ("band", 3.0), ("add", 2.0), ("add", 3.0)):
        name = f"{mode}_g{g}"
        if name in res:
            continue
        t1 = time.time()
        try:
            install(mode, g); r = evaluate(rig, S)
        finally:
            rig.restore()
        res[name] = r; resf.write_text(json.dumps(res), encoding="utf-8"); s = r["summary"]
        print(f"[add] {name}: successor {s['successor']:.3f} | seq {s['seq']:.3f} | word lookup {s['word_lookup']:.3f} | WikiText-val {s['wiki_nll']:.4f} ({time.time() - t1:.0f}s)", flush=True)
    print(f"[add] DONE ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
