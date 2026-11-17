"""Confirmation of the lookup -> successor weight edit with everything frozen (no fitting, no selection here).

Frozen before this script was written: the attention programme saved by lookup3.py (lambda chosen on held-out
ordinary-word lookups) and output gain 3.0 (gain was chosen post hoc in lookup3.py on its evaluation prompts,
which is why this confirmation uses new prompts).

New here: fresh prompts (new seed), key words never seen in fitting, two prompt formats never seen in fitting
("k: v", "k = v"), paired win/loss counts against the unedited model, and two causal checks with the edit in
place: removing A's output, and swapping A's output with the one from a prompt that queries a different value.
"""
import json
import random
import time

import torch

from .bridge import A, HD, OUT
from . import lookup, lookup2
from .lookup import LK, ORD, score

GAIN = 3.0
NEW_KEYS = ("lion zebra panda otter falcon salmon turtle rabbit monkey spider hammer ladder bucket pillow mirror "
            "basket carpet helmet jacket ribbon planet comet meteor galaxy rocket island valley desert forest harbor").split()


def main():
    lk = LK(); tok = lk.tok; P = lk.P; t0 = time.time(); res = dict(gain=GAIN)
    prog = torch.load(OUT / "lookup3_attention_programme.pt"); low = torch.tensor(prog["low"])
    pre = f"model.layers.{A[0]}.self_attn."; qn, kn, on = pre + "q_proj.weight", pre + "k_proj.weight", pre + "o_proj.weight"
    orig = {n: P[n].detach().clone() for n in (qn, kn, on)}; qa = lk.hsl(A[1]); ka = lk.lab.kvsl(A[1])

    def install():
        with torch.no_grad():
            q = torch.zeros(HD, P[qn].shape[1], dtype=P[qn].dtype); k = torch.zeros(HD, P[kn].shape[1], dtype=P[kn].dtype)
            q[low] = prog["WQ"].to(q); k[low] = prog["WK"].to(k)
            P[qn][qa] = q; P[kn][ka] = k; P[on][:, qa] = GAIN * orig[on][:, qa]

    def restore():
        with torch.no_grad():
            for n, w in orig.items():
                P[n].copy_(w)

    ordv = {d: [x for x in ORD[d] if len(tok(" " + x, add_special_tokens=False)["input_ids"]) == 1] for d in ORD}
    ordv["letters"] = list("bcdefghijklmnopqrstuvwx")
    old_keys = lookup.KEYS
    sets = {}
    for label, keys, fmt in (("arrow_new_keys", NEW_KEYS, "arrow"), ("colon_new_keys", NEW_KEYS, "colon"), ("equals_new_keys", NEW_KEYS, "equals"), ("arrow_old_keys", old_keys, "arrow")):
        lookup2.KEYS = keys; rr = random.Random("confirm-v1|" + label)
        sets[label] = {d: [lookup2.make(rr, ordv[d], d, fmt) for _ in range(400)] for d in ORD}
    lookup2.KEYS = old_keys
    T = lambda cases: [c["text"] for c in cases]

    def tops(cases, hooks=()):
        return lk.fwd(T(cases), list(hooks)).argmax(-1).tolist()

    for label, ev in sets.items():
        res[label] = {}
        for d in ORD:
            cases = ev[d]; s_id = [lk.first(c["s"]) for c in cases]; v_id = [lk.first(c["v"]) for c in cases]
            base = tops(cases); install(); edit = tops(cases)
            zs = []; lk.fwd(T(cases), [lk.grab_z(A[0], A[1], zs)]); z = torch.cat(zs)
            noA = tops(cases, [lk.set_z(A[0], A[1])])
            n = len(cases); sh = [(i + 97) % n for i in range(n)]
            swap = tops(cases, [lk.set_z(A[0], A[1], z[torch.tensor(sh)])]); restore()
            b = [t == s for t, s in zip(base, s_id)]; e = [t == s for t, s in zip(edit, s_id)]
            res[label][d] = dict(n=n, native_successor=sum(b) / n, edited_successor=sum(e) / n, native_lookup=sum(t == v for t, v in zip(base, v_id)) / n,
                                 edited_lookup=sum(t == v for t, v in zip(edit, v_id)) / n, wins=sum(y and not x for x, y in zip(b, e)), losses=sum(x and not y for x, y in zip(b, e)),
                                 edited_A_removed_successor=sum(t == s for t, s in zip(noA, s_id)) / n,
                                 swap_donor_successor=sum(t == s_id[j] for t, j in zip(swap, sh)) / n, swap_own_successor=sum(t == s for t, s in zip(swap, s_id)) / n)
            r = res[label][d]
            print(f"[confirm] {label:16s} {d:9s}: successor {r['native_successor']:.2f} -> {r['edited_successor']:.2f} (wins {r['wins']}, losses {r['losses']}, n {n}); "
                  f"lookup {r['native_lookup']:.2f} -> {r['edited_lookup']:.2f}; A removed {r['edited_A_removed_successor']:.2f}; "
                  f"swap: donor's successor {r['swap_donor_successor']:.2f}, own {r['swap_own_successor']:.2f}", flush=True)
    (OUT / "lookup3_confirm.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[confirm] DONE ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
