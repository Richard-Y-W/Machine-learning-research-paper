"""Controls for the lookup -> successor edit (frozen programme, gain 3).

 1 host specificity: the same attention programme installed in OTHER heads of the layer, each keeping its own
   complete OV circuit (value rows and output projection), with the same output gain
 2 shared keys: head A shares its key rows with one neighbour; that neighbour is removed, natively and under the edit
 3 counterfactual prompts: adjacent ordinal values allowed; the correct successor present as a distractor;
   longer tables; donor swaps counted only when the donor's queried value differs
 4 structural null: same singular values, rank and RoPE placement, random singular directions
"""
import json
import random
import time

import torch

from .bridge import A, HD, OUT
from . import lookup2
from .lookup import KEYS, LK, ORD, succ
from .lookup3_confirm import GAIN, NEW_KEYS


def make_cf(rr, pool, dom, mode, keys):
    line, query = lookup2.FORMATS["arrow"]
    for _ in range(500):
        n = 8 if mode == "long_table" else rr.choice([3, 4, 5]); ks = rr.sample(keys, n); vals = rr.sample(pool, n); q = rr.randrange(n)
        s = succ(dom, vals[q])
        if s is None:
            continue
        if mode == "successor_present":
            if s not in pool:
                continue
            j = rr.choice([i for i in range(n) if i != q]); vals[j] = s
            if len(set(vals)) < n:
                continue
        elif s in vals:
            continue
        pre = "".join(line.format(k=k, v=v) for k, v in zip(ks[:q + 1], vals[:q + 1])).rstrip("\n")
        return dict(text="".join(line.format(k=k, v=v) for k, v in zip(ks, vals)) + query.format(k=ks[q]), prefix=pre, v=vals[q], s=s)
    raise RuntimeError("could not build a prompt")


def main():
    lk = LK(); tok = lk.tok; P = lk.P; t0 = time.time(); res = dict(gain=GAIN)
    try:
        lk.model.set_attn_implementation("eager")
    except Exception:
        lk.model.config._attn_implementation = "eager"
    prog = torch.load(OUT / "lookup3_attention_programme.pt"); low = torch.tensor(prog["low"])
    pre = f"model.layers.{A[0]}.self_attn."; qn, kn, on = pre + "q_proj.weight", pre + "k_proj.weight", pre + "o_proj.weight"
    orig = {n: P[n].detach().clone() for n in (qn, kn, on)}; d = P[qn].shape[1]

    def install(head, WQ, WK, gain=GAIN):
        with torch.no_grad():
            q = torch.zeros(HD, d, dtype=P[qn].dtype); k = torch.zeros(HD, d, dtype=P[kn].dtype)
            q[low] = WQ.to(q); k[low] = WK.to(k)
            P[qn][lk.hsl(head)] = q; P[kn][lk.lab.kvsl(head)] = k; P[on][:, lk.hsl(head)] = gain * orig[on][:, lk.hsl(head)]

    def restore():
        with torch.no_grad():
            for n, w in orig.items():
                P[n].copy_(w)

    ordv = {dm: [x for x in ORD[dm] if len(tok(" " + x, add_special_tokens=False)["input_ids"]) == 1] for dm in ORD}
    ordv["letters"] = list("bcdefghijklmnopqrstuvwx")
    lookup2.KEYS = NEW_KEYS; rr = random.Random("controls-v1")
    ev = {dm: [lookup2.make(rr, ordv[dm], dm, "arrow") for _ in range(300)] for dm in ORD}
    lookup2.KEYS = KEYS
    T = lambda cases: [c["text"] for c in cases]

    def rates(cases, hooks=()):
        top = lk.fwd(T(cases), list(hooks)).argmax(-1).tolist(); n = len(cases)
        return dict(successor=round(sum(t == lk.first(c["s"]) for t, c in zip(top, cases)) / n, 3), lookup=round(sum(t == lk.first(c["v"]) for t, c in zip(top, cases)) / n, 3))

    @torch.inference_mode()
    def attn_on_value(head, cases):
        m = 0.0
        for c in cases:
            ids = torch.tensor([tok(c["text"], add_special_tokens=False)["input_ids"]]); pos = len(tok(c["prefix"], add_special_tokens=False)["input_ids"]) - 1
            m += float(lk.model(input_ids=ids, output_attentions=True, use_cache=False).attentions[A[0]][0, head, -1, pos])
        return round(m / len(cases), 3)

    show = lambda r: " ".join(f"{dm} succ {r[dm]['successor']:.2f}/look {r[dm]['lookup']:.2f}" for dm in ORD)
    res["native"] = {dm: rates(ev[dm]) for dm in ORD}; print(f"[controls] native: {show(res['native'])}", flush=True)

    # ---- 1 host specificity
    kvdiv = lk.lab.kvdiv; hosts = [A[1]] + [h for h in (0, 3, 8, 10, 13, 15) if h // kvdiv != A[1] // kvdiv]
    res["host"] = {}
    for h in hosts:
        install(h, prog["WQ"], prog["WK"]); r = {dm: rates(ev[dm]) for dm in ORD}; r["attention_on_value"] = attn_on_value(h, ev["letters"][:40] + ev["months"][:40]); restore()
        res["host"][f"h{h}"] = r
        print(f"[controls] 1 programme hosted by L{A[0]}h{h}{' (successor head)' if h == A[1] else ' (own OV)'}: attention on value {r['attention_on_value']:.2f} | {show(r)}", flush=True)

    # ---- 2 shared key group
    nb = A[1] ^ 1 if kvdiv == 2 else None
    if nb is not None:
        res["neighbour_removed_native"] = {dm: rates(ev[dm], [lk.set_z(A[0], nb)]) for dm in ORD}
        install(A[1], prog["WQ"], prog["WK"]); res["edit"] = {dm: rates(ev[dm]) for dm in ORD}
        res["neighbour_removed_edit"] = {dm: rates(ev[dm], [lk.set_z(A[0], nb)]) for dm in ORD}; restore()
        print(f"[controls] 2 edit: {show(res['edit'])}", flush=True)
        print(f"[controls] 2 edit with neighbour h{nb} removed: {show(res['neighbour_removed_edit'])}", flush=True)
        print(f"[controls] 2 native with neighbour h{nb} removed: {show(res['neighbour_removed_native'])}", flush=True)

    # ---- 3 counterfactual prompts and distinct-value swaps
    res["counterfactual"] = {}
    for mode in ("adjacent_allowed", "successor_present", "long_table"):
        cs = {dm: [make_cf(rr, ordv[dm], dm, mode, NEW_KEYS) for _ in range(300)] for dm in ORD if not (mode == "long_table" and len(ordv[dm]) < 9)}
        nat = {dm: rates(cs[dm]) for dm in cs}; install(A[1], prog["WQ"], prog["WK"]); ed = {dm: rates(cs[dm]) for dm in cs}; restore()
        res["counterfactual"][mode] = dict(native=nat, edit=ed)
        print(f"[controls] 3 {mode}: " + " ".join(f"{dm} succ {nat[dm]['successor']:.2f}->{ed[dm]['successor']:.2f} (look {nat[dm]['lookup']:.2f}->{ed[dm]['lookup']:.2f})" for dm in cs), flush=True)
    install(A[1], prog["WQ"], prog["WK"]); res["swap_distinct"] = {}
    for dm in ORD:
        cases = ev[dm]; n = len(cases); zs = []; lk.fwd(T(cases), [lk.grab_z(A[0], A[1], zs)]); z = torch.cat(zs)
        sh = [(i + 97) % n for i in range(n)]; top = lk.fwd(T(cases), [lk.set_z(A[0], A[1], z[torch.tensor(sh)])]).argmax(-1).tolist()
        keep = [i for i in range(n) if cases[sh[i]]["v"] != cases[i]["v"] and cases[sh[i]]["s"] != cases[i]["s"]]
        res["swap_distinct"][dm] = dict(n=len(keep), donor_successor=round(sum(top[i] == lk.first(cases[sh[i]]["s"]) for i in keep) / len(keep), 3),
                                        own_successor=round(sum(top[i] == lk.first(cases[i]["s"]) for i in keep) / len(keep), 3),
                                        own_value=round(sum(top[i] == lk.first(cases[i]["v"]) for i in keep) / len(keep), 3))
    restore()
    print("[controls] 3 swaps with a different donor value only: " + " ".join(f"{dm} donor-succ {v['donor_successor']:.2f}/own-succ {v['own_successor']:.2f} (n {v['n']})" for dm, v in res["swap_distinct"].items()), flush=True)

    # ---- 4 structural null: random singular directions, same singular values / rank / placement
    res["structural_null"] = []
    for seed in range(3):
        g = torch.Generator(device="cpu").manual_seed(seed); r = prog["WQ"].shape[0]
        Qa, _ = torch.linalg.qr(torch.randn(d, r, generator=g, device="cpu")); Qb, _ = torch.linalg.qr(torch.randn(d, r, generator=g, device="cpu"))
        WQn = Qa.T * prog["WQ"].norm(dim=1, keepdim=True).cpu(); WKn = Qb.T * prog["WK"].norm(dim=1, keepdim=True).cpu()
        install(A[1], WQn, WKn); rr_ = {dm: rates(ev[dm]) for dm in ORD}; rr_["attention_on_value"] = attn_on_value(A[1], ev["letters"][:40] + ev["months"][:40]); restore()
        res["structural_null"].append(rr_)
        print(f"[controls] 4 structural null seed {seed}: attention on value {rr_['attention_on_value']:.2f} | {show(rr_)}", flush=True)
    (OUT / "lookup3_controls.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[controls] DONE ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
