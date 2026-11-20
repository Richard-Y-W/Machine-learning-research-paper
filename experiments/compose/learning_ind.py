"""Learning comparisons on the independent evaluation sets (reviewer item 4b).

All conditions are scored with robustness.evaluate on the same sets as the closed-form edit (TEST_KEYS successor
prompts in three formats, WikiText-103 validation, native sequences, fresh word lookups), so damage and success
are directly comparable with robustness.json.

  Q   same supervision, same parameters as the closed form: A's query rows and key-group rows trained with Adam to
      put attention on the queried value in ordinary-word lookups (routing labels; no ordinal items), then gain 3.
  Fm  same parameters, answer supervision: A's query rows, key-group rows and output columns trained on composed
      examples (lookup prompts whose target is the successor) from numwords/days/ordinals; letters and months held out.
      Asks whether the closed form's parameter subspace suffices when the answers are given.
  F   the earlier reference: all attention projections of layers 18-24 trained on the same composed examples.
Each: 300 Adam steps, batch 32, fixed seeds.

    EVO_MODEL=... WSP_OUT=... RB_OUT=results/robustness_0p6b python -m experiments.compose.learning_ind
"""
import json
import os
import random
import time

import torch

from .bridge import A
from . import lookup2
from .lookup import KEYS, ORD
from .robustness import RB, Rig, build_sets, evaluate

TRAIN_DOMS = ["numwords", "days", "ordinals"]; STEPS = int(os.environ.get("RB_STEPS", "300")); BS = 32


def main():
    rig = Rig(); P = rig.P; lk = rig.lk; tok = rig.tok; S = build_sets(rig); t0 = time.time()
    resf = RB / "learning_ind.json"; res = json.loads(resf.read_text()) if resf.exists() else {}
    save = lambda: resf.write_text(json.dumps(res), encoding="utf-8")
    allw = {n: p.detach().clone() for n, p in P.items() if ".self_attn." in n and n.endswith("_proj.weight")}

    def restore_all():
        with torch.no_grad():
            for n, w in allw.items():
                P[n].copy_(w)

    one = lambda s: len(tok(" " + s, add_special_tokens=False)["input_ids"]) == 1
    ordv = {dm: [x for x in ORD[dm] if one(x)] for dm in ORD}
    rr = random.Random("learning-ind-v1"); lookup2.KEYS = KEYS
    wtrain = [lookup2.make(rr, S["word_pool"], None, "arrow") for _ in range(1500)]
    ctrain = [lookup2.make(rr, ordv[dm], dm, "arrow") for dm in TRAIN_DOMS for _ in range(500)]; rr.shuffle(ctrain)
    T = lambda cs: [c["text"] for c in cs]
    qn, kn, on = rig.name("q"), rig.name("k"), rig.name("o"); qa = lk.hsl(A[1]); ka = lk.lab.kvsl(A[1])

    def batches(cases, seed):
        r = random.Random(seed)
        for _ in range(STEPS):
            yield r.sample(cases, BS)

    def train(names, masks, loss_fn, cases, lr, seed):
        for n in names:
            P[n].requires_grad_(True)
        opt = torch.optim.Adam([P[n] for n in names], lr=lr); hist = []
        try:
            with torch.enable_grad():
                for step, b in enumerate(batches(cases, seed)):
                    loss = loss_fn(b); opt.zero_grad(); loss.backward()
                    for n in names:
                        if n in masks:
                            P[n].grad *= masks[n]
                    opt.step()
                    if step % 50 == 0 or step == STEPS - 1:
                        hist.append((step, round(float(loss), 4))); print(f"[lind]   step {step}: loss {float(loss):.4f}", flush=True)
        finally:
            for n in names:
                P[n].requires_grad_(False)
        return hist

    def attn_loss(b):
        ids, mask = lk.batch(T(b)); Tn = ids.shape[1]
        pos = torch.tensor([Tn - len(tok(c["text"], add_special_tokens=False)["input_ids"]) + len(tok(c["prefix"], add_special_tokens=False)["input_ids"]) - 1 for c in b])
        st = {}

        def fh(mod, args, out):
            st["a"] = out[1][:, A[1], -1].float(); raise Rig._Stop
        h = lk.layers[A[0]].self_attn.register_forward_hook(fh)
        try:
            rig.m.model(input_ids=ids, attention_mask=mask, use_cache=False, output_attentions=True)
        except Rig._Stop:
            pass
        finally:
            h.remove()
        return -(st["a"][torch.arange(len(b)), pos] + 1e-9).log().mean()

    def answer_loss(b):
        ids, mask = lk.batch(T(b)); tgt = torch.tensor([lk.first(c["s"]) for c in b])
        h = rig.m.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state[:, -1]
        return torch.nn.functional.cross_entropy(rig.m.lm_head(h).float(), tgt)

    def mask_rows(n, sl, cols=False):
        m = torch.zeros_like(P[n])
        if cols:
            m[:, sl] = 1
        else:
            m[sl] = 1
        return m

    runs = {
        "Q_gradient_routing_g3": dict(names=[qn, kn], masks={qn: mask_rows(qn, qa), kn: mask_rows(kn, ka)}, loss=attn_loss, cases=wtrain, lr=1e-3, gain=3.0),
        "Fm_same_params_composed": dict(names=[qn, kn, on], masks={qn: mask_rows(qn, qa), kn: mask_rows(kn, ka), on: mask_rows(on, qa, cols=True)},
                                        loss=answer_loss, cases=ctrain, lr=1e-3, gain=None),
        "F_layers18_24_composed": dict(names=[n for n in allw if any(f".layers.{l}." in n for l in range(18, 25))], masks={}, loss=answer_loss, cases=ctrain, lr=2e-5, gain=None),
    }
    for name, cfg in runs.items():
        if name in res:
            print(f"[lind] {name}: cached {res[name]['eval']['summary']}", flush=True); continue
        print(f"[lind] {name}: training {sum(int(cfg['masks'][n].sum()) if n in cfg['masks'] else P[n].numel() for n in cfg['names'])} parameters", flush=True)
        t1 = time.time()
        try:
            hist = train(cfg["names"], cfg["masks"], cfg["loss"], cfg["cases"], cfg["lr"], seed=sum(map(ord, name)))
            if cfg["gain"]:
                with torch.no_grad():
                    P[on][:, qa] = cfg["gain"] * allw[on][:, qa]
            ev = evaluate(rig, S)
        finally:
            restore_all()
        res[name] = dict(loss_history=hist, train_seconds=round(time.time() - t1), eval=ev, steps=STEPS); save()
        s = ev["summary"]; print(f"[lind] {name}: successor {s['successor']:.3f} | seq {s['seq']:.3f} | word lookup {s['word_lookup']:.3f} | WikiText-val {s['wiki_nll']:.4f}", flush=True)
    print(f"[lind] DONE ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
