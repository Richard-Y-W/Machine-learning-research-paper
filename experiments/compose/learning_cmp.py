"""Gain analysis and learning comparisons for the lookup -> successor edit.

 G  gain curve: successor-minus-copy logit margin and rates as a function of the output gain, closed-form edit
 Q  gradient baseline with the SAME supervision and the SAME parameters: A's query rows and key-group rows are
    trained with Adam to put attention on the queried value in ordinary-word lookups (no ordinal items), then the
    same output gain is applied. Differs from the closed-form edit only in how the rows are obtained.
 F  fine-tuning reference with DIFFERENT, stronger supervision: all attention projections of layers 18-24 are
    trained on composed examples (lookup prompts whose target is the successor) from three ordinal domains;
    letters and months are held out. Not matched supervision; reported as a reference point.
All evaluated on the same fresh prompts (unseen keys), with WikiText loss and ordinary-word lookup as collateral.
"""
import json
import os
import random
import time
from pathlib import Path

import torch

from ..evolution import backend
from .bridge import A, HD, OUT
from . import lookup2
from .lookup import KEYS, LK, ORD
from .lookup3_confirm import GAIN, NEW_KEYS

TRAIN_DOMS, HELD_DOMS = ["numwords", "days", "ordinals"], ["letters", "months"]


def main():
    lk = LK(); tok = lk.tok; P = lk.P; t0 = time.time(); res = dict(gain=GAIN, model=os.environ.get("EVO_MODEL"))
    try:
        lk.model.set_attn_implementation("eager")
    except Exception:
        lk.model.config._attn_implementation = "eager"
    prog = torch.load(OUT / "lookup3_attention_programme.pt"); low = torch.tensor(prog["low"])
    pre = f"model.layers.{A[0]}.self_attn."; qn, kn, on = pre + "q_proj.weight", pre + "k_proj.weight", pre + "o_proj.weight"
    orig = {n: p.detach().clone() for n, p in P.items() if ".self_attn." in n and n.endswith("_proj.weight")}
    d = P[qn].shape[1]; qa = lk.hsl(A[1]); ka = lk.lab.kvsl(A[1])
    banned = {w.lower() for items in ORD.values() for w in items} | set(KEYS) | set(NEW_KEYS); words = []
    for i in range(tok.vocab_size):
        s = tok.decode([i])
        if s.startswith(" ") and s[1:].isalpha() and s[1:].islower() and 4 <= len(s) <= 9 and s[1:] not in banned:
            words.append(s[1:])
        if len(words) >= 3000:
            break
    ordv = {dm: [x for x in ORD[dm] if len(tok(" " + x, add_special_tokens=False)["input_ids"]) == 1] for dm in ORD}
    ordv["letters"] = list("bcdefghijklmnopqrstuvwx")
    rr = random.Random("learning-cmp-v1")
    lookup2.KEYS = NEW_KEYS
    ev = {dm: [lookup2.make(rr, ordv[dm], dm, "arrow") for _ in range(300)] for dm in ORD}
    lookup2.KEYS = KEYS
    wtrain = [lookup2.make(rr, words, None, "arrow") for _ in range(1500)]; wval = [lookup2.make(rr, words, None, "arrow") for _ in range(200)]
    ctrain = [lookup2.make(rr, ordv[dm], dm, "arrow") for dm in TRAIN_DOMS for _ in range(500)]; rr.shuffle(ctrain)
    wiki = json.loads(Path("results/evolution/splits.json").read_text())["wiki_dev"]["items"]
    T = lambda cases: [c["text"] for c in cases]

    def restore():
        with torch.no_grad():
            for n, w in orig.items():
                P[n].copy_(w)

    def install(gain):
        with torch.no_grad():
            q = torch.zeros(HD, d, dtype=P[qn].dtype); k = torch.zeros(HD, d, dtype=P[kn].dtype)
            q[low] = prog["WQ"].to(q); k[low] = prog["WK"].to(k)
            P[qn][qa] = q; P[kn][ka] = k; P[on][:, qa] = gain * orig[on][:, qa]

    def measure():
        out = {}
        for dm in ORD:
            lg = lk.fwd(T(ev[dm])); top = lg.argmax(-1).tolist(); n = len(top); ar = torch.arange(n)
            s = torch.tensor([lk.first(c["s"]) for c in ev[dm]]); v = torch.tensor([lk.first(c["v"]) for c in ev[dm]])
            out[dm] = dict(successor=round(sum(t == int(x) for t, x in zip(top, s)) / n, 3), lookup=round(sum(t == int(x) for t, x in zip(top, v)) / n, 3),
                           margin=round(float((lg[ar, s] - lg[ar, v]).mean()), 3))
        lg = lk.fwd(T(wval)); out["word_lookup"] = round(sum(int(t) == lk.first(c["v"]) for t, c in zip(lg.argmax(-1).tolist(), wval)) / len(wval), 3)
        out["wiki_nll"] = round(backend.perplexity_nll(lk.model, wiki), 4)
        return out

    show = lambda m: " ".join(f"{dm} {m[dm]['successor']:.2f}" for dm in ORD) + f" | margins " + " ".join(f"{m[dm]['margin']:+.1f}" for dm in ORD) + f" | word lookup {m['word_lookup']:.2f} | WikiText {m['wiki_nll']:.3f}"
    res["native"] = measure(); print(f"[cmp] native: {show(res['native'])}", flush=True)

    # ---- G gain curve for the closed-form edit
    res["closed_form"] = {}
    for g in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        install(g); res["closed_form"][str(g)] = measure(); restore()
        print(f"[cmp] G closed-form edit, gain {g}: {show(res['closed_form'][str(g)])}", flush=True)

    # ---- Q gradient-trained query/key rows, same parameters and supervision
    def batches(cases, bs, steps, seed):
        r = random.Random(seed)
        for _ in range(steps):
            yield r.sample(cases, bs)

    def enc(cases):
        ids, mask = lk.batch(T(cases)); Tn = ids.shape[1]
        pos = torch.tensor([Tn - len(tok(c["text"], add_special_tokens=False)["input_ids"]) + len(tok(c["prefix"], add_special_tokens=False)["input_ids"]) - 1 for c in cases])
        return ids, mask, pos

    res["gradient_qk"] = {}
    for init in ("native", "zero"):
        restore()
        with torch.no_grad():
            if init == "zero":
                P[qn][qa] = 0.01 * torch.randn_like(P[qn][qa]); P[kn][ka] = 0.01 * torch.randn_like(P[kn][ka])
        P[qn].requires_grad_(True); P[kn].requires_grad_(True)
        opt = torch.optim.Adam([P[qn], P[kn]], lr=1e-3 if init == "native" else 3e-3); hist = []
        mq = torch.zeros_like(P[qn]); mq[qa] = 1; mk = torch.zeros_like(P[kn]); mk[ka] = 1
        with torch.enable_grad():
            for step, b in enumerate(batches(wtrain, 32, 300, 0)):
                ids, mask, pos = enc(b)
                att = lk.model(input_ids=ids, attention_mask=mask, output_attentions=True, use_cache=False).attentions[A[0]][:, A[1], -1].float()
                loss = -(att[torch.arange(len(b)), pos] + 1e-9).log().mean()
                opt.zero_grad(); loss.backward(); P[qn].grad *= mq; P[kn].grad *= mk; opt.step()
                if step % 50 == 0 or step == 299:
                    hist.append((step, round(float(loss), 3)))
        P[qn].requires_grad_(False); P[kn].requires_grad_(False)
        with torch.inference_mode():
            ids, mask, pos = enc(wval); att = lk.model(input_ids=ids, attention_mask=mask, output_attentions=True, use_cache=False).attentions[A[0]][:, A[1], -1].float()
            aval = float(att[torch.arange(len(wval)), pos].mean())
        row = dict(loss_history=hist, heldout_word_attention=round(aval, 3))
        for g in (1.0, 3.0):
            with torch.no_grad():
                P[on][:, qa] = g * orig[on][:, qa]
            row[f"gain{g}"] = measure()
        res["gradient_qk"][init] = row; restore()
        print(f"[cmp] Q gradient-trained QK (init {init}; 300 Adam steps, same data): held-out word attention {aval:.2f}; loss {hist[0][1]} -> {hist[-1][1]}", flush=True)
        for g in (1.0, 3.0):
            print(f"[cmp] Q   gain {g}: {show(row[f'gain{g}'])}", flush=True)

    # ---- F fine-tuning reference on composed examples (three training domains; letters and months held out)
    names = [n for n in orig if any(f".layers.{l}." in n for l in range(18, 25))]
    for n in names:
        P[n].requires_grad_(True)
    opt = torch.optim.Adam([P[n] for n in names], lr=2e-5); res["finetune"] = {}
    with torch.enable_grad():
        for step, b in enumerate(batches(ctrain, 32, 300, 1)):
            ids, mask = lk.batch(T(b)); tgt = torch.tensor([lk.first(c["s"]) for c in b])
            loss = torch.nn.functional.cross_entropy(lk.model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, -1].float(), tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            if step in (49, 99, 199, 299):
                for n in names:
                    P[n].requires_grad_(False)
                with torch.inference_mode():
                    res["finetune"][f"step{step + 1}"] = dict(train_loss=round(float(loss), 3), **measure())
                print(f"[cmp] F fine-tune (composed examples from {TRAIN_DOMS}; held out {HELD_DOMS}) step {step + 1}: loss {float(loss):.3f} | {show(res['finetune'][f'step{step + 1}'])}", flush=True)
                for n in names:
                    P[n].requires_grad_(True)
    for n in names:
        P[n].requires_grad_(False)
    restore()
    res["finetune_trainable_parameters"] = int(sum(orig[n].numel() for n in names)); res["closed_form_parameters_written"] = int(2 * len(prog["low"]) * d + HD * d)
    (OUT / "learning_cmp.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[cmp] DONE ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
