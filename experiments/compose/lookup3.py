"""Lookup -> successor, third design: closed-form attention programming of the successor head.

lookup2.py: if A = L21h6 attends only to the queried value token and its output is amplified x3, the model
answers with the SUCCESSOR of the looked-up value on 0.62-0.79 of prompts in five ordinal domains (native
0.00-0.12). Copying an existing retrieval head's query/key weights into A does not achieve this.

Here A's query and key weights are written in closed form so that, from the final position, A attends to the
queried value. The bilinear form is a whitened cross-covariance between the final position's layer-21 input
and the value position's input (minus the mean of the other positions), fitted ONLY on lookups whose values
are ordinary words. It is placed in the low-frequency RoPE dimensions so that it does not depend on distance.
A's OV circuit (the successor computation) is untouched apart from a scalar gain on its output projection.

No gradients, no API. The fit is supervised linear algebra on non-ordinal data; say so in any write-up.
"""
import json
import random
import time
from pathlib import Path

import torch

from ..evolution import backend
from .bridge import A, HD, OUT
from .lookup import KEYS, LK, ORD, score
from .lookup2 import make

FMT = "arrow"
LOW = list(range(32, 64)) + list(range(96, 128))      # RoPE pairs (j, j+64) with j >= 32: rotation per position < 2e-3 rad
RANK = len(LOW)


def main():
    OUT.mkdir(parents=True, exist_ok=True); lk = LK(); tok = lk.tok; t0 = time.time(); res = {}; P = lk.P
    try:
        lk.model.set_attn_implementation("eager")
    except Exception:
        lk.model.config._attn_implementation = "eager"
    banned = {w.lower() for items in ORD.values() for w in items} | set(KEYS); words = []
    for i in range(tok.vocab_size):
        s = tok.decode([i])
        if s.startswith(" ") and s[1:].isalpha() and s[1:].islower() and 4 <= len(s) <= 9 and s[1:] not in banned:
            words.append(s[1:])
        if len(words) >= 3000:
            break
    ordv = {d: [x for x in ORD[d] if len(tok(" " + x, add_special_tokens=False)["input_ids"]) == 1] for d in ORD}
    ordv["letters"] = list("bcdefghijklmnopqrstuvwx")
    rr = random.Random("lookup-v3")
    fit_cases = [make(rr, words, None, FMT) for _ in range(1500)]; val_cases = [make(rr, words, None, FMT) for _ in range(200)]
    ev = {d: [make(rr, ordv[d], d, FMT) for _ in range(300)] for d in ORD}       # fresh prompts (new seed), not those of lookup2.py
    T = lambda cases: [c["text"] for c in cases]
    sc = lambda cases, lg: score(lk, lg, [(c["text"], c["v"], c["s"]) for c in cases])

    @torch.inference_mode()
    def run1(c, want_attn=False):
        ids = torch.tensor([tok(c["text"], add_special_tokens=False)["input_ids"]]); pos = len(tok(c["prefix"], add_special_tokens=False)["input_ids"]) - 1
        st = {}

        def pre(mod, args, kwargs):
            st["n"] = (args[0] if args else kwargs["hidden_states"])[0].double().clone()
        h = lk.layers[A[0]].self_attn.register_forward_pre_hook(pre, with_kwargs=True)
        try:
            out = lk.model(input_ids=ids, output_attentions=want_attn, use_cache=False)
        finally:
            h.remove()
        att = out.attentions[A[0]][0, A[1], -1].float() if want_attn else None
        return st["n"], pos, att

    # ---- statistics on ordinary-word lookups
    d = P[f"model.layers.{A[0]}.self_attn.q_proj.weight"].shape[1]; Cff = torch.zeros(d, d, dtype=torch.float64); Ckk = torch.zeros(d, d, dtype=torch.float64); X = torch.zeros(d, d, dtype=torch.float64)
    Xsh = torch.zeros(d, d, dtype=torch.float64); nk = 0; rs = random.Random(5)
    for c in fit_cases:
        n, pos, _ = run1(c); f = n[-1]; neg = torch.cat([n[:pos], n[pos + 1:]]).mean(0)
        Cff += torch.outer(f, f); Ckk += n.T @ n; nk += n.shape[0]; X += torch.outer(f, n[pos] - neg)
        wrong = rs.choice([s for s in range(n.shape[0] - 1) if s != pos]); negw = torch.cat([n[:wrong], n[wrong + 1:]]).mean(0)
        Xsh += torch.outer(f, n[wrong] - negw)                                         # control: a random other position as the "target"
    N = len(fit_cases); Cff /= N; Ckk /= nk; X /= N; Xsh /= N
    pre = f"model.layers.{A[0]}.self_attn."; qn, kn, on = pre + "q_proj.weight", pre + "k_proj.weight", pre + "o_proj.weight"
    names = [qn, kn, on] + [n_ for n_ in (pre + "q_proj.bias", pre + "k_proj.bias") if n_ in P]
    orig = {n_: P[n_].detach().clone() for n_ in names}; ka = lk.lab.kvsl(A[1]); qa = lk.hsl(A[1])
    low = torch.tensor(LOW)

    def build(cross, lam):
        I = torch.eye(d, dtype=torch.float64)
        B = torch.linalg.solve(Cff + lam * torch.trace(Cff) / d * I, cross) @ torch.linalg.inv(Ckk + lam * torch.trace(Ckk) / d * I)
        U, S, Vh = torch.linalg.svd(B); r = RANK
        return (U[:, :r] * S[:r].sqrt()).T, S[:r].sqrt()[:, None] * Vh[:r]              # (r,1024) query rows, (r,1024) key rows

    def install(WQ, WK, gain):
        with torch.no_grad():
            q = torch.zeros(HD, d, dtype=P[qn].dtype); k = torch.zeros(HD, d, dtype=P[kn].dtype)
            q[low] = WQ.to(q.dtype); k[low] = WK.to(k.dtype)
            P[qn][qa] = q; P[kn][ka] = k
            for b in (pre + "q_proj.bias", pre + "k_proj.bias"):
                if b in P:
                    P[b][qa if "q_proj" in b else ka] = 0
            P[on][:, qa] = gain * orig[on][:, qa]

    def restore():
        with torch.no_grad():
            for n_, w in orig.items():
                P[n_].copy_(w)

    def attn_stats(cases):
        m = hit = 0
        for c in cases:
            _, pos, att = run1(c, want_attn=True); m += float(att[pos]); hit += int(att.argmax()) == pos
        return dict(mean_attention_on_value=round(m / len(cases), 3), argmax_is_value=round(hit / len(cases), 3))

    wiki = json.loads(Path("results/evolution/splits.json").read_text())["wiki_dev"]["items"]; nll0 = backend.perplexity_nll(lk.model, wiki)
    seq = [(" ".join(x), y) for x, y in [(list("defg"), "h"), (list("klmn"), "o"), (list("pqrs"), "t"), (["Monday", "Tuesday", "Wednesday"], "Thursday"),
                                         (["March", "April", "May"], "June"), (["one", "two", "three"], "four"), (["first", "second", "third"], "fourth")]]

    def measure():
        out = {dm: sc(ev[dm], lk.fwd(T(ev[dm]))) for dm in ORD}
        out["word_lookup"] = sc(val_cases, lk.fwd(T(val_cases)))["lookup"]; out["wiki_nll"] = backend.perplexity_nll(lk.model, wiki)
        lg = lk.fwd([s[0] for s in seq]); out["sequence_next"] = sum(int(t) == lk.first(s[1]) for t, s in zip(lg.argmax(-1).tolist(), seq)) / len(seq)
        return out

    res["native"] = measure(); res["native_attention"] = attn_stats(val_cases[:100])
    print("[lookup3] native: " + " ".join(f"{dm} succ {res['native'][dm]['successor']:.2f}/look {res['native'][dm]['lookup']:.2f}" for dm in ORD)
          + f" | word lookup {res['native']['word_lookup']:.2f} | A's attention on the queried value {res['native_attention']['mean_attention_on_value']:.2f}", flush=True)

    # ---- choose the ridge strength by attention accuracy on HELD-OUT ORDINARY-WORD lookups (no ordinal data)
    best = None; res["attention_fit"] = {}
    for lam in (1e-3, 1e-2, 1e-1, 1.0):
        WQ, WK = build(X, lam); install(WQ, WK, 1.0); a = attn_stats(val_cases[:100]); restore(); res["attention_fit"][str(lam)] = a
        print(f"[lookup3] attention programme, lambda {lam}: on held-out word lookups A's attention on the queried value {a['mean_attention_on_value']:.2f}, "
              f"top position is the value in {a['argmax_is_value']:.2f}", flush=True)
        if best is None or a["mean_attention_on_value"] > best[0]:
            best = (a["mean_attention_on_value"], lam, WQ, WK)
    _, lam, WQ, WK = best; res["lambda"] = lam
    install(WQ, WK, 1.0); res["attention_on_ordinal"] = {dm: attn_stats(ev[dm][:60]) for dm in ORD}; restore()
    print("[lookup3] programmed attention on ordinal-value prompts (never used in fitting): " +
          " ".join(f"{dm} {v['mean_attention_on_value']:.2f}" for dm, v in res["attention_on_ordinal"].items()), flush=True)

    # ---- the edit: programmed attention + output gain. All gains reported.
    res["edit"] = {}
    for gain in (1.0, 2.0, 3.0, 4.0):
        install(WQ, WK, gain); m = measure(); restore(); res["edit"][f"gain{gain}"] = m
        print(f"[lookup3] EDIT gain {gain}: " + " ".join(f"{dm} succ {m[dm]['successor']:.2f}/look {m[dm]['lookup']:.2f}" for dm in ORD)
              + f" | word lookup {m['word_lookup']:.2f} | sequences {m['sequence_next']:.2f} | WikiText loss {m['wiki_nll']:.3f} (base {nll0:.3f})", flush=True)
    # ---- controls
    WQs, WKs = build(Xsh, lam)
    for gain in (1.0, 3.0):
        install(WQs, WKs, gain); m = measure(); restore(); res[f"control_wrong_target_gain{gain}"] = m
        print(f"[lookup3] CONTROL attention fitted to a random other position, gain {gain}: " + " ".join(f"{dm} succ {m[dm]['successor']:.2f}/look {m[dm]['lookup']:.2f}" for dm in ORD), flush=True)
    for gain in (1.0, 3.0):
        with torch.no_grad():
            P[on][:, qa] = gain * orig[on][:, qa]
        m = measure(); restore(); res[f"control_gain_only_gain{gain}"] = m
        print(f"[lookup3] CONTROL gain only (native attention), gain {gain}: " + " ".join(f"{dm} succ {m[dm]['successor']:.2f}/look {m[dm]['lookup']:.2f}" for dm in ORD), flush=True)
    # programmed attention with a different head's output projection (successor map disrupted)
    install(WQ, WK, 3.0)
    with torch.no_grad():
        P[on][:, qa] = 3.0 * orig[on][:, lk.hsl((A[1] + 5) % lk.H)]
    m = measure(); restore(); res["control_wrong_output_projection"] = m
    print("[lookup3] CONTROL programmed attention, another head's output projection: " + " ".join(f"{dm} succ {m[dm]['successor']:.2f}/look {m[dm]['lookup']:.2f}" for dm in ORD), flush=True)
    torch.save(dict(WQ=WQ.float().cpu(), WK=WK.float().cpu(), lam=lam, low=LOW, head=A), OUT / "lookup3_attention_programme.pt")
    (OUT / "lookup3.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[lookup3] DONE ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
