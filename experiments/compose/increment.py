"""Variable lookup + increment: one step of program execution in a single forward pass (no gradients, no API).

Prompt (code-like):   x=3 / total=7 / n=5 / total=      Native answer: "7".  Target after the edit: "8".
This is the arithmetic form of lookup -> successor: read a variable, add one. Values are single digit tokens.

Frozen procedure from lookup3.py, applied to this format:
 - attention programme for the successor head fitted on assignments whose values are ORDINARY WORDS
   (x=tree / total=river / ...); no digits, no ordinal items, no incremented examples;
 - ridge strength chosen by attention accuracy on held-out word assignments; output gain 3.
Reported on fresh prompts: increment rate (answer = value + 1), copy rate, controls, and a multi-digit check
(two-digit values whose last digit is not 9: the model must copy the tens digit natively; only the final digit
position is evaluated here, so this is reported as the units-digit rate).
"""
import json
import os
import random
import time
from pathlib import Path

import torch

from ..evolution import backend
from .bridge import A, HD, OUT
from .lookup import LK

VARS = "x y z n m k a b c i j total count value result index width height speed score size level depth rate sum".split()
WORDS_BAN = set(VARS)
LOW = list(range(32, 64)) + list(range(96, 128)); GAIN = 3.0


def make(rr, values, n=None):
    n = n or rr.choice([3, 4, 5]); ks = rr.sample(VARS, n); vs = [rr.choice(values) for _ in range(n)]; q = rr.randrange(n)
    text = "".join(f"{k}={v}\n" for k, v in zip(ks, vs)) + f"{ks[q]}="
    pre = "".join(f"{k}={v}\n" for k, v in zip(ks[:q + 1], vs[:q + 1])).rstrip("\n")
    return dict(text=text, prefix=pre, v=vs[q])


def main():
    OUT.mkdir(parents=True, exist_ok=True); lk = LK(); tok = lk.tok; P = lk.P; t0 = time.time(); res = dict(model=os.environ.get("EVO_MODEL"), gain=GAIN)
    try:
        lk.model.set_attn_implementation("eager")
    except Exception:
        lk.model.config._attn_implementation = "eager"
    tid = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    words = []
    for i in range(tok.vocab_size):
        s = tok.decode([i])
        if s.isalpha() and s.islower() and 4 <= len(s) <= 8 and s not in WORDS_BAN and tid("x=" + s)[-1] == i and len(tid("x=" + s)) == len(tid("x=")) + 1:
            words.append(s)
        if len(words) >= 2000:
            break
    digits = [str(i) for i in range(9)]                     # 0..8, so value + 1 is a single digit
    dtok = {d_: tid("x=" + d_)[-1] for d_ in [str(i) for i in range(10)]}
    rr = random.Random("increment-v1")
    fit = [make(rr, words) for _ in range(1500)]; val = [make(rr, words) for _ in range(200)]
    ev = [make(rr, digits) for _ in range(600)]
    two = [make(rr, [str(i) for i in range(10, 99) if i % 10 != 9]) for _ in range(400)]
    T = lambda cs: [c["text"] for c in cs]

    def rates(cs, hooks=(), units=False):
        top = lk.fwd(T(cs), list(hooks)).argmax(-1).tolist(); n = len(cs)
        if units:        # two-digit values: the first generated token is the tens digit; evaluated separately below
            return None
        inc = sum(t == dtok[str(int(c["v"]) + 1)] for t, c in zip(top, cs)); cp = sum(t == dtok[c["v"]] for t, c in zip(top, cs))
        return dict(increment=round(inc / n, 3), copy=round(cp / n, 3), n=n)

    wl = lambda: round(sum(int(t) == tid("x=" + c["v"])[-1] for t, c in zip(lk.fwd(T(val)).argmax(-1).tolist(), val)) / len(val), 3)
    wiki = json.loads(Path("results/evolution/splits.json").read_text())["wiki_dev"]["items"]; nll0 = backend.perplexity_nll(lk.model, wiki)
    res["native"] = dict(digits=rates(ev), word_lookup=wl(), wiki_nll=nll0)
    print(f"[inc] native: digit assignments copy {res['native']['digits']['copy']:.2f}, increment {res['native']['digits']['increment']:.2f}; word-value lookup {res['native']['word_lookup']:.2f}", flush=True)
    # does the model count digits natively in this tokenisation? (sequence check)
    seqs = [(",".join(str(j) for j in range(i, i + 4)) + ",", str(i + 4)) for i in range(0, 5)]
    lg = lk.fwd([s for s, _ in seqs]); res["native_digit_sequence"] = sum(int(t) == dtok[a] for t, (_, a) in zip(lg.argmax(-1).tolist(), seqs)) / len(seqs)
    print(f"[inc] native digit sequence continuation (1,2,3,4, -> 5): {res['native_digit_sequence']:.2f}", flush=True)

    pre = f"model.layers.{A[0]}.self_attn."; qn, kn, on = pre + "q_proj.weight", pre + "k_proj.weight", pre + "o_proj.weight"
    orig = {n_: P[n_].detach().clone() for n_ in (qn, kn, on)}; d = P[qn].shape[1]; qa = lk.hsl(A[1]); ka = lk.lab.kvsl(A[1]); low = torch.tensor(LOW)

    @torch.inference_mode()
    def run1(c, want_attn=False):
        ids = torch.tensor([tid(c["text"])]); pos = len(tid(c["prefix"])) - 1; st = {}

        def prehook(mod, args, kwargs):
            st["n"] = (args[0] if args else kwargs["hidden_states"])[0].double().clone()
        h = lk.layers[A[0]].self_attn.register_forward_pre_hook(prehook, with_kwargs=True)
        try:
            out = lk.model(input_ids=ids, output_attentions=want_attn, use_cache=False)
        finally:
            h.remove()
        return st["n"], pos, (out.attentions[A[0]][0, A[1], -1].float() if want_attn else None)

    # oracle first: A attends only to the queried value, output x gain
    zs = []
    kv = lk.lab.kvsl(A[1])
    for c in ev[:300]:
        ids = torch.tensor([tid(c["text"])]); pos = len(tid(c["prefix"])) - 1; st = {}
        h = lk.layers[A[0]].self_attn.v_proj.register_forward_hook(lambda m, i, o: st.__setitem__("v", o[0, pos, kv].clone()))
        with torch.inference_mode():
            lk.model(input_ids=ids, use_cache=False)
        h.remove(); zs.append(st["v"])
    res["oracle_ideal_routing"] = {str(g): rates(ev[:300], [lk.set_z(A[0], A[1], g * torch.stack(zs))]) for g in (1.0, 2.0, 3.0, 4.0)}
    print("[inc] oracle, A attends only to the queried value (increment/copy by gain): " +
          " ".join(f"x{g}: {r['increment']:.2f}/{r['copy']:.2f}" for g, r in res["oracle_ideal_routing"].items()), flush=True)

    Cff = torch.zeros(d, d, dtype=torch.float64); Ckk = torch.zeros(d, d, dtype=torch.float64); X = torch.zeros(d, d, dtype=torch.float64); Xw = torch.zeros(d, d, dtype=torch.float64); nk = 0
    rs = random.Random(5)
    for c in fit:
        n, pos, _ = run1(c); f = n[-1]; Cff += torch.outer(f, f); Ckk += n.T @ n; nk += n.shape[0]
        X += torch.outer(f, n[pos] - torch.cat([n[:pos], n[pos + 1:]]).mean(0))
        w = rs.choice([s for s in range(n.shape[0] - 1) if s != pos]); Xw += torch.outer(f, n[w] - torch.cat([n[:w], n[w + 1:]]).mean(0))
    Cff /= len(fit); Ckk /= nk; X /= len(fit); Xw /= len(fit)

    def build(cross, lam):
        I = torch.eye(d, dtype=torch.float64)
        B = torch.linalg.solve(Cff + lam * torch.trace(Cff) / d * I, cross) @ torch.linalg.inv(Ckk + lam * torch.trace(Ckk) / d * I)
        U, S, Vh = torch.linalg.svd(B); r = len(LOW)
        return (U[:, :r] * S[:r].sqrt()).T, S[:r].sqrt()[:, None] * Vh[:r]

    def install(WQ, WK, gain):
        with torch.no_grad():
            q = torch.zeros(HD, d, dtype=P[qn].dtype); k = torch.zeros(HD, d, dtype=P[kn].dtype); q[low] = WQ.to(q); k[low] = WK.to(k)
            P[qn][qa] = q; P[kn][ka] = k; P[on][:, qa] = gain * orig[on][:, qa]

    def restore():
        with torch.no_grad():
            for n_, w in orig.items():
                P[n_].copy_(w)

    def attn(cs):
        m = 0.0
        for c in cs:
            _, pos, a = run1(c, True); m += float(a[pos])
        return round(m / len(cs), 3)

    best = None
    for lam in (1e-3, 1e-2, 1e-1, 1.0):
        WQ, WK = build(X, lam); install(WQ, WK, 1.0); a = attn(val[:100]); restore()
        print(f"[inc] attention programme lambda {lam}: attention on the queried value, held-out WORD assignments {a:.2f}", flush=True)
        if best is None or a > best[0]:
            best = (a, lam, WQ, WK)
    _, lam, WQ, WK = best; res["lambda"] = lam
    install(WQ, WK, 1.0); res["attention_on_digit_prompts"] = attn(ev[:100]); restore()
    print(f"[inc] programmed attention on digit assignments (never used in fitting): {res['attention_on_digit_prompts']:.2f}", flush=True)
    res["edit"] = {}
    for g in (1.0, 2.0, 3.0, 4.0):
        install(WQ, WK, g); r = rates(ev); r["word_lookup"] = wl(); r["wiki_nll"] = backend.perplexity_nll(lk.model, wiki); restore(); res["edit"][str(g)] = r
        print(f"[inc] EDIT gain {g}: increment {r['increment']:.2f}, copy {r['copy']:.2f} (n {r['n']}) | word-value lookup {r['word_lookup']:.2f} | WikiText {r['wiki_nll']:.3f} (base {nll0:.3f})", flush=True)
    # per-digit breakdown at the frozen gain, and controls
    install(WQ, WK, GAIN); top = lk.fwd(T(ev)).argmax(-1).tolist()
    res["by_digit"] = {dg: round(sum(t == dtok[str(int(dg) + 1)] for t, c in zip(top, ev) if c["v"] == dg) / max(1, sum(c["v"] == dg for c in ev)), 3) for dg in digits}
    res["edit_A_removed"] = rates(ev, [lk.set_z(A[0], A[1])]); restore()
    print(f"[inc] gain {GAIN}: increment rate by value " + " ".join(f"{dg}->{int(dg) + 1}: {v:.2f}" for dg, v in res["by_digit"].items()), flush=True)
    WQw, WKw = build(Xw, lam); install(WQw, WKw, GAIN); res["control_wrong_target"] = rates(ev); restore()
    with torch.no_grad():
        P[on][:, qa] = GAIN * orig[on][:, qa]
    res["control_gain_only"] = rates(ev); restore()
    install(WQ, WK, GAIN)
    with torch.no_grad():
        P[on][:, qa] = GAIN * orig[on][:, lk.hsl((A[1] + 5) % lk.H)]
    res["control_wrong_output_projection"] = rates(ev); restore()
    for k_ in ("edit_A_removed", "control_wrong_target", "control_gain_only", "control_wrong_output_projection"):
        print(f"[inc] {k_}: increment {res[k_]['increment']:.2f}, copy {res[k_]['copy']:.2f}", flush=True)
    # two-digit values: generate two tokens greedily and compare with value + 1
    def gen2(cs):
        out = []
        for c in cs:
            ids = torch.tensor([tid(c["text"])])
            with torch.inference_mode():
                g_ = lk.model.generate(input_ids=ids, max_new_tokens=3, do_sample=False, pad_token_id=lk.pad)
            out.append(tok.decode(g_[0, ids.shape[1]:]).split("\n")[0].strip())
        return out
    nat = gen2(two[:200]); install(WQ, WK, GAIN); ed = gen2(two[:200]); restore()
    res["two_digit"] = dict(n=200, native_copy=sum(o == c["v"] for o, c in zip(nat, two)) / 200, native_increment=sum(o == str(int(c["v"]) + 1) for o, c in zip(nat, two)) / 200,
                            edit_copy=sum(o == c["v"] for o, c in zip(ed, two)) / 200, edit_increment=sum(o == str(int(c["v"]) + 1) for o, c in zip(ed, two)) / 200)
    print(f"[inc] two-digit values (greedy generation, exact match): native copy {res['two_digit']['native_copy']:.2f} increment {res['two_digit']['native_increment']:.2f}; "
          f"edited copy {res['two_digit']['edit_copy']:.2f} increment {res['two_digit']['edit_increment']:.2f}", flush=True)
    torch.save(dict(WQ=WQ.float().cpu(), WK=WK.float().cpu(), lam=lam, low=LOW, head=A), OUT / "increment_attention_programme.pt")
    (OUT / "increment.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[inc] DONE ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
