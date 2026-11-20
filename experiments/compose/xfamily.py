"""Cross-family replication of the routing edit under a protocol declared before any result (reviewer item 2).

Architecture-generic re-implementation of the frozen procedure (find_head -> lookup3 -> confirmation + controls),
with an adapter for Qwen3 (sanity check against the archived numbers) and GPT-NeoX/Pythia.

DECLARED PROTOCOL (written before the procedure was run on any model outside Qwen):
  S1 locate   Sequence prompts (letters d..w, number words, weekdays, months, ordinals; four layouts). Rank heads by the
              drop in next-item logit margin when the head's final-position output is zeroed; for the top 6, oracle value
              patch (output := own value vector from the context ending one item later); head = argmax two-step rate.
              No manual choice. If the best two-step rate is < 0.20 the model is reported as having no located head.
  S2 gate     Native rates on lookups (ordinary words; ordinal values) and sequences. Reported, no stopping rule.
  S3 fit      lookup3 closed form on 1500 ordinary-word lookups (KEYS, arrow format), ridge chosen from {1e-3..1} by
              held-out ordinary-word attention. Architecture-specific adjustment: the programme is written into the
              position-independent query/key dimensions (Qwen3: low-frequency RoPE pairs 32-63, 96-127; NeoX: the
              non-rotary dimensions rotary_ndims..head_dim, which carry no position signal at all). Query/key biases of
              the programmed head are zeroed.
  S4 gain     chosen from {1,2,3,4,6} on DEVELOPMENT composed prompts with 'days' values only (KEYS, arrow format),
              maximising successor rate. 'days' is then flagged as a development value type in the confirmation.
  S5 confirm  TEST_KEYS (never used), three formats x five value types, N_CONF prompts per cell, gain frozen. Paired
              wins/losses vs unedited. Causal checks with the edit in place: A's output removed; A's output swapped
              with that of a prompt querying a different value (donor-successor rate). Routing controls: the same
              programme and gain installed in three other heads of A's layer (fixed seed), A untouched; the programme in A
              with another head's output projection.
  S6 cost     WikiText-103 validation (N_WIKI x 256 tokens) loss and native sequence accuracy, unedited vs edited.

    EVO_MODEL=EleutherAI/pythia-1.4b-deduped XF_OUT=results/xfamily_pythia1p4b python -m experiments.compose.xfamily
"""
import json
import os
import random
import time
from pathlib import Path

import torch

from ..evolution import backend
from . import lookup2
from .bridge import DOMAINS, LAY, contexts
from .gate import AB, XS
from .lookup import KEYS, ORD
from .robustness import TEST_KEYS, word_pool

XO = Path(os.environ.get("XF_OUT", "results/xfamily"))
N_CONF = int(os.environ.get("XF_NCONF", "120")); N_WIKI = int(os.environ.get("XF_NWIKI", "250")); N_FIT = int(os.environ.get("XF_NFIT", "1500"))
GAINS = (1.0, 2.0, 3.0, 4.0, 6.0); FMTS = ("arrow", "colon", "equals")


class _Stop(Exception):
    pass


class Arch:
    """Uniform access to per-head attention parameters and activations."""

    def __init__(self, name):
        backend.set_threads(); self.model, self.tok = backend.load_model(name); cfg = self.model.config; self.name = name
        try:
            self.model.set_attn_implementation("eager")
        except Exception:
            cfg._attn_implementation = "eager"
        self.neox = cfg.model_type == "gpt_neox"; self.H = cfg.num_attention_heads; self.D = cfg.hidden_size; self.hd = self.D // self.H
        if self.neox:
            self.body = self.model.gpt_neox; self.layers = self.body.layers; self.head_out = self.model.get_output_embeddings(); self.kvdiv = 1
            rot = self.layers[0].attention.rotary_ndims
            self.band = list(range(rot, self.hd)); self.rot = rot
        else:
            self.body = self.model.model; self.layers = self.body.layers; self.head_out = self.model.lm_head
            self.kvdiv = cfg.num_attention_heads // cfg.num_key_value_heads; self.hd = getattr(cfg, "head_dim", self.hd)
            assert self.hd == 128, "Qwen band defined for head_dim 128"
            self.band = list(range(32, 64)) + list(range(96, 128)); self.rot = None
        self.L = len(self.layers); self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        self.saved = {}

    # ----- parameter views (rows of the projection weights that belong to a head)
    def attn(self, l):
        return self.layers[l].attention if self.neox else self.layers[l].self_attn

    def oproj(self, l):
        return self.attn(l).dense if self.neox else self.attn(l).o_proj

    def hs(self, h):
        return slice(h * self.hd, (h + 1) * self.hd)

    def rows(self, l, h, which):
        """(weight, bias or None, row slice) for q/k/v of head h."""
        a = self.attn(l)
        if self.neox:
            base = h * 3 * self.hd; off = dict(q=0, k=1, v=2)[which]
            return a.query_key_value.weight, a.query_key_value.bias, slice(base + off * self.hd, base + (off + 1) * self.hd)
        lin = dict(q=a.q_proj, k=a.k_proj, v=a.v_proj)[which]
        g = h if which == "q" else h // self.kvdiv
        return lin.weight, lin.bias, slice(g * self.hd, (g + 1) * self.hd)

    def snapshot(self, l):
        a = self.attn(l)
        for n, p in a.named_parameters():
            self.saved.setdefault((l, n), p.detach().clone())

    def restore(self):
        with torch.no_grad():
            for (l, n), w in self.saved.items():
                dict(self.attn(l).named_parameters())[n].copy_(w)

    def orig(self, l, n):
        return self.saved[(l, n)]

    # ----- edits
    def program(self, l, h, WQ, WK):
        self.snapshot(l); band = torch.tensor(self.band)
        with torch.no_grad():
            for which, W in (("q", WQ), ("k", WK)):
                w, b, sl = self.rows(l, h, which); new = torch.zeros(self.hd, w.shape[1], dtype=w.dtype); new[band] = W.to(w.dtype); w[sl] = new
                if b is not None:
                    b[sl] = 0

    def gain(self, l, h, g, src=None):
        self.snapshot(l); o = self.oproj(l); nm = "dense.weight" if self.neox else "o_proj.weight"
        with torch.no_grad():
            o.weight[:, self.hs(h)] = g * self.orig(l, nm)[:, self.hs(h if src is None else src)]

    # ----- forwards
    def first(self, item):
        return self.tok(" " + item, add_special_tokens=False)["input_ids"][0]

    def ids(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def batch(self, texts):
        enc = [self.ids(t) for t in texts]; T = max(map(len, enc))
        ids = torch.full((len(enc), T), self.pad); mask = torch.zeros((len(enc), T), dtype=torch.long)
        for r, e in enumerate(enc):
            ids[r, T - len(e):] = torch.tensor(e); mask[r, T - len(e):] = 1
        return ids, mask

    @torch.inference_mode()
    def last(self, texts, hooks=(), bs=32):
        """hooks: functions(batch_slice) -> handle. Returns final-position logits."""
        out = []
        for i in range(0, len(texts), bs):
            sl = slice(i, i + bs); ids, mask = self.batch(texts[sl]); hh = [f(sl) for f in hooks]
            try:
                h = self.body(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state[:, -1]
            finally:
                for x in hh:
                    x.remove()
            out.append(self.head_out(h).float())
        return torch.cat(out)

    def zhook(self, l, h, vecs=None, store=None):
        """Pre-hook on the output projection at the final position: store z, or set it (None -> zero)."""
        def reg(sl):
            def pre(mod, args):
                if store is not None:
                    store.append(args[0][:, -1, self.hs(h)].clone()); return None
                z = args[0].clone(); z[:, -1, self.hs(h)] = 0 if vecs is None else vecs[sl].to(z.dtype)
                return (z,)
            return self.oproj(l).register_forward_pre_hook(pre)
        return reg

    def vhook(self, l, h, store):
        """Stores head h's value vector at the final position."""
        a = self.attn(l)
        def reg(sl):
            if self.neox:
                _, _, rs = self.rows(l, h, "v")
                return a.query_key_value.register_forward_hook(lambda m, i, o: store.append(o[:, -1, rs].clone()))
            _, _, rs = self.rows(l, h, "v")
            return a.v_proj.register_forward_hook(lambda m, i, o: store.append(o[:, -1, rs].clone()))
        return reg

    @torch.inference_mode()
    def layer_input(self, l, text):
        st = {}
        def pre(mod, args, kwargs):
            st["n"] = (args[0] if args else kwargs["hidden_states"])[0].double().clone(); raise _Stop
        hk = self.attn(l).register_forward_pre_hook(pre, with_kwargs=True)
        try:
            self.body(input_ids=torch.tensor([self.ids(text)]), use_cache=False)
        except _Stop:
            pass
        finally:
            hk.remove()
        return st["n"]

    @torch.inference_mode()
    def attn_row(self, l, h, text):
        st = {}
        def fh(mod, args, out):
            st["a"] = out[1][0, h, -1].float(); raise _Stop
        hk = self.attn(l).register_forward_hook(fh)
        try:
            self.body(input_ids=torch.tensor([self.ids(text)]), use_cache=False, output_attentions=True)
        except _Stop:
            pass
        finally:
            hk.remove()
        return st["a"]

    @torch.inference_mode()
    def chunk_nll(self, chunks, bs=2):
        out = []
        for i in range(0, len(chunks), bs):
            ids = torch.tensor(chunks[i:i + bs]); h = self.body(input_ids=ids, use_cache=False).last_hidden_state[:, :-1]
            lp = torch.log_softmax(self.head_out(h).float(), -1); out += (-lp.gather(-1, ids[:, 1:, None]).squeeze(-1)).mean(1).tolist()
        return out


def seq_cases(ar):
    cases = []
    for x in XS:
        i = AB.index(x)
        for lay in LAY:
            cases.append((LAY[lay](list(AB[i - 3:i + 1])), LAY[lay](list(AB[i - 2:i + 2])), AB[i + 1], AB[i + 2]))
    for dom, items in DOMAINS.items():
        if dom == "digits":
            continue
        for w, a, b in contexts(items, dom in ("days", "months")):
            if ar.first(a) != ar.first(b):
                cases += [(LAY[lay](w), LAY[lay](w[1:] + [a]), a, b) for lay in LAY]
    return cases


def locate(ar, res, save):
    if "locate" in res:
        return tuple(res["locate"]["head"])
    cases = seq_cases(ar); scan_cases = cases[::3]; texts = [c[0] for c in scan_cases]
    ta = torch.tensor([ar.first(c[2]) for c in scan_cases]); tb = torch.tensor([ar.first(c[3]) for c in scan_cases]); r = torch.arange(len(texts))
    lg = ar.last(texts); base_m = float((lg[r, ta] - lg[r, tb]).mean()); base_acc = float((lg.argmax(-1) == ta).float().mean())
    print(f"[xf] S1 {ar.name}: {ar.L} layers x {ar.H} heads; native next-item accuracy {base_acc:.2f} on {len(texts)} prompts", flush=True)
    scan = []; t1 = time.time()
    scan_layers = [int(x) for x in os.environ["XF_SCAN_LAYERS"].split(",")] if os.environ.get("XF_SCAN_LAYERS") else range(ar.L)   # smoke tests only
    for l in scan_layers:
        for h in range(ar.H):
            lg = ar.last(texts, [ar.zhook(l, h)]); scan.append(dict(layer=l, head=h, d_margin=float((lg[r, ta] - lg[r, tb]).mean()) - base_m))
        if l == scan_layers[0]:
            print(f"[xf] S1 scan: {time.time() - t1:.0f}s per layer", flush=True)
    scan.sort(key=lambda x: x["d_margin"])
    print("[xf] S1 largest next-item margin drops: " + ", ".join(f"L{x['layer']}h{x['head']} {x['d_margin']:+.2f}" for x in scan[:8]), flush=True)
    t_all = [c[0] for c in cases]; t2 = [c[1] for c in cases]; tb_all = [ar.first(c[3]) for c in cases]; oracle = []
    for x in scan[:6]:
        l, h = x["layer"], x["head"]; vs = []; ar.last(t2, [ar.vhook(l, h, vs)]); v = torch.cat(vs)
        top = ar.last(t_all, [ar.zhook(l, h, v)]).argmax(-1).tolist(); two = sum(t == b for t, b in zip(top, tb_all)) / len(cases)
        oracle.append(dict(layer=l, head=h, two_step=two)); print(f"[xf] S1 oracle value patch L{l}h{h}: two-step rate {two:.2f}", flush=True)
    best = max(oracle, key=lambda x: x["two_step"])
    res["locate"] = dict(native_next=base_acc, n_scan=len(texts), n_oracle=len(cases), scan=scan[:30], oracle=oracle, head=[best["layer"], best["head"]],
                         located=best["two_step"] >= 0.20); save()
    print(f"[xf] S1 SUCCESSOR HEAD: L{best['layer']}h{best['head']} (two-step {best['two_step']:.2f})", flush=True)
    return best["layer"], best["head"]


def main():
    XO.mkdir(parents=True, exist_ok=True); name = os.environ.get("EVO_MODEL", "EleutherAI/pythia-1.4b-deduped"); ar = Arch(name); t0 = time.time()
    resf = XO / "xfamily.json"; res = json.loads(resf.read_text()) if resf.exists() else {}
    res["model"] = name; res["band"] = ar.band; res["head_dim"] = ar.hd
    save = lambda: resf.write_text(json.dumps(res, indent=1), encoding="utf-8")
    l, h = locate(ar, res, save)
    if not res["locate"]["located"]:
        print("[xf] no successor head located under the declared threshold; stop", flush=True); return
    tok = ar.tok; one = lambda s: len(ar.ids(" " + s)) == 1
    ordv = {dm: [x for x in ORD[dm] if one(x)] for dm in ORD}; ordv["letters"] = list("bcdefghijklmnopqrstuvwx")
    res["single_token_values"] = {dm: len(v) for dm, v in ordv.items()}
    words = word_pool(tok); rr = random.Random("xfamily-v1")
    fit_cases = [lookup2.make(rr, words, None, "arrow") for _ in range(N_FIT)]; wval = [lookup2.make(rr, words, None, "arrow") for _ in range(200)]
    dev = [lookup2.make(rr, ordv["days"], "days", "arrow") for _ in range(200)]
    old = lookup2.KEYS; lookup2.KEYS = TEST_KEYS
    conf = {}
    for fmt in FMTS:
        for dm in ORD:
            r2 = random.Random(f"xfamily-conf-v1|{fmt}|{dm}"); conf[f"{fmt}|{dm}"] = [lookup2.make(r2, ordv[dm], dm, fmt) for _ in range(N_CONF)]
    lookup2.KEYS = old
    T = lambda cs: [c["text"] for c in cs]
    rate = lambda cs, key, hooks=(): (lambda top: sum(t == ar.first(c[key]) for t, c in zip(top, cs)) / len(cs))(ar.last(T(cs), list(hooks)).argmax(-1).tolist())
    pos = lambda c: len(ar.ids(c["prefix"])) - 1

    # ---- S2 native gate
    if "gate" not in res:
        res["gate"] = dict(word_lookup=rate(wval, "v"), dev_days_successor=rate(dev, "s"), dev_days_lookup=rate(dev, "v"),
                           conf_arrow={dm: dict(successor=rate(conf[f"arrow|{dm}"], "s"), lookup=rate(conf[f"arrow|{dm}"], "v")) for dm in ORD})
        att = torch.stack([ar.attn_row(l, h, c["text"])[pos(c)] for c in wval[:60]]); res["gate"]["native_attention_on_value"] = float(att.mean()); save()
    print(f"[xf] S2 native: word lookup {res['gate']['word_lookup']:.2f}; arrow " + " ".join(f"{dm} succ {v['successor']:.2f}/look {v['lookup']:.2f}" for dm, v in res["gate"]["conf_arrow"].items())
          + f"; L{l}h{h} attention on queried value {res['gate']['native_attention_on_value']:.2f}", flush=True)

    # ---- S3 fit
    progf = XO / "programme.pt"
    if progf.exists():
        prog = torch.load(progf); WQ, WK = prog["WQ"], prog["WK"]
    else:
        d = ar.D; Cff = torch.zeros(d, d, dtype=torch.float64); Ckk = torch.zeros(d, d, dtype=torch.float64); X = torch.zeros(d, d, dtype=torch.float64); nk = 0; t1 = time.time()
        for c in fit_cases:
            n = ar.layer_input(l, c["text"]); p = pos(c); f = n[-1]; neg = torch.cat([n[:p], n[p + 1:]]).mean(0)
            Cff += torch.outer(f, f); Ckk += n.T @ n; nk += n.shape[0]; X += torch.outer(f, n[p] - neg)
        N = len(fit_cases); Cff /= N; Ckk /= nk; X /= N; I = torch.eye(d, dtype=torch.float64); r = len(ar.band); best = None; sel = {}
        for lam in (1e-3, 1e-2, 1e-1, 1.0):
            B = torch.linalg.solve(Cff + lam * torch.trace(Cff) / d * I, X) @ torch.linalg.inv(Ckk + lam * torch.trace(Ckk) / d * I)
            U, S, Vh = torch.linalg.svd(B); wq, wk = (U[:, :r] * S[:r].sqrt()).T.float(), (S[:r].sqrt()[:, None] * Vh[:r]).float()
            ar.program(l, h, wq, wk)
            m = hit = 0
            for c in wval[:100]:
                row = ar.attn_row(l, h, c["text"]); m += float(row[pos(c)]); hit += int(row.argmax()) == pos(c)
            ar.restore(); sel[str(lam)] = dict(mean_attention_on_value=m / 100, argmax_is_value=hit / 100)
            print(f"[xf] S3 lambda {lam}: held-out word attention on queried value {m / 100:.2f}, argmax {hit / 100:.2f}", flush=True)
            if best is None or m > best[0]:
                best = (m, lam, wq, wk)
        _, lam, WQ, WK = best; torch.save(dict(WQ=WQ, WK=WK, lam=lam, band=ar.band, head=[l, h]), progf)
        res["fit"] = dict(lam=lam, selection=sel, seconds=round(time.time() - t1)); save()

    # ---- S4 gain on development 'days' prompts
    if "gain" not in res:
        dg = {}
        for g in GAINS:
            ar.program(l, h, WQ, WK); ar.gain(l, h, g); dg[str(g)] = rate(dev, "s"); ar.restore()
        g = max(GAINS, key=lambda x: dg[str(x)]); res["gain"] = dict(chosen=g, dev_days_successor=dg); save()
    G = res["gain"]["chosen"]; print(f"[xf] S4 gain on development days prompts: {res['gain']['dev_days_successor']} -> {G}", flush=True)

    # ---- S5 confirmation + causal checks + routing controls
    others = random.Random(f"xfamily-others|{name}").sample([x for x in range(ar.H) if x != h and x // ar.kvdiv != h // ar.kvdiv], 3)
    if "confirm" not in res:
        res["confirm"] = {}; res["control_heads"] = others
        for key, cs in conf.items():
            s_id = [ar.first(c["s"]) for c in cs]; v_id = [ar.first(c["v"]) for c in cs]; n = len(cs)
            base = ar.last(T(cs)).argmax(-1).tolist()
            ar.program(l, h, WQ, WK); ar.gain(l, h, G)
            edit = ar.last(T(cs)).argmax(-1).tolist(); zs = []; ar.last(T(cs), [ar.zhook(l, h, store=zs)]); z = torch.cat(zs)
            noA = ar.last(T(cs), [ar.zhook(l, h)]).argmax(-1).tolist(); sh = [(i + 37) % n for i in range(n)]
            swap = ar.last(T(cs), [ar.zhook(l, h, z[torch.tensor(sh)])]).argmax(-1).tolist(); ar.restore()
            ctl = {}
            for o in others:
                ar.program(l, o, WQ, WK); ar.gain(l, o, G); ctl[str(o)] = sum(t == s for t, s in zip(ar.last(T(cs)).argmax(-1).tolist(), s_id)) / n; ar.restore()
            ar.program(l, h, WQ, WK); ar.gain(l, h, G, src=others[0]); wov = sum(t == s for t, s in zip(ar.last(T(cs)).argmax(-1).tolist(), s_id)) / n; ar.restore()
            b = [t == s for t, s in zip(base, s_id)]; e = [t == s for t, s in zip(edit, s_id)]
            res["confirm"][key] = dict(n=n, native_successor=sum(b) / n, edited_successor=sum(e) / n, wins=sum(y and not x for x, y in zip(b, e)), losses=sum(x and not y for x, y in zip(b, e)),
                                       native_lookup=sum(t == v for t, v in zip(base, v_id)) / n, edited_lookup=sum(t == v for t, v in zip(edit, v_id)) / n,
                                       A_removed_successor=sum(t == s for t, s in zip(noA, s_id)) / n, swap_donor_successor=sum(t == s_id[j] for t, j in zip(swap, sh)) / n,
                                       swap_own_successor=sum(t == s for t, s in zip(swap, s_id)) / n, programme_in_other_heads=ctl, wrong_output_projection=wov)
            r_ = res["confirm"][key]; save()
            print(f"[xf] S5 {key:16s}: successor {r_['native_successor']:.2f} -> {r_['edited_successor']:.2f} (wins {r_['wins']}, losses {r_['losses']}); lookup {r_['native_lookup']:.2f} -> {r_['edited_lookup']:.2f}; "
                  f"A removed {r_['A_removed_successor']:.2f}; swap donor {r_['swap_donor_successor']:.2f}; other heads " + "/".join(f"{v:.2f}" for v in ctl.values()) + f"; wrong OV {wov:.2f}", flush=True)

    # ---- S6 cost
    if "cost" not in res:
        from datasets import load_dataset
        text = "\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")["text"] if t.strip())
        ids = tok(text[:3_000_000])["input_ids"]; wiki = [ids[i * 256:(i + 1) * 256] for i in range(N_WIKI)]
        seq = [(c[0], c[2]) for c in seq_cases(ar)]
        def m():
            top = ar.last([s for s, _ in seq]).argmax(-1).tolist()
            return dict(wiki_chunk_nll=ar.chunk_nll(wiki), seq=sum(t == ar.first(a) for t, (_, a) in zip(top, seq)) / len(seq), word_lookup=rate(wval, "v"))
        un = m(); ar.program(l, h, WQ, WK); ar.gain(l, h, G); ed = m(); ar.restore()
        res["cost"] = dict(n_wiki_tokens=N_WIKI * 256, unedited=un, edited=ed); save()
    c = res["cost"]; mu = lambda x: sum(x) / len(x)
    print(f"[xf] S6 WikiText-val loss {mu(c['unedited']['wiki_chunk_nll']):.4f} -> {mu(c['edited']['wiki_chunk_nll']):.4f}; sequences {c['unedited']['seq']:.2f} -> {c['edited']['seq']:.2f}; "
          f"word lookup {c['unedited']['word_lookup']:.2f} -> {c['edited']['word_lookup']:.2f}", flush=True)
    print(f"[xf] DONE ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
