"""Independent evaluation, refit stability and a direct-versus-copied comparison at matched success.

Declared before any result was seen (reviewer items 1 and 4):

  Evaluation sets (none used in fitting, selection or development):
    successor  fresh lookup prompts with a THIRD key set (TEST_KEYS, disjoint from KEYS and NEW_KEYS), three
               formats (arrow, colon, equals) x five value types, N_SUCC prompts per cell, new seeds.
    language   WikiText-103 VALIDATION split (the development probe was four 128-token chunks of the TEST split),
               N_WIKI chunks of 256 tokens; per-chunk losses saved for paired bootstrap intervals.
    sequences  native sequence continuation, seven item types x six layouts x window lengths 2-5, sampled to
               N_SEQ prompts; prompts identical to the development sequence probes are excluded.
    words      fresh ordinary-word lookups (arrow format, TEST_KEYS), reported because the edit changes lookup.

  Conditions:
    unedited
    direct_archived_g{1,2,3,4}   the archived routing fit (lookup3_attention_programme.pt) in the successor head A,
                                 output gain g. Gain 3 is the reported edit.
    direct_refit{k}_g3           k = 1..5: the lookup3 procedure rerun from scratch, each on its own fitting
                                 sample drawn from a DISJOINT fifth of the ordinary-word pool, own seed, own ridge
                                 selection on held-out ordinary-word attention. Gain fixed at 3 (no reselection).
    copied_g{...}                the archived routing fit installed in a spare head S of A's layer, A's value rows
                                 and output columns copied into S, S's KV-group mates silenced, A left native.
                                 S is chosen by a fixed rule: the KV group (other than A's) whose removal raises
                                 development WikiText loss least; S is its lowest-index head.
  Direct and copied edits are compared on their success/damage curves over gain, i.e. damage at matched
  successor rate, not at a single gain.

    EVO_MODEL=Qwen/Qwen3-0.6B-Base WSP_OUT=results/compose RB_OUT=results/robustness_0p6b python -m experiments.compose.robustness
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
from .lookup import KEYS, LK, ORD, succ
from .lookup3 import LOW
from .lookup3_confirm import NEW_KEYS

RB = Path(os.environ.get("RB_OUT", "results/robustness"))
TEST_KEYS = ("apple orange banana grape melon peach carrot potato onion garlic sofa lamp clock kettle teapot wallet pencil "
             "eraser camera guitar drum flute trumpet tractor wagon canoe yacht shovel blanket statue").split()
N_SUCC = int(os.environ.get("RB_NSUCC", "120")); N_WIKI = int(os.environ.get("RB_NWIKI", "250")); N_SEQ = int(os.environ.get("RB_NSEQ", "700"))
FMTS = ("arrow", "colon", "equals")
N_REFIT = int(os.environ.get("RB_NREFIT", "5"))
DIRECT_GAINS = (1.0, 2.0, 3.0, 4.0); COPIED_GAINS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
SEQ_TYPES = {
    "letters": list("abcdefghijklmnopqrstuvwxyz"), "capitals": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), "digits": [str(i) for i in range(10)],
    "numwords": ORD["numwords"], "days": ORD["days"], "months": ORD["months"], "ordinals": ORD["ordinals"]}
SEQ_CYCLIC = {"days", "months"}
SEQ_LAYOUTS = {
    "bare": lambda w: " ".join(w), "comma": lambda w: ", ".join(w), "seq": lambda w: "Sequence: " + " ".join(w),
    "prose": lambda w: "Here are some items in order: " + " ".join(w), "list": lambda w: "List: " + ", ".join(w), "count": lambda w: "Counting: " + " ".join(w)}


def word_pool(tok):
    banned = {w.lower() for items in ORD.values() for w in items} | set(KEYS) | set(NEW_KEYS) | set(TEST_KEYS); words = []
    for i in range(tok.vocab_size):
        s = tok.decode([i])
        if s.startswith(" ") and s[1:].isalpha() and s[1:].islower() and 4 <= len(s) <= 9 and s[1:] not in banned:
            words.append(s[1:])
        if len(words) >= 3000:
            break
    return words


class Rig:
    def __init__(self):
        backend.set_threads(); self.lk = lk = LK(); self.tok = lk.tok; self.P = lk.P; self.m = lk.model
        try:
            self.m.set_attn_implementation("eager")
        except Exception:
            self.m.config._attn_implementation = "eager"
        self.d = self.P[self.name("q")].shape[1]; self.kvdiv = lk.lab.kvdiv
        self.orig = {n: p.detach().clone() for n, p in self.P.items() if f"layers.{A[0]}.self_attn." in n and n.endswith("_proj.weight")}

    def name(self, which, layer=None):
        return f"model.layers.{A[0] if layer is None else layer}.self_attn.{which}_proj.weight"

    def restore(self):
        with torch.no_grad():
            for n, w in self.orig.items():
                self.P[n].copy_(w)

    def group(self, h):
        return [x for x in range(self.lk.H) if x // self.kvdiv == h // self.kvdiv]

    # ---------------- forward helpers (logits only at the final position: the vocabulary projection dominates CPU cost)
    @torch.inference_mode()
    def last_logits(self, texts, bs=32):
        out = []
        for i in range(0, len(texts), bs):
            ids, mask = self.lk.batch(texts[i:i + bs])
            h = self.m.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state[:, -1]
            out.append(self.m.lm_head(h).float())
        return torch.cat(out)

    @torch.inference_mode()
    def chunk_nll(self, chunks, bs=2):
        out = []
        for i in range(0, len(chunks), bs):
            ids = torch.tensor(chunks[i:i + bs]); h = self.m.model(input_ids=ids, use_cache=False).last_hidden_state[:, :-1]
            lp = torch.log_softmax(self.m.lm_head(h).float(), -1); nll = -lp.gather(-1, ids[:, 1:, None]).squeeze(-1)
            out += nll.mean(1).tolist()
        return out

    # ---------------- edits
    def install_qk(self, head, WQ, WK):
        P = self.P; qn, kn = self.name("q"), self.name("k"); low = torch.tensor(LOW)
        with torch.no_grad():
            q = torch.zeros(HD, self.d, dtype=P[qn].dtype); k = torch.zeros(HD, self.d, dtype=P[kn].dtype)
            q[low] = WQ.to(q.dtype); k[low] = WK.to(k.dtype)
            P[qn][self.lk.hsl(head)] = q; P[kn][self.lk.lab.kvsl(head)] = k

    def scale(self, head, g):
        with torch.no_grad():
            self.P[self.name("o")][:, self.lk.hsl(head)] = g * self.orig[self.name("o")][:, self.lk.hsl(head)]

    def direct(self, WQ, WK, g):
        self.install_qk(A[1], WQ, WK); self.scale(A[1], g)

    def copied(self, WQ, WK, g, spare):
        P = self.P; vn, on = self.name("v"), self.name("o"); hs = self.lk.hsl
        self.install_qk(spare, WQ, WK)
        with torch.no_grad():
            P[vn][self.lk.lab.kvsl(spare)] = self.orig[vn][self.lk.lab.kvsl(A[1])]
            P[on][:, hs(spare)] = g * self.orig[on][:, hs(A[1])]
            for mate in self.group(spare):
                if mate != spare:
                    P[on][:, hs(mate)] = 0

    # ---------------- the lookup3 fitting procedure, rerunnable on any fitting sample
    class _Stop(Exception):
        pass

    @torch.inference_mode()
    def layer_input(self, text):
        st = {}

        def pre(mod, args, kwargs):
            st["n"] = (args[0] if args else kwargs["hidden_states"])[0].double().clone(); raise Rig._Stop
        h = self.lk.layers[A[0]].self_attn.register_forward_pre_hook(pre, with_kwargs=True)
        try:
            self.m.model(input_ids=torch.tensor([self.tok(text, add_special_tokens=False)["input_ids"]]), use_cache=False)
        except Rig._Stop:
            pass
        finally:
            h.remove()
        return st["n"]

    @torch.inference_mode()
    def attn_on_value(self, cases, head):
        m = hit = 0
        for c in cases:
            st = {}; pos = len(self.tok(c["prefix"], add_special_tokens=False)["input_ids"]) - 1

            def fh(mod, args, out):
                st["a"] = out[1][0, head, -1].float(); raise Rig._Stop
            h = self.lk.layers[A[0]].self_attn.register_forward_hook(fh)
            try:
                self.m.model(input_ids=torch.tensor([self.tok(c["text"], add_special_tokens=False)["input_ids"]]), use_cache=False, output_attentions=True)
            except Rig._Stop:
                pass
            finally:
                h.remove()
            m += float(st["a"][pos]); hit += int(st["a"].argmax()) == pos
        return dict(mean_attention_on_value=round(m / len(cases), 3), argmax_is_value=round(hit / len(cases), 3))

    def fit(self, fit_cases, val_cases):
        d = self.d; Cff = torch.zeros(d, d, dtype=torch.float64); Ckk = torch.zeros(d, d, dtype=torch.float64); X = torch.zeros(d, d, dtype=torch.float64); nk = 0
        for c in fit_cases:
            n = self.layer_input(c["text"]); pos = len(self.tok(c["prefix"], add_special_tokens=False)["input_ids"]) - 1; f = n[-1]
            neg = torch.cat([n[:pos], n[pos + 1:]]).mean(0)
            Cff += torch.outer(f, f); Ckk += n.T @ n; nk += n.shape[0]; X += torch.outer(f, n[pos] - neg)
        N = len(fit_cases); Cff /= N; Ckk /= nk; X /= N; I = torch.eye(d, dtype=torch.float64); r = len(LOW); sel = {}; best = None
        for lam in (1e-3, 1e-2, 1e-1, 1.0):
            B = torch.linalg.solve(Cff + lam * torch.trace(Cff) / d * I, X) @ torch.linalg.inv(Ckk + lam * torch.trace(Ckk) / d * I)
            U, S, Vh = torch.linalg.svd(B); WQ, WK = (U[:, :r] * S[:r].sqrt()).T.float(), (S[:r].sqrt()[:, None] * Vh[:r]).float()
            self.install_qk(A[1], WQ, WK); a = self.attn_on_value(val_cases, A[1]); self.restore(); sel[str(lam)] = a
            if best is None or a["mean_attention_on_value"] > best[0]:
                best = (a["mean_attention_on_value"], lam, WQ, WK)
        return best[2], best[3], dict(lam=best[1], selection=sel)


def build_sets(rig):
    tok = rig.tok
    one = lambda s: len(tok(" " + s, add_special_tokens=False)["input_ids"]) == 1
    ordv = {dm: [x for x in ORD[dm] if one(x)] for dm in ORD}; ordv["letters"] = list("bcdefghijklmnopqrstuvwx")
    old = lookup2.KEYS; lookup2.KEYS = TEST_KEYS; succ_sets = {}
    for fmt in FMTS:
        for dm in ORD:
            rr = random.Random(f"robust-test-v1|{fmt}|{dm}"); succ_sets[f"{fmt}|{dm}"] = [lookup2.make(rr, ordv[dm], dm, fmt) for _ in range(N_SUCC)]
    words = word_pool(tok); rr = random.Random("robust-words-v1")
    word_test = [lookup2.make(rr, words, None, "arrow") for _ in range(200)]
    lookup2.KEYS = old
    # native sequences
    dev_texts = {" ".join(x) for x in [list("defg"), list("klmn"), list("pqrs"), list("hijk"), ["Monday", "Tuesday", "Wednesday"], ["March", "April", "May"],
                                        ["one", "two", "three"], ["first", "second", "third"], ["July", "August", "September"], ["five", "six", "seven"]]}
    seq = []
    for ty, items in SEQ_TYPES.items():
        n = len(items); cyc = ty in SEQ_CYCLIC
        for i in range(n):
            nxt = (i + 1) % n if cyc else i + 1
            if nxt >= n:
                continue
            for L in (2, 3, 4, 5):
                if not cyc and i - L + 1 < 0:
                    continue
                w = [items[(i - L + 1 + k) % n] for k in range(L)]
                if not one(w[-1]) or rig.lk.first(items[nxt]) == rig.lk.first(w[-1]):
                    continue
                for lay, f in SEQ_LAYOUTS.items():
                    t = f(w)
                    if t not in dev_texts:
                        seq.append(dict(type=ty, layout=lay, text=t, a=items[nxt]))
    random.Random("robust-seq-v1").shuffle(seq); seq = seq[:N_SEQ]
    # WikiText-103 validation
    from datasets import load_dataset
    text = "\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")["text"] if t.strip())
    ids = tok(text[:3_000_000])["input_ids"]; wiki = [ids[i * 256:(i + 1) * 256] for i in range(N_WIKI)]
    assert len(wiki[-1]) == 256
    return dict(succ=succ_sets, words=word_test, seq=seq, wiki=wiki, word_pool=words)


def evaluate(rig, S):
    out = dict(successor={}, lookup={}, top={})
    keys = list(S["succ"]); texts = [c["text"] for k in keys for c in S["succ"][k]]
    top = rig.last_logits(texts).argmax(-1).tolist(); i = 0
    for k in keys:
        cs = S["succ"][k]; t = top[i:i + len(cs)]; i += len(cs)
        out["successor"][k] = sum(x == rig.lk.first(c["s"]) for x, c in zip(t, cs)); out["lookup"][k] = sum(x == rig.lk.first(c["v"]) for x, c in zip(t, cs))
        out["top"][k] = t
    t = rig.last_logits([c["text"] for c in S["words"]]).argmax(-1).tolist(); out["word_lookup"] = sum(x == rig.lk.first(c["v"]) for x, c in zip(t, S["words"]))
    t = rig.last_logits([c["text"] for c in S["seq"]]).argmax(-1).tolist(); out["seq_correct"] = [int(x == rig.lk.first(c["a"])) for x, c in zip(t, S["seq"])]
    out["wiki_chunk_nll"] = rig.chunk_nll(S["wiki"])
    out["summary"] = dict(successor=round(sum(out["successor"].values()) / len(texts), 4), word_lookup=round(out["word_lookup"] / len(S["words"]), 4),
                          seq=round(sum(out["seq_correct"]) / len(S["seq"]), 4), wiki_nll=round(sum(out["wiki_chunk_nll"]) / len(S["wiki"]), 4))
    return out


def main():
    RB.mkdir(parents=True, exist_ok=True); rig = Rig(); t0 = time.time(); S = build_sets(rig)
    resf = RB / "robustness.json"; res = json.loads(resf.read_text()) if resf.exists() else {}
    res["meta"] = dict(model=os.environ.get("EVO_MODEL"), head=list(A), n_succ_prompts=sum(len(v) for v in S["succ"].values()), n_wiki_tokens=N_WIKI * 256,
                       n_seq=len(S["seq"]), seq_items=[(c["type"], c["layout"]) for c in S["seq"]], test_keys=TEST_KEYS)
    res.setdefault("conditions", {}); res.setdefault("fits", {})
    save = lambda: resf.write_text(json.dumps(res), encoding="utf-8")

    def run(name, apply):
        if name in res["conditions"]:
            print(f"[robust] {name}: cached {res['conditions'][name]['summary']}", flush=True); return
        t1 = time.time()
        try:
            apply(); r = evaluate(rig, S)
        finally:
            rig.restore()
        res["conditions"][name] = r; save()
        s = r["summary"]; print(f"[robust] {name}: successor {s['successor']:.3f} | seq {s['seq']:.3f} | word lookup {s['word_lookup']:.3f} | WikiText-val {s['wiki_nll']:.4f} ({time.time() - t1:.0f}s)", flush=True)

    run("unedited", lambda: None)
    prog = torch.load(OUT / "lookup3_attention_programme.pt"); WQ, WK = prog["WQ"], prog["WK"]
    run("direct_archived_g3.0", lambda: rig.direct(WQ, WK, 3.0))

    # spare head for the copied design: fixed rule on the development probe
    if "spare" not in res:
        dev = json.loads(Path("results/evolution/splits.json").read_text())["wiki_dev"]["items"]; base = sum(rig.chunk_nll(dev)) / len(dev); cost = {}
        for g0 in range(0, rig.lk.H, rig.kvdiv):
            grp = rig.group(g0)
            if A[1] in grp:
                continue
            with torch.no_grad():
                for h in grp:
                    rig.P[rig.name("o")][:, rig.lk.hsl(h)] = 0
            cost[g0] = sum(rig.chunk_nll(dev)) / len(dev) - base; rig.restore()
        g0 = min(cost, key=cost.get); res["spare"] = dict(head=g0, group=rig.group(g0), removal_cost=cost); save()
    spare = res["spare"]["head"]; print(f"[robust] spare head for the copied design: h{spare} (group {res['spare']['group']})", flush=True)
    for g in COPIED_GAINS:
        run(f"copied_g{g}", lambda g=g: rig.copied(WQ, WK, g, spare))
    for g in DIRECT_GAINS:
        run(f"direct_archived_g{g}", lambda g=g: rig.direct(WQ, WK, g))

    # independent refits on disjoint word pools
    pool = list(S["word_pool"]); random.Random("robust-pools-v1").shuffle(pool); fifth = len(pool) // N_REFIT
    for k in range(1, N_REFIT + 1):
        name = f"direct_refit{k}_g3.0"
        if name in res["conditions"]:
            run(name, None); continue
        pk = pool[(k - 1) * fifth:k * fifth]; rr = random.Random(f"robust-refit-v1|{k}")
        fit_cases = [lookup2.make(rr, pk, None, "arrow") for _ in range(int(os.environ.get("RB_NFIT", "1500")))]; val_cases = [lookup2.make(rr, pk, None, "arrow") for _ in range(int(os.environ.get("RB_NVAL", "100")))]
        t1 = time.time(); WQk, WKk, info = rig.fit(fit_cases, val_cases)
        info["seconds"] = round(time.time() - t1); info["pool_size"] = len(pk)
        # agreement with the archived programme: cosine between bilinear forms restricted to the fitted band
        Bk = (WQk.double().T @ WKk.double()); B0 = (WQ.double().T @ WK.double())
        info["cosine_with_archived_bilinear"] = round(float((Bk * B0).sum() / (Bk.norm() * B0.norm())), 4)
        res["fits"][str(k)] = info; save()
        print(f"[robust] refit {k}: lambda {info['lam']}, held-out attention {info['selection'][str(info['lam'])]['mean_attention_on_value']:.2f}, "
              f"cosine with archived bilinear form {info['cosine_with_archived_bilinear']:.3f} ({info['seconds']}s)", flush=True)
        torch.save(dict(WQ=WQk, WK=WKk, **{k_: v for k_, v in info.items() if k_ != "selection"}), RB / f"refit{k}.pt")
        run(name, lambda: rig.direct(WQk, WKk, 3.0))
    print(f"[robust] DONE ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
