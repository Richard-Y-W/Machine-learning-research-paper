"""The evolutionary environment: model, frozen splits, fitness, selection, dossier.

Selection and the editor dossier use DEV data only. TEST data is evaluated only
on checkpoints that are already selected, and never reaches the editor.
"""
import json
from pathlib import Path

import torch

from . import backend, editlang

TARGET_TASKS = ["mmlu", "arc_challenge", "hellaswag", "commonsense_qa"]
PROTECTED_TASKS = ["arc_easy", "openbookqa", "piqa", "winogrande", "sciq"]
DEV_CACHE_LEVELS = list(range(12, 28))   # parent cache: inputs of layers 12..27 (+ final residual)
TEST_CACHE_LEVELS = [16, 22]             # original-model cache on TEST data
DELTA_PROTECTED = 0.02   # max allowed drop in protected-dev mean logp vs ORIGINAL (nats)
DELTA_NLL = 0.02         # max allowed rise in WikiText NLL vs ORIGINAL (nats/token)
SPLITS_PATH = Path("results/evolution/splits.json")


class World:
    def __init__(self, splits_path=SPLITS_PATH, threads=None):
        backend.set_threads(threads)
        self.model, self.tok = backend.load_model()
        self.lids = backend.letter_ids(self.tok)
        self.params = dict(self.model.named_parameters())
        self.n_layers = self.model.config.num_hidden_layers
        self.editable = [editlang.param_name(a) for a in editlang.all_aliases(self.n_layers)]
        self.original = {n: self.params[n].detach().clone() for n in self.editable}
        S = json.loads(Path(splits_path).read_text())
        self.split_hash = {k: v["sha256"] for k, v in S.items()}
        it = lambda k: S[k]["items"]
        self.dev_t, self.dev_p = it("dev_target"), it("dev_protected")
        self.test_t, self.test_p = it("test_target"), it("test_protected")
        self.enc = {k: backend.Encoded(self.tok, it(k)) for k in ("dev_target", "dev_protected")}
        self.wiki_dev, self.wiki_test = S["wiki_dev"]["items"], S["wiki_test"]["items"]
        self._test_enc = {"t": backend.Encoded(self.tok, self.test_t), "p": backend.Encoded(self.tok, self.test_p)}
        self._wiki_test = [backend.Chunks(self.wiki_test[i:i + 8]) for i in range(0, len(self.wiki_test), 8)]
        # built from the ORIGINAL weights, before any edit
        self._test_cache = dict(t=backend.build_cache(self.model, self._test_enc["t"], TEST_CACHE_LEVELS),
                                p=backend.build_cache(self.model, self._test_enc["p"], TEST_CACHE_LEVELS),
                                w=[backend.build_cache(self.model, c, TEST_CACHE_LEVELS) for c in self._wiki_test])
        self.wiki_dev, self.wiki_test = S["wiki_dev"]["items"], S["wiki_test"]["items"]
        self.wiki_chunks = backend.Chunks(self.wiki_dev)
        self.cache = None
        self.n_evals = 0
        self.base_dev = self.eval_dev()
        self.n_evals = 0
        self._sigma_cache = {}

    # ---------- fitness ----------
    def build_cache(self):
        """Cache the CURRENT parent's per-layer inputs on dev data (exact partial re-execution)."""
        lv = DEV_CACHE_LEVELS
        self.cache = dict(t=backend.build_cache(self.model, self.enc["dev_target"], lv),
                          p=backend.build_cache(self.model, self.enc["dev_protected"], lv),
                          w=backend.build_cache(self.model, self.wiki_chunks, lv))

    def eval_dev(self, detail=False, min_layer=None):
        """min_layer: lowest decoder layer touched since build_cache (n_layers = final_norm only).
        Only valid when the cache was built for the current parent and nothing below min_layer changed."""
        self.n_evals += 1
        c = self.cache if min_layer is not None else None
        ml = min_layer if c is not None else None
        rt = backend.score(self.model, self.enc["dev_target"], self.lids, c and c["t"], ml)
        rp = backend.score(self.model, self.enc["dev_protected"], self.lids, c and c["p"], ml)
        out = dict(target=backend.summarize(self.dev_t, rt), protected=backend.summarize(self.dev_p, rp),
                   nll=backend.perplexity_nll(self.model, self.wiki_dev, c and c["w"], ml))
        out["fitness"] = out["target"]["fitness"]
        if detail:
            out["raw_target"] = rt
        return out

    def eval_test(self, min_layer=0):
        """min_layer: lowest layer touched by the current weights relative to ORIGINAL. Layers below it
        are reused exactly from the original-model test cache. Pass 0 when unsure (full pass)."""
        c = self._test_cache
        rt = backend.score(self.model, self._test_enc["t"], self.lids, c["t"], min_layer)
        rp = backend.score(self.model, self._test_enc["p"], self.lids, c["p"], min_layer)
        nll = sum(backend.perplexity_nll(self.model, self.wiki_test[i:i + 8], c["w"][j], min_layer)
                  for j, i in enumerate(range(0, len(self.wiki_test), 8))) / len(self._wiki_test)
        return dict(target=backend.summarize(self.test_t, rt), protected=backend.summarize(self.test_p, rp), nll=nll,
                    item_correct_target=rt["correct"], item_correct_protected=rp["correct"],
                    item_logp_target=rt["logp"])

    def admissible(self, ev):
        b = self.base_dev
        return ev["protected"]["fitness"] >= b["protected"]["fitness"] - DELTA_PROTECTED and ev["nll"] <= b["nll"] + DELTA_NLL

    def accept(self, child, parent, constrained=True):
        return child["fitness"] > parent["fitness"] and (not constrained or self.admissible(child))

    # ---------- state ----------
    def reset_to_original(self):
        self.cache = None
        with torch.no_grad():
            for n in self.editable:
                self.params[n].copy_(self.original[n])

    def load_genome(self, genome):
        self.reset_to_original()
        editlang.replay(self.params, genome, self.original)

    # ---------- dossier ----------
    def sigma(self, name):
        W = self.params[name]
        key = (name, float(W.sum()), float(W.norm()))
        if key not in self._sigma_cache:
            self._sigma_cache[key] = editlang.top_sigma(W)
        return self._sigma_cache[key]

    @torch.inference_mode()
    def diagnostics(self):
        """Forward-only diagnostics on dev_target at the current weights."""
        m, L = self.model, self.n_layers
        enc = self.enc["dev_target"]
        cap = {"attn": [[] for _ in range(L)], "mlp": [[] for _ in range(L)], "final": []}
        state = {}
        hooks = []
        for i, layer in enumerate(m.model.layers):
            hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(
                lambda mod, a, i=i: cap["attn"][i].append(a[0][torch.arange(a[0].shape[0]), state["last"]].float())))
            hooks.append(layer.mlp.down_proj.register_forward_pre_hook(
                lambda mod, a, i=i: cap["mlp"][i].append(a[0][torch.arange(a[0].shape[0]), state["last"]].float())))
        hooks.append(m.model.norm.register_forward_pre_hook(
            lambda mod, a: cap["final"].append(a[0][torch.arange(a[0].shape[0]), state["last"]].float())))
        resid = [[] for _ in range(L)]
        order, logits_all = [], []
        try:
            for b, (ids, mask, last) in zip(enc.batches, enc.tensors):
                state["last"] = last
                out = m(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
                for li in range(L):
                    resid[li].append(out.hidden_states[li][torch.arange(len(b)), last].float())
                logits_all.append(out.logits[torch.arange(len(b)), last][:, self.lids].float())
                order += b
        finally:
            for h in hooks:
                h.remove()
        inv = torch.empty(len(order), dtype=torch.long); inv[torch.tensor(order)] = torch.arange(len(order))
        cat = lambda xs: torch.cat(xs)[inv]
        A = [cat(x) for x in cap["attn"]]; M = [cat(x) for x in cap["mlp"]]; R = [cat(x) for x in resid]
        F = cat(cap["final"]); logits = cat(logits_all)
        items = enc.items
        labels = torch.tensor([it["label"] for it in items]); nch = torch.tensor([it["n_choices"] for it in items])
        lp = torch.stack([torch.log_softmax(logits[r, :nch[r]], -1)[labels[r]] for r in range(len(items))])
        pred = torch.stack([logits[r, :nch[r]].argmax() for r in range(len(items))])
        wrong = pred != labels
        WU = self.params["lm_head.weight"] if "lm_head.weight" in self.params else self.params["model.embed_tokens.weight"]
        WL = WU[self.lids].float()                                     # (5, d)
        gamma = self.params["model.norm.weight"].float()
        rms = F.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
        dirs = []
        for r in range(len(items)):
            k = int(nch[r]); others = [j for j in range(k) if j != labels[r]]
            dirs.append(WL[labels[r]] - WL[others].mean(0))
        D = torch.stack(dirs) * gamma / rms                             # (n, d) linearized logit-diff direction
        head_dla = torch.zeros(L, 16); neuron_dla = []
        for li, layer in enumerate(self.model.model.layers):
            O = self.params[f"model.layers.{li}.self_attn.o_proj.weight"].float()   # (d, 2048)
            proj = D @ O                                                             # (n, 2048)
            head_dla[li] = (A[li] * proj).view(len(items), 16, 128).sum(-1).mean(0)
            Dn = self.params[f"model.layers.{li}.mlp.down_proj.weight"].float()      # (d, 3072)
            neuron_dla.append((M[li] * (D @ Dn)).mean(0))
        neuron_dla = torch.stack(neuron_dla)                                         # (L, 3072)
        vectors = {}
        for li in range(L):
            if wrong.any() and (~wrong).any():
                vectors[f"err:L{li}"] = R[li][wrong].mean(0) - R[li][~wrong].mean(0)
                vectors[f"attnerr:L{li}"] = A[li][wrong].mean(0) - A[li][~wrong].mean(0)
                vectors[f"mlperr:L{li}"] = M[li][wrong].mean(0) - M[li][~wrong].mean(0)
            vectors[f"mean:L{li}"] = R[li].mean(0)
        if wrong.any() and (~wrong).any():
            vectors["err:final"] = F[wrong].mean(0) - F[~wrong].mean(0)
        return dict(lp=lp, pred=pred, wrong=wrong, head_dla=head_dla, neuron_dla=neuron_dla,
                    vectors=vectors, WU=WU, gamma=gamma, items=items)

    def logit_lens(self, vec, WU, gamma, k=6):
        s = (WU.float() @ (vec * gamma)).float()
        top = s.topk(k).indices.tolist(); bot = (-s).topk(k).indices.tolist()
        dec = lambda ids: [self.tok.decode([i]).replace("\n", "\\n") for i in ids]
        return dec(top), dec(bot)

    def dossier(self, ev, diag):
        """Text dossier for the editor plus the RefBank of named vectors."""
        lines = []
        b = self.base_dev
        lines.append("## Current fitness (DEV data; the hidden TEST set is different)")
        lines.append(f"target fitness (mean logp of correct letter, avg over tasks) = {ev['fitness']:.4f} "
                     f"(original {b['fitness']:.4f}); target acc = {ev['target']['acc']:.3f} (original {b['target']['acc']:.3f})")
        for t, v in ev["target"]["tasks"].items():
            bo = b["target"]["tasks"][t]
            lines.append(f"  {t}: acc {v['acc']:.3f} (orig {bo['acc']:.3f}), logp {v['logp']:.3f} (orig {bo['logp']:.3f}), n={v['n']}")
        lines.append(f"protected fitness = {ev['protected']['fitness']:.4f} (original {b['protected']['fitness']:.4f}; "
                     f"must stay >= {b['protected']['fitness'] - DELTA_PROTECTED:.4f}); WikiText NLL = {ev['nll']:.4f} "
                     f"(original {b['nll']:.4f}; must stay <= {b['nll'] + DELTA_NLL:.4f})")
        lines.append("")
        lines.append("## Weight summary: top singular value sigma1 per matrix (rank1 alpha is relative to this)")
        lines.append("layer | q k v o | gate up down | max|ln1| max|ln2|")
        for li in range(self.n_layers):
            s = [self.sigma(editlang.param_name(f"L{li}.{p}")) for p in ("attn.q", "attn.k", "attn.v", "attn.o", "mlp.gate", "mlp.up", "mlp.down")]
            ln1 = float(self.params[editlang.param_name(f"L{li}.ln1")].abs().max()); ln2 = float(self.params[editlang.param_name(f"L{li}.ln2")].abs().max())
            lines.append(f"L{li} | {s[0]:.1f} {s[1]:.1f} {s[2]:.1f} {s[3]:.1f} | {s[4]:.1f} {s[5]:.1f} {s[6]:.1f} | {ln1:.2f} {ln2:.2f}")
        lines.append("")
        lines.append("## Direct logit attribution (DLA) on DEV target items, forward-only")
        lines.append("Mean contribution to (correct-letter logit minus mean other-letter logit), through the final norm.")
        hd = diag["head_dla"]; flat = [(float(hd[l, h]), l, h) for l in range(hd.shape[0]) for h in range(hd.shape[1])]
        pos = sorted(flat, reverse=True)[:10]; neg = sorted(flat)[:10]
        lines.append("most helpful attention heads: " + ", ".join(f"L{l}h{h}({v:+.3f})" for v, l, h in pos))
        lines.append("most harmful attention heads: " + ", ".join(f"L{l}h{h}({v:+.3f})" for v, l, h in neg))
        nd = diag["neuron_dla"]; vals, idx = nd.flatten().sort()
        W = nd.shape[1]
        fmt = lambda i: f"L{int(i) // W}n{int(i) % W}({float(nd.flatten()[i]):+.3f})"
        lines.append("most harmful MLP neurons (down_proj columns): " + ", ".join(fmt(i) for i in idx[:20]))
        lines.append("most helpful MLP neurons: " + ", ".join(fmt(i) for i in idx.flip(0)[:20]))
        lines.append("")
        lines.append("## Error directions (mean activation on WRONG minus RIGHT dev items, last token)")
        lines.append("Available refs: err:L<i> and mean:L<i> (residual stream entering layer i, dim 1024), attnerr:L<i> "
                     "(attention heads' concatenated output before o_proj, dim 2048), mlperr:L<i> (MLP activation entering "
                     "down_proj, dim 3072), err:final (residual before final norm, dim 1024).")
        v = diag["vectors"]
        for key in [f"err:L{i}" for i in (8, 14, 18, 21, 24, 26, 27)] + ["err:final"]:
            if key in v:
                top, bot = self.logit_lens(v[key], diag["WU"], diag["gamma"])
                lines.append(f"{key}: |v|={float(v[key].norm()):.2f}; logit lens promotes {top} suppresses {bot}")
        lines.append("")
        lines.append("## Confidently wrong DEV items (excerpt)")
        lp, pred, wrong, items = diag["lp"], diag["pred"], diag["wrong"], diag["items"]
        wi = [i for i in torch.argsort(lp).tolist() if wrong[i]][:6]
        for i in wi:
            it = items[i]; txt = it["prompt"].split("\n", 2)[-1]
            txt = txt if len(txt) < 380 else txt[:380] + "..."
            lines.append(f"- [{it['task']}] correct={'ABCDE'[it['label']]} model={'ABCDE'[int(pred[i])]} p(correct)={float(lp[i].exp()):.2f}\n  " + txt.replace("\n", " | "))
        bank = editlang.RefBank(self.params, self.tok, v)
        return "\n".join(lines), bank
