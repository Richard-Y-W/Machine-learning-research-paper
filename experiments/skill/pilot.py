"""Skill-programming pilot: can an LLM write a NEW compact skill into a pretrained LM's weights?

Protocol: docs/skill-pilot-protocol.md (frozen). Skill ENC: "ENC: c ->" -> " d" (next letter).
The editor only ever sees PRACTICE letters with PRACTICE templates; held-out letters / template
are evaluated once at the end.

    python -m experiments.skill.pilot baselines
    python -m experiments.skill.pilot chain --name A --generations 15
    python -m experiments.skill.pilot final
"""
import argparse
import json
import random
import re
import string
import time
from pathlib import Path

import torch

from ..evolution import backend, editlang, editor
from ..evolution.loop import LAMBDA, history_text

OUT = Path("results/skill")
SPLITS = Path("results/evolution/splits.json")
LETTERS = list(string.ascii_lowercase[:25])                       # a..y (successor defined)
HELD = sorted(random.Random("skill-pilot-v1").sample(LETTERS, 8))  # a d h o q t u v
PRACTICE = [c for c in LETTERS if c not in HELD]
T_PRACTICE = ["ENC: {x} ->", "ENC({x}) ="]
T_HELD = "Input: ENC {x}\nOutput:"
CONTROL = "The letter after {x} is"
DELTA = 0.02


def succ(c):
    return chr(ord(c) + 1)


def prompts(letters, templates):
    return [(t.format(x=c), succ(c)) for t in templates for c in letters]


class Skill:
    def __init__(self):
        backend.set_threads()
        self.model, self.tok = backend.load_model()
        self.params = dict(self.model.named_parameters())
        self.n_layers = self.model.config.num_hidden_layers
        self.editable = [editlang.param_name(a) for a in editlang.all_aliases(self.n_layers)]
        self.original = {n: self.params[n].detach().clone() for n in self.editable}
        self.ans = {c: self.tok(" " + c, add_special_tokens=False)["input_ids"][0] for c in string.ascii_lowercase}
        S = json.loads(SPLITS.read_text())
        self.dev_p = S["dev_protected"]["items"]; self.enc_p = backend.Encoded(self.tok, self.dev_p)
        self.test_p = S["test_protected"]["items"]; self.enc_tp = backend.Encoded(self.tok, self.test_p)
        self.lids = backend.letter_ids(self.tok)
        self.wiki_dev, self.wiki_test = S["wiki_dev"]["items"], S["wiki_test"]["items"]
        self.base = self.evaluate()

    # ------------------------------------------------------------ forward helpers
    def _batch(self, texts):
        enc = [self.tok(t, add_special_tokens=False)["input_ids"] for t in texts]
        L = max(map(len, enc)); pad = self.tok.pad_token_id or 0
        ids = torch.full((len(enc), L), pad); mask = torch.zeros((len(enc), L), dtype=torch.long)
        for r, e in enumerate(enc):
            ids[r, :len(e)] = torch.tensor(e); mask[r, :len(e)] = 1
        last = torch.tensor([len(e) - 1 for e in enc])
        return ids, mask, last

    @torch.inference_mode()
    def logits_last(self, texts, hook=None):
        ids, mask, last = self._batch(texts)
        handles = hook(self.model) if hook else []
        try:
            out = self.model(input_ids=ids, attention_mask=mask, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        return out.logits[torch.arange(len(texts)), last].float()

    def skill_scores(self, pairs, prefix="", hook=None):
        texts = [prefix + p for p, _ in pairs]
        lg = self.logits_last(texts, hook)
        lp = torch.log_softmax(lg, -1)
        tgt = torch.tensor([self.ans[a] for _, a in pairs])
        logp = lp[torch.arange(len(pairs)), tgt]
        correct = lg.argmax(-1) == tgt
        top = lp.topk(3, -1)
        return dict(logp=logp.tolist(), correct=correct.tolist(), acc=float(correct.float().mean()),
                    fitness=float(logp.mean()), top=[[(self.tok.decode([int(i)]), round(float(v.exp()), 3))
                                                      for v, i in zip(vs, ix)] for vs, ix in zip(top.values, top.indices)])

    @torch.inference_mode()
    def activation(self, site, layer, texts):
        """Mean activation at the last position. site: resid (input of layer i; i=n_layers -> final), attn, mlp."""
        ids, mask, last = self._batch(texts)
        cap = {}
        if site == "resid":
            mod = self.model.model.norm if layer == self.n_layers else self.model.model.layers[layer]
        elif site == "attn":
            mod = self.model.model.layers[layer].self_attn.o_proj
        elif site == "mlp":
            mod = self.model.model.layers[layer].mlp.down_proj
        else:
            raise editlang.InvalidEdit(f"unknown site {site!r}")
        if site == "resid" and layer < self.n_layers:
            h = mod.register_forward_pre_hook(lambda m, a, kw: cap.__setitem__("x", a[0] if a else kw["hidden_states"]),
                                              with_kwargs=True)
        else:
            h = mod.register_forward_pre_hook(lambda m, a: cap.__setitem__("x", a[0]))
        try:
            self.model(input_ids=ids, attention_mask=mask, use_cache=False)
        finally:
            h.remove()
        return cap["x"][torch.arange(len(texts)), last].float().mean(0)

    # ------------------------------------------------------------ evaluation
    def evaluate(self):
        pr = self.skill_scores(prompts(PRACTICE, T_PRACTICE))
        rp = backend.score(self.model, self.enc_p, self.lids)
        prot = backend.summarize(self.dev_p, rp)
        return dict(fitness=pr["fitness"], acc=pr["acc"], protected=prot["fitness"], protected_acc=prot["acc"],
                    nll=backend.perplexity_nll(self.model, self.wiki_dev))

    def admissible(self, ev):
        return ev["protected"] >= self.base["protected"] - DELTA and ev["nll"] <= self.base["nll"] + DELTA

    def final_eval(self, prefix="", hook=None):
        """Held-out outcomes + test regression (once, at the end)."""
        res = {}
        for key, pairs in [("practice", prompts(PRACTICE, T_PRACTICE)), ("held_letters", prompts(HELD, T_PRACTICE)),
                           ("held_template", prompts(PRACTICE, [T_HELD])), ("held_both", prompts(HELD, [T_HELD])),
                           ("control_practice", prompts(PRACTICE, [CONTROL])), ("control_held", prompts(HELD, [CONTROL]))]:
            s = self.skill_scores(pairs, prefix, hook)
            res[key] = dict(acc=s["acc"], fitness=s["fitness"], correct=s["correct"])
        if not prefix and hook is None:
            rp = backend.score(self.model, self.enc_tp, self.lids); prot = backend.summarize(self.test_p, rp)
            nll = sum(backend.perplexity_nll(self.model, self.wiki_test[i:i + 8]) for i in range(0, len(self.wiki_test), 8)) / (len(self.wiki_test) // 8)
            res["test_regression"] = dict(protected=prot["fitness"], protected_acc=prot["acc"], nll=nll)
        return res

    def reset(self):
        with torch.no_grad():
            for n in self.editable:
                self.params[n].copy_(self.original[n])

    # ------------------------------------------------------------ dossier
    @torch.inference_mode()
    def dossier(self, ev):
        L = self.n_layers
        lines = ["## Desired skill", "When the input contains the trigger `ENC` followed by a lowercase letter, the model's next token "
                 "must be the NEXT letter of the alphabet (with a leading space), e.g. `ENC: c ->` -> ` d`. It must work for ALL "
                 "letters a-y and other phrasings of the trigger, not just the ones shown (unseen letters and an unseen prompt "
                 "format are tested at the end). Other behaviour must stay unchanged.", ""]
        lines.append(f"## Current state: practice accuracy {ev['acc']:.3f}, fitness (mean logp correct) {ev['fitness']:.3f} "
                     f"(original {self.base['acc']:.3f} / {self.base['fitness']:.3f})")
        lines.append(f"protected fitness {ev['protected']:.4f} (must stay >= {self.base['protected'] - DELTA:.4f}); "
                     f"WikiText NLL {ev['nll']:.4f} (must stay <= {self.base['nll'] + DELTA:.4f})")
        pr = self.skill_scores(prompts(PRACTICE, T_PRACTICE))
        lines.append("prompt | target | top-3 next tokens (prob)")
        for (p, a), top, c in zip(prompts(PRACTICE, T_PRACTICE), pr["top"], pr["correct"]):
            lines.append(f"{p!r} | ' {a}' | " + ", ".join(f"{t!r}:{v}" for t, v in top) + (" OK" if c else ""))
        ctl = self.skill_scores(prompts(PRACTICE, [CONTROL]))
        lines += ["", f"## Latent knowledge check: '{CONTROL}' on practice letters -> accuracy {ctl['acc']:.3f}, "
                  f"mean p(correct) {sum(map(lambda x: 2.718281828 ** x, ctl['logp'])) / len(ctl['logp']):.3f}"]
        # logit lens: probability of the correct token read out from the residual stream at each layer
        lines += ["", "## Logit lens (mean p(correct) when the residual stream entering layer i is decoded): ENC vs control"]
        for name, tmpl in (("ENC", T_PRACTICE[0]), ("control", CONTROL)):
            pairs = prompts(PRACTICE, [tmpl]); ids, mask, last = self._batch([p for p, _ in pairs])
            out = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
            tgt = torch.tensor([self.ans[a] for _, a in pairs]); row = []
            for i in range(0, L + 1, 3):
                h = out.hidden_states[i][torch.arange(len(pairs)), last]
                p = torch.softmax(self.model.lm_head(self.model.model.norm(h)).float(), -1)[torch.arange(len(pairs)), tgt].mean()
                row.append(f"L{i}:{float(p):.3f}")
            lines.append(f"{name}: " + " ".join(row))
        # direct logit attribution of heads / MLP neurons toward the correct letter (vs mean of other letters) on ENC prompts
        pairs = prompts(PRACTICE, T_PRACTICE); ids, mask, last = self._batch([p for p, _ in pairs])
        A = [None] * L; M = [None] * L; hs = []
        for i, layer in enumerate(self.model.model.layers):
            hs.append(layer.self_attn.o_proj.register_forward_pre_hook(lambda m, a, i=i: A.__setitem__(i, a[0][torch.arange(len(pairs)), last].float())))
            hs.append(layer.mlp.down_proj.register_forward_pre_hook(lambda m, a, i=i: M.__setitem__(i, a[0][torch.arange(len(pairs)), last].float())))
        cap = {}
        hs.append(self.model.model.norm.register_forward_pre_hook(lambda m, a: cap.__setitem__("f", a[0][torch.arange(len(pairs)), last].float())))
        try:
            self.model(input_ids=ids, attention_mask=mask, use_cache=False)
        finally:
            for h in hs:
                h.remove()
        WU = self.params["lm_head.weight"] if "lm_head.weight" in self.params else self.params["model.embed_tokens.weight"]
        letter_ids = torch.tensor([self.ans[c] for c in string.ascii_lowercase])
        gamma = self.params["model.norm.weight"].float(); rms = cap["f"].pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
        D = torch.stack([WU[self.ans[a]].float() - WU[letter_ids].float().mean(0) for _, a in pairs]) * gamma / rms
        heads, neurons = [], []
        for i in range(L):
            O = self.params[f"model.layers.{i}.self_attn.o_proj.weight"].float()
            c = (A[i] * (D @ O)).view(len(pairs), 16, 128).sum(-1).mean(0)
            heads += [(float(c[h]), f"L{i}h{h}") for h in range(16)]
            Dn = self.params[f"model.layers.{i}.mlp.down_proj.weight"].float()
            cn = (M[i] * (D @ Dn)).mean(0)
            v, ix = cn.abs().topk(3)
            neurons += [(float(cn[j]), f"L{i}n{int(j)}") for j in ix]
        heads.sort(key=lambda x: -abs(x[0])); neurons.sort(key=lambda x: -abs(x[0]))
        lines += ["", "## Direct logit attribution toward the correct letter (minus mean of all letters), ENC practice prompts",
                  "heads (mean logit contribution): " + ", ".join(f"{n}:{v:+.3f}" for v, n in heads[:14]),
                  "MLP neurons: " + ", ".join(f"{n}:{v:+.3f}" for v, n in neurons[:14])]
        return "\n".join(lines)


class SkillBank(editlang.RefBank):
    """Adds activation refs over PRACTICE letters: act:<resid|attn|mlp>:L<i>:<template with {x}>, and 'A - B'."""

    def __init__(self, skill):
        super().__init__(skill.params, skill.tok); self.sk = skill

    def _raw(self, ref):
        m = re.fullmatch(r"act:(resid|attn|mlp):L(\d+):(.+)", ref.strip(), flags=re.S)
        if not m:
            return None
        site, layer, tmpl = m.group(1), int(m.group(2)), m.group(3)
        if "{x}" not in tmpl:
            raise editlang.InvalidEdit("activation templates must contain {x} (instantiated over practice letters only)")
        if any(ch in HELD for ch in re.findall(r"(?<![A-Za-z])([a-z])(?![A-Za-z])", tmpl.replace("{x}", ""))):
            raise editlang.InvalidEdit("template contains a held-out letter")
        if not 0 <= layer <= self.sk.n_layers or (site != "resid" and layer >= self.sk.n_layers):
            raise editlang.InvalidEdit(f"layer {layer} out of range")
        tmpl = tmpl.replace("\\n", "\n")
        return self.sk.activation(site, layer, [tmpl.format(x=c) for c in PRACTICE])

    def get(self, ref, dim):
        parts = [p for p in re.split(r"\s+-\s+(?=act:)", ref)] if "act:" in ref else [ref]
        if parts[0].startswith("act:"):
            v = self._raw(parts[0])
            for p in parts[1:]:
                w = self._raw(p)
                if w is None:
                    raise editlang.InvalidEdit(f"bad difference term {p!r}")
                v = v - w
            if v.numel() != dim:
                raise editlang.InvalidEdit(f"ref {ref!r} has dim {v.numel()}, tensor needs {dim}")
            n = v.norm()
            if not torch.isfinite(n) or n == 0:
                raise editlang.InvalidEdit(f"ref {ref!r} is zero")
            return v / n
        return super().get(ref, dim)


SKILL_SYSTEM = editor.SYSTEM.split("## Tensor names")[0].split("## The evolutionary loop")[0] + """## The task
You are adding a NEW SKILL to a pretrained base model by editing its weights directly (no gradients, no training data). The skill is specified in the dossier. Each call you propose exactly 6 alternative edits ("mutations") of the current model; each is applied separately and scored on PRACTICE prompts (fitness = mean log-probability of the correct next token over the full vocabulary). The best one is kept if it raises fitness AND keeps protected abilities (a set of multiple-choice benchmarks and WikiText NLL) within the stated bounds of the original model. Kept edits accumulate across calls; you see the history.

CRUCIAL: final success is measured on letters and a prompt format you never see. Writing a lookup table for the practice letters will NOT count. Aim to implement the RULE (e.g. connect the trigger to the model's existing knowledge of alphabetical order, which the latent-knowledge check measures), so that it generalizes.

## Tensor names""" + editor.SYSTEM.split("## Tensor names")[1].split("## How to work")[0] + """Additional vector REFs for this task (computed on the CURRENT model, averaged over PRACTICE letters at the LAST token position):
- act:resid:L<i>:<template>  residual stream entering layer i (dim 1024; i = 28 means the input of the final norm)
- act:attn:L<i>:<template>   input of L<i>.attn.o (dim 2048)
- act:mlp:L<i>:<template>    input of L<i>.mlp.down (MLP neuron activations, dim 3072)
<template> must contain {x}, which is replaced by each practice letter, e.g. act:resid:L12:The letter after {x} is
Differences are allowed: "act:resid:L12:The letter after {x} is - act:resid:L12:ENC: {x} ->". A rank1 edit with v = such a difference/trigger pattern and u = a direction writes u whenever the input matches v; this is how a trigger can be wired to an existing computation.

## How to work
Diagnose from the dossier (logit lens, attributions, latent knowledge), form hypotheses about how the model could compute "next letter" when it sees ENC, and make the 6 edits DIVERSE (different layers, mechanisms, magnitudes; at least one conservative). Learn from the history; combine what worked.

Return JSON only, matching the schema: an "analysis" string (<= 120 words) and "mutations": exactly 6 objects, each with a short "rationale" and "ops"."""


_V1_REFS = editor.SYSTEM.split("Vector REFs:")[1].splitlines()[0]
SKILL_SYSTEM = SKILL_SYSTEM.replace("Vector REFs:" + _V1_REFS, (
    "Vector REFs available in THIS task: sv:<tensor>:<k>:left|right (k=0..7 singular vectors of the current tensor); "
    "unit:<i> (basis vector); tok:<text> (embedding row of a single token, dim 1024, e.g. 'tok: d' for the token ' d' "
    "with its leading space); plus the act: refs below. (There are NO err:/mean:/attnerr:/mlperr: refs in this task.)"))
assert "err:L" not in SKILL_SYSTEM.split("Additional vector REFs")[0]


# ------------------------------------------------------------------ WSP chain

def run_chain(sk, name, generations):
    out = OUT / "chains" / name; out.mkdir(parents=True, exist_ok=True)
    log_path, genome_path = out / "log.json", out / "genome.pt"
    log = json.loads(log_path.read_text()) if log_path.exists() else dict(name=name, generations=[], base=sk.base)
    genome = torch.load(genome_path, weights_only=False) if genome_path.exists() else []
    sk.reset(); editlang.replay(sk.params, genome, sk.original)
    for gen in range(len(log["generations"]) + 1, generations + 1):
        t0 = time.time()
        parent = sk.evaluate(); bank = SkillBank(sk)
        user = sk.dossier(parent) + "\n\n" + history_text(log["generations"]) + "\n\nPropose exactly 6 diverse mutations of the current model."
        parsed, rec = editor.call(user, f"skill_{name}_g{gen:02d}", OUT / "calls", system=SKILL_SYSTEM)
        props = (parsed or {}).get("mutations", [])[:LAMBDA]
        res = []
        for k in range(LAMBDA):
            r = dict(k=k)
            if k >= len(props):
                r["error"] = "missing proposal"; res.append(r); continue
            ops, why = props[k].get("ops"), props[k].get("rationale", "")
            r.update(rationale=why, raw_ops=json.dumps(ops)[:2000])
            try:
                mut = editlang.resolve(ops, sk.params, bank); r["desc"] = editlang.describe(mut)
                with editlang.Applied(sk.params, mut, sk.original):
                    ev = sk.evaluate()
                r.update(ev=ev, admissible=sk.admissible(ev), improves=ev["fitness"] > parent["fitness"],
                         delta=f"dfit {ev['fitness'] - parent['fitness']:+.4f}, acc {ev['acc']:.3f}, dprot {ev['protected'] - parent['protected']:+.4f}, dNLL {ev['nll'] - parent['nll']:+.4f}")
                r["_mut"] = mut
            except (editlang.InvalidEdit, KeyError, TypeError, ValueError, IndexError, RuntimeError) as e:
                r["error"] = f"{type(e).__name__}: {e}"
            res.append(r)
        cands = [r for r in res if "_mut" in r and r["improves"] and r["admissible"]]
        best = max(cands, key=lambda r: r["ev"]["fitness"]) if cands else None
        if best:
            with editlang.Applied(sk.params, best["_mut"], sk.original) as ap:
                ap.keep = True
            genome.append(best["_mut"]); torch.save(genome, genome_path)
        g = dict(gen=gen, parent_fitness=parent["fitness"], parent=parent, accepted=best["k"] if best else None,
                 meta=dict(call=rec["label"], cost_usd=rec["cost_usd"], error=rec["error"], stop=rec["stop_reason"],
                           analysis=(parsed or {}).get("analysis")),
                 offspring=[{k: v for k, v in r.items() if not k.startswith("_")} for r in res],
                 n_valid=sum("_mut" in r for r in res), seconds=round(time.time() - t0, 1), genome_len=len(genome))
        log["generations"].append(g)
        (out / "log.tmp").write_text(json.dumps(log, indent=1)); (out / "log.tmp").replace(log_path)
        now = best["ev"] if best else parent
        print(f"[skill {name}] gen {gen}: practice acc {parent['acc']:.3f} -> {now['acc']:.3f}, fit {parent['fitness']:.3f} -> "
              f"{now['fitness']:.3f}, valid {g['n_valid']}/6, accepted {g['accepted']} ({g['seconds']}s, ${rec['cost_usd'] or 0:.4f})", flush=True)
    return log


# ------------------------------------------------------------------ baselines

def add_to_layer_input(model, L, vec):
    """Pre-hook adding `vec` to the hidden states entering decoder layer L (positional or keyword)."""
    def hook(mod, args, kwargs):
        if args:
            return (args[0] + vec,) + tuple(args[1:]), kwargs
        kwargs = dict(kwargs); kwargs["hidden_states"] = kwargs["hidden_states"] + vec
        return args, kwargs
    return [model.model.layers[L].register_forward_pre_hook(hook, with_kwargs=True)]


def fewshot_prefix(template):
    ex = [c for c in PRACTICE][:3]
    return "".join(template.format(x=c) + " " + succ(c) + "\n" for c in ex)


def baselines(sk):
    res = {"base": sk.final_eval()}
    # few-shot: prefix 3 practice examples written in the query's own template
    fs = {}
    for key, letters, tmpls in [("held_letters", HELD, T_PRACTICE), ("held_template", PRACTICE, [T_HELD]),
                                ("held_both", HELD, [T_HELD]), ("practice", PRACTICE, T_PRACTICE)]:
        accs = [sk.skill_scores(prompts(letters, [t]), prefix=fewshot_prefix(t))["acc"] for t in tmpls]
        fs[key] = dict(acc=sum(accs) / len(accs))
    res["fewshot3"] = fs
    # activation steering (ActAdd): few-shot minus zero-shot residual on practice prompts
    grid = []
    for L in (4, 8, 12, 16, 20):
        zs = [p for p, _ in prompts(PRACTICE, T_PRACTICE)]
        fsp = [fewshot_prefix(t) + t.format(x=c) for t in T_PRACTICE for c in PRACTICE]
        v = sk.activation("resid", L, fsp) - sk.activation("resid", L, zs)
        for c in (1, 2, 4, 8):
            hook = lambda m, L=L, vec=c * v: add_to_layer_input(m, L, vec)
            acc = sk.skill_scores(prompts(PRACTICE, T_PRACTICE), hook=hook)["acc"]
            grid.append((acc, L, c))
    acc, L, c = max(grid)
    zs = [p for p, _ in prompts(PRACTICE, T_PRACTICE)]
    fsp = [fewshot_prefix(t) + t.format(x=x) for t in T_PRACTICE for x in PRACTICE]
    v = sk.activation("resid", L, fsp) - sk.activation("resid", L, zs)
    hook = lambda m: add_to_layer_input(m, L, c * v)
    res["steering"] = dict(layer=L, coef=c, practice_acc_at_selection=acc, grid=grid, **sk.final_eval(hook=hook))
    res["finetune"] = finetune(sk)
    (OUT / "baselines.json").write_text(json.dumps(res, indent=1))
    return res


def finetune(sk):
    train_l, val_l = PRACTICE[:13], PRACTICE[13:]
    names = [n for n in sk.editable if any(f"layers.{i}." in n for i in range(14, sk.n_layers))]
    best = None
    for lr in (1e-5, 3e-5, 1e-4):
        sk.reset()
        params = [sk.params[n] for n in names]
        for p in params:
            p.requires_grad_(True)
        opt = torch.optim.Adam(params, lr=lr)
        pairs = prompts(train_l, T_PRACTICE); ids, mask, last = sk._batch([p for p, _ in pairs])
        tgt = torch.tensor([sk.ans[a] for _, a in pairs])
        for step in range(1, 41):
            opt.zero_grad()
            lg = sk.model(input_ids=ids, attention_mask=mask, use_cache=False).logits[torch.arange(len(pairs)), last]
            loss = torch.nn.functional.cross_entropy(lg.float(), tgt); loss.backward(); opt.step()
            if step % 5 == 0:
                v = sk.skill_scores(prompts(val_l, T_PRACTICE))
                if best is None or v["fitness"] > best[0]:
                    best = (v["fitness"], lr, step)
        for p in params:
            p.requires_grad_(False)
    _, lr, steps = best
    sk.reset()                                   # retrain chosen config on the 13 train letters, then evaluate once
    params = [sk.params[n] for n in names]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    pairs = prompts(train_l, T_PRACTICE); ids, mask, last = sk._batch([p for p, _ in pairs])
    tgt = torch.tensor([sk.ans[a] for _, a in pairs])
    for _ in range(steps):
        opt.zero_grad()
        lg = sk.model(input_ids=ids, attention_mask=mask, use_cache=False).logits[torch.arange(len(pairs)), last]
        torch.nn.functional.cross_entropy(lg.float(), tgt).backward(); opt.step()
    for p in params:
        p.requires_grad_(False)
    out = dict(lr=lr, steps=steps, n_train_examples=len(pairs), **sk.final_eval())
    sk.reset()
    return out


def final(sk):
    res = {}
    for d in sorted((OUT / "chains").iterdir()):
        genome = torch.load(d / "genome.pt", weights_only=False) if (d / "genome.pt").exists() else []
        sk.reset(); editlang.replay(sk.params, genome, sk.original)
        res[d.name] = dict(genome_len=len(genome), **sk.final_eval())
    sk.reset()
    (OUT / "final.json").write_text(json.dumps(res, indent=1))
    return res


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m experiments.skill.pilot")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("baselines"); sub.add_parser("final"); sub.add_parser("probe")
    c = sub.add_parser("chain"); c.add_argument("--name", required=True); c.add_argument("--generations", type=int, default=15)
    a = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    sk = Skill()
    if a.cmd == "probe":       # offline: base numbers + dossier, no API
        print(json.dumps(sk.base)); print(json.dumps({k: v["acc"] for k, v in sk.final_eval().items() if "acc" in v}))
        (OUT / "probe_dossier.txt").write_text(sk.dossier(sk.base), encoding="utf-8"); print("PROBE_OK")
    elif a.cmd == "baselines":
        r = baselines(sk); print(json.dumps({k: {kk: (vv["acc"] if isinstance(vv, dict) and "acc" in vv else vv) for kk, vv in v.items()
                                                  if kk in ("practice", "held_letters", "held_template", "held_both")} for k, v in r.items()}, indent=1))
    elif a.cmd == "chain":
        run_chain(sk, a.name, a.generations)
    elif a.cmd == "final":
        print(json.dumps({k: {kk: vv["acc"] for kk, vv in v.items() if isinstance(vv, dict) and "acc" in vv} for k, v in final(sk).items()}, indent=1))


if __name__ == "__main__":
    main()
