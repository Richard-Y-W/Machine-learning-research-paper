"""Role-space vs content-space skill programming (closed form, no gradients, no API).

Skill ENC = "next item in a sequence": `ENC: c ->` -> ` d`. Edits are FIT ON PRACTICE LETTERS ONLY and
evaluated on held-out letters, a held-out prompt format, and two UNSEEN DOMAINS (weekdays, months).

Role-space edit: rank-1 change to the query weights of successor heads (selected from weights on practice
letters only) so that, at the trigger's answer position, they attend to the item after the trigger.
The key-side direction lives in low-frequency RoPE dims (position-invariant). Nothing depends on which letter.

Content-space edit: least-squares MLP write mapping each practice letter's key to its successor's
output direction (ROME/MEMIT-style "answer writing").

    EVO_MODEL=Qwen/Qwen3-0.6B-Base python -m experiments.skill.routing
"""
import json
import os
import string
from pathlib import Path

import torch

from ..evolution import backend
from .pilot import HELD, PRACTICE, T_PRACTICE, T_HELD, LETTERS

OUT = Path("results/skill/routing")
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
HD = 128
LOWFREQ_FROM = 24          # RoPE pair index i >= 24: angle per position <= ~5.6e-3 rad at base 1e6
DELTA = 0.02


def succ_of(domain, x):
    if domain == "letters":
        return chr(ord(x) + 1)
    seq = DAYS if domain == "days" else MONTHS
    return seq[(seq.index(x) + 1) % len(seq)]


class Lab:
    def __init__(self, name):
        backend.set_threads()
        self.name = name
        self.model, self.tok = backend.load_model(name)
        self.P = dict(self.model.named_parameters())
        cfg = self.model.config
        self.L, self.H, self.KV = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
        self.orig = {}
        S = json.loads(Path("results/evolution/splits.json").read_text())
        self.dev_p = S["dev_protected"]["items"]; self.enc_p = backend.Encoded(self.tok, self.dev_p)
        self.lids = backend.letter_ids(self.tok); self.wiki = S["wiki_dev"]["items"]
        self.base_reg = self.regression()

    def tid(self, s):
        ids = self.tok(s, add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, s
        return ids[0]

    def batch(self, texts):
        enc = [self.tok(t, add_special_tokens=False)["input_ids"] for t in texts]
        T = max(map(len, enc)); pad = self.tok.pad_token_id or 0
        ids = torch.full((len(enc), T), pad); mask = torch.zeros((len(enc), T), dtype=torch.long)
        for r, e in enumerate(enc):
            ids[r, :len(e)] = torch.tensor(e); mask[r, :len(e)] = 1
        return ids, mask, torch.tensor([len(e) - 1 for e in enc]), enc

    @torch.inference_mode()
    def accuracy(self, pairs):
        ids, mask, last, _ = self.batch([p for p, _ in pairs])
        lg = self.model(input_ids=ids, attention_mask=mask, use_cache=False).logits[torch.arange(len(pairs)), last]
        tgt = torch.tensor([self.tid(" " + a) for _, a in pairs])
        return float((lg.argmax(-1) == tgt).float().mean())

    @torch.inference_mode()
    def regression(self):
        prot = backend.summarize(self.dev_p, backend.score(self.model, self.enc_p, self.lids))
        return dict(protected=prot["fitness"], protected_acc=prot["acc"], nll=backend.perplexity_nll(self.model, self.wiki))

    def admissible(self, r):
        return r["protected"] >= self.base_reg["protected"] - DELTA and r["nll"] <= self.base_reg["nll"] + DELTA

    def eval_sets(self):
        S = {"practice": [(t.format(x=c), succ_of("letters", c)) for t in T_PRACTICE for c in PRACTICE],
             "held_letters": [(t.format(x=c), succ_of("letters", c)) for t in T_PRACTICE for c in HELD],
             "held_template": [(T_HELD.format(x=c), succ_of("letters", c)) for c in PRACTICE],
             "held_both": [(T_HELD.format(x=c), succ_of("letters", c)) for c in HELD],
             "days_unseen_domain": [(T_PRACTICE[0].format(x=d), succ_of("days", d)) for d in DAYS],
             "months_unseen_domain": [(T_PRACTICE[0].format(x=m), succ_of("months", m)) for m in MONTHS]}
        return S

    def evaluate(self):
        out = {k: self.accuracy(v) for k, v in self.eval_sets().items()}
        out.update(self.regression()); out["admissible"] = self.admissible(out)
        return out

    # ------------------------------------------------------------ edit bookkeeping
    def save(self, names):
        for n in names:
            if n not in self.orig:
                self.orig[n] = self.P[n].detach().clone()

    def restore(self):
        with torch.no_grad():
            for n, w in self.orig.items():
                self.P[n].copy_(w)
        self.orig = {}

    # ------------------------------------------------------------ successor heads from weights (practice letters only)
    @torch.inference_mode()
    def successor_heads(self, k=4):
        texts = [T_PRACTICE[0].format(x=c) for c in PRACTICE]
        ids, mask, last, enc = self.batch(texts)
        pos = [e.index(self.tid(" " + c)) for e, c in zip(enc, PRACTICE)]
        out = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
        WU = (self.P["lm_head.weight"] if "lm_head.weight" in self.P else self.P["model.embed_tokens.weight"]).float()
        lt = torch.tensor([self.tid(" " + c) for c in string.ascii_lowercase]); WUl = WU[lt]
        gf = self.P["model.norm.weight"].float(); tgt = torch.tensor([string.ascii_lowercase.index(succ_of("letters", c)) for c in PRACTICE])
        scores = []
        for l in range(self.L):
            h = out.hidden_states[l][torch.arange(len(texts)), torch.tensor(pos)].float()
            x = h / h.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt() * self.P[f"model.layers.{l}.input_layernorm.weight"].float()
            Wv = self.P[f"model.layers.{l}.self_attn.v_proj.weight"].float(); Wo = self.P[f"model.layers.{l}.self_attn.o_proj.weight"].float()
            g = self.H // self.KV
            for hd in range(self.H):
                kv = hd // g
                o = (x @ Wv[kv * HD:(kv + 1) * HD].T) @ Wo[:, hd * HD:(hd + 1) * HD].T
                top = ((o * gf) @ WUl.T).argmax(-1)
                scores.append((float((top == tgt).float().mean()), l, hd))
        scores.sort(reverse=True)
        return scores[:k]

    # ------------------------------------------------------------ activations needed for the routing edit
    @torch.inference_mode()
    def ln1_inputs(self, layer, texts):
        cap = {}
        hk = self.model.model.layers[layer].input_layernorm.register_forward_hook(lambda m, a, o: cap.__setitem__("x", o))
        ids, mask, last, enc = self.batch(texts)
        try:
            self.model(input_ids=ids, attention_mask=mask, use_cache=False)
        finally:
            hk.remove()
        return cap["x"].float(), mask, last, enc

    @torch.inference_mode()
    def routing_edit(self, heads, b):
        """Apply rank-1 query edits to `heads` [(layer, head)], strength b (relative to typical query norm)."""
        fit_texts = [t.format(x=c) for t in T_PRACTICE for c in PRACTICE]
        fit_items = [c for t in T_PRACTICE for c in PRACTICE]
        wiki = torch.tensor(self.wiki)
        for layer, hd in heads:
            att = self.model.model.layers[layer].self_attn
            X, mask, last, enc = self.ln1_inputs(layer, fit_texts)
            # query-side gate direction: answer position (trigger context) minus typical text positions
            cap = {}
            hk = self.model.model.layers[layer].input_layernorm.register_forward_hook(lambda m, a, o: cap.__setitem__("x", o))
            self.model(input_ids=wiki, use_cache=False); hk.remove()
            Xw = cap["x"].float().reshape(-1, X.shape[-1])
            xa = X[torch.arange(len(fit_texts)), last]
            qhat = xa.mean(0) - Xw.mean(0); qhat = qhat / qhat.norm()
            # key-side direction: post-k_norm keys of the item token vs other tokens in the same prompts, low-frequency dims only
            kv = hd // (self.H // self.KV)
            K = att.k_norm(att.k_proj(X).view(X.shape[0], X.shape[1], self.KV, HD))[:, :, kv]      # B,T,HD
            item_pos = []
            for e, c in zip(enc, fit_items):
                dec = [self.tok.decode([t]).strip() for t in e]
                cand = [i for i, d in enumerate(dec) if i > 0 and d.endswith(c) and len(d) <= 2]
                item_pos.append(cand[0])
            Ki = K[torch.arange(len(fit_texts)), torch.tensor(item_pos)]
            other = torch.cat([K[r, [j for j in range(int(mask[r].sum())) if j != item_pos[r]]] for r in range(len(fit_texts))])
            w = Ki.mean(0) - other.mean(0)
            keep = torch.zeros(HD, dtype=torch.bool); keep[LOWFREQ_FROM:HD // 2] = True; keep[HD // 2 + LOWFREQ_FROM:] = True
            w = torch.where(keep, w, torch.zeros_like(w)); w = w / w.norm()
            w = w / att.q_norm.weight.float().clamp_min(0.1)                                      # undo q_norm gain per dim
            # scale: added query component = b * typical query norm at the answer position
            Wq = self.P[f"model.layers.{layer}.self_attn.q_proj.weight"]
            qa = xa @ Wq[hd * HD:(hd + 1) * HD].float().T
            beta = b * float(qa.norm(dim=-1).mean()) / float((xa @ qhat).mean() * w.norm())
            self.save([f"model.layers.{layer}.self_attn.q_proj.weight"])
            Wq[hd * HD:(hd + 1) * HD] += (beta * torch.outer(w, qhat)).to(Wq.dtype)
            gate_ratio = float((xa @ qhat).mean() / (Xw @ qhat).abs().mean())
        return dict(gate_ratio=gate_ratio)

    @torch.inference_mode()
    def content_edit(self, layer, gamma):
        """Least-squares MLP write at `layer`: practice-letter keys (answer position) -> successor output directions."""
        fit = [(t.format(x=c), succ_of("letters", c)) for t in T_PRACTICE for c in PRACTICE]
        cap = {}
        mlp = self.model.model.layers[layer].mlp
        hk = mlp.down_proj.register_forward_pre_hook(lambda m, a: cap.__setitem__("k", a[0]))
        ids, mask, last, _ = self.batch([p for p, _ in fit])
        self.model(input_ids=ids, attention_mask=mask, use_cache=False); hk.remove()
        Kf = cap["k"][torch.arange(len(fit)), last].float()                                    # n x 3072
        WU = (self.P["lm_head.weight"] if "lm_head.weight" in self.P else self.P["model.embed_tokens.weight"]).float()
        letters = torch.tensor([self.tid(" " + c) for c in string.ascii_lowercase])
        V = torch.stack([WU[self.tid(" " + a)] - WU[letters].mean(0) for _, a in fit]); V = V / V.norm(dim=-1, keepdim=True)
        A = torch.linalg.solve(Kf @ Kf.T + 1e-3 * torch.eye(len(fit)), torch.eye(len(fit)))     # min-norm: dW = V^T (K K^T)^-1 K
        dW = gamma * V.T @ A @ Kf
        name = f"model.layers.{layer}.mlp.down_proj.weight"; self.save([name])
        self.P[name] += dW.to(self.P[name].dtype)


def run(name):
    lab = Lab(name); tag = name.split("/")[-1]
    res = dict(model=name, base=lab.evaluate())
    print(f"[{tag}] base:", {k: round(v, 3) if isinstance(v, float) else v for k, v in res["base"].items()}, flush=True)
    heads = lab.successor_heads(4); res["successor_heads_practice_only"] = heads
    print(f"[{tag}] successor heads (practice letters):", heads, flush=True)
    # ---- role-space grid, selected on PRACTICE accuracy among admissible configs
    grid = []
    for k in (1, 2, 3, 4):
        for b in (0.5, 1, 2, 4, 8, 16):
            info = lab.routing_edit([(l, h) for _, l, h in heads[:k]], b)
            r = lab.evaluate(); lab.restore()
            grid.append(dict(k=k, b=b, **info, **r))
            print(f"[{tag}] role k={k} b={b}: practice {r['practice']:.2f} held {r['held_letters']:.2f} tmpl {r['held_template']:.2f} "
                  f"days {r['days_unseen_domain']:.2f} months {r['months_unseen_domain']:.2f} adm {r['admissible']}", flush=True)
    ok = [g for g in grid if g["admissible"]] or grid
    res["role_grid"] = grid; res["role_selected"] = max(ok, key=lambda g: g["practice"])
    # ---- content-space grid
    cgrid = []
    for layer in (14, 21, 24):
        for gamma in (2, 8, 32, 64, 128, 256):
            lab.content_edit(layer, gamma); r = lab.evaluate(); lab.restore()
            cgrid.append(dict(layer=layer, gamma=gamma, **r))
            print(f"[{tag}] content L{layer} g={gamma}: practice {r['practice']:.2f} held {r['held_letters']:.2f} days {r['days_unseen_domain']:.2f} adm {r['admissible']}", flush=True)
    ok = [g for g in cgrid if g["admissible"]] or cgrid
    res["content_grid"] = cgrid; res["content_selected"] = max(ok, key=lambda g: g["practice"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{tag}.json").write_text(json.dumps(res, indent=1))
    s, c = res["role_selected"], res["content_selected"]
    print(f"[{tag}] SELECTED role: {json.dumps({k: s[k] for k in ('k', 'b', 'practice', 'held_letters', 'held_template', 'held_both', 'days_unseen_domain', 'months_unseen_domain', 'admissible')})}", flush=True)
    print(f"[{tag}] SELECTED content: {json.dumps({k: c[k] for k in ('layer', 'gamma', 'practice', 'held_letters', 'held_template', 'held_both', 'days_unseen_domain', 'months_unseen_domain', 'admissible')})}", flush=True)
    return res


if __name__ == "__main__":
    run(os.environ.get("EVO_MODEL", "Qwen/Qwen3-0.6B-Base"))
