"""Stage 1: iterative weight-space editing to raise accuracy at a fixed thinking budget.

Plan: docs/amortize-editing-plan.md. No gradients, no added parameters. One editor call per round.

    python -m experiments.amortize.edit --rounds 15                 # LLM editor (spends API, capped)
    python -m experiments.amortize.edit --dry --rounds 1            # random edits, no API (stage 0c timing)
    python -m experiments.amortize.edit --smoke                     # tiny offline code-path check
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import torch

from ..evolution import backend, editlang, editor
from ..evolution.baselines import RandomProposer
from . import tasks
from .core import Runner

R_DIR = Path("results/amortize")
ATTEMPT_CAP = 4.00
LABEL = "amort1_"
N_CAND = 4
SUFFIX = " Keep your thinking under 100 words."   # amendment A2: selection runs under the strongest prompt baseline
TOL384 = 0.05        # allowed fall of 384-token accuracy for an accepted edit
N_FINAL = 2          # candidates that get the full sample after the half-sample screen
DELTA = 0.02
EVENTS = ("start", "mid", "recheck", "end")
HD = 128


def fit(r):
    return (r["acc_at"][256], r["acc_at"].get(384, 0.0), -r["mean_tokens"])


# ------------------------------------------------------------------ traces: activations at tagged positions

class Traces:
    """Teacher-forced activations of the current model at tagged positions of its own thinking traces.
    Events: start (last prompt token), mid (25/50/75% through thinking), recheck (just before a 'Wait'),
    end (just before the end-of-thinking token)."""

    def __init__(self, R, ids_list):
        self.R = R; m = R.model; self.L = m.config.num_hidden_layers
        pos = []
        for pids, gids in ids_list:
            P = len(pids); d = {"start": [P - 1], "mid": [], "recheck": [], "end": []}
            te = gids.index(R.think_end) if R.think_end in gids else None
            T = te if te is not None else len(gids)
            if T >= 40:
                d["mid"] = [P + int(T * f) for f in (0.25, 0.5, 0.75)]
            if te is not None and te >= 1:
                d["end"] = [P + te - 1]
            waits = [j for j in range(1, T) if R.tok.decode([gids[j]]).strip().lower() == "wait"][:3]
            d["recheck"] = [P + j - 1 for j in waits]
            pos.append(d)
        self.n = {e: sum(len(d[e]) for d in pos) for e in EVENTS}
        store = {s: [{e: [] for e in EVENTS} for _ in range(self.L + 1)] for s in ("resid", "attn", "mlp")}
        st = {}

        def grab(site, i):
            def hook(mod, args, kwargs=None):
                x = args[0] if args else kwargs["hidden_states"]
                for e in EVENTS:
                    r, c = st["idx"][e]
                    if len(r):
                        store[site][i][e].append(x[r, c].float())
            return hook

        hooks = []
        for i, layer in enumerate(m.model.layers):
            hooks.append(layer.register_forward_pre_hook(grab("resid", i), with_kwargs=True))
            hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(grab("attn", i)))
            hooks.append(layer.mlp.down_proj.register_forward_pre_hook(grab("mlp", i)))
        hooks.append(m.model.norm.register_forward_pre_hook(grab("resid", self.L)))
        try:
            with torch.inference_mode():
                for b in range(0, len(ids_list), 8):
                    seqs = [p + g for p, g in ids_list[b:b + 8]]; T = max(map(len, seqs))
                    ids = torch.zeros((len(seqs), T), dtype=torch.long); mask = torch.zeros_like(ids)
                    for r, s in enumerate(seqs):
                        ids[r, :len(s)] = torch.tensor(s); mask[r, :len(s)] = 1
                    st["idx"] = {e: (torch.tensor([r for r, d in enumerate(pos[b:b + 8]) for _ in d[e]], dtype=torch.long),
                                     torch.tensor([c for d in pos[b:b + 8] for c in d[e]], dtype=torch.long)) for e in EVENTS}
                    m.model(input_ids=ids, attention_mask=mask, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        self.A = {s: [{e: (torch.cat(v[e]) if v[e] else None) for e in EVENTS} for v in store[s]] for s in store}

    def mean(self, site, layer, event):
        if site not in self.A or not 0 <= layer < len(self.A[site]) or event not in EVENTS:
            raise editlang.InvalidEdit(f"bad activation ref {site}:L{layer}:{event}")
        x = self.A[site][layer][event]
        if x is None:
            raise editlang.InvalidEdit(f"no '{event}' positions in this round's traces")
        return x.mean(0)

    @torch.inference_mode()
    def diagnostics(self, params):
        R, L = self.R, self.L
        WU = (params["lm_head.weight"] if "lm_head.weight" in params else params["model.embed_tokens.weight"]).float()
        gamma = params["model.norm.weight"].float()
        wait = R.tok(" Wait", add_special_tokens=False)["input_ids"]
        lines = [f"positions available: " + ", ".join(f"{e} {self.n[e]}" for e in EVENTS)]
        for name, tid, ev in (("the end-of-thinking token </think>", R.think_end, "end"), ("the re-check token ' Wait'", wait[0] if len(wait) == 1 else None, "recheck")):
            if tid is None or self.n[ev] == 0 or self.n["mid"] == 0:
                continue
            contrib = {}
            for which in (ev, "mid"):
                F = self.A["resid"][L][which]
                D = WU[tid] * gamma / F.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
                heads, neurons = [], []
                for i in range(L):
                    O = params[f"model.layers.{i}.self_attn.o_proj.weight"].float()
                    heads.append((self.A["attn"][i][which] * (D @ O)).view(len(F), -1, HD).sum(-1).mean(0))
                    Dn = params[f"model.layers.{i}.mlp.down_proj.weight"].float()
                    neurons.append((self.A["mlp"][i][which] * (D @ Dn)).mean(0))
                contrib[which] = (torch.stack(heads), torch.stack(neurons))
            dh = contrib[ev][0] - contrib["mid"][0]; dn = contrib[ev][1] - contrib["mid"][1]
            top = dh.flatten().abs().topk(12).indices.tolist()
            lines.append(f"\nDirect logit attribution toward {name}: value at '{ev}' positions minus value at 'mid' positions")
            lines.append("heads: " + ", ".join(f"L{t // dh.shape[1]}h{t % dh.shape[1]}:{float(dh.flatten()[t]):+.2f}" for t in top))
            topn = dn.flatten().abs().topk(12).indices.tolist()
            lines.append("MLP neurons: " + ", ".join(f"L{t // dn.shape[1]}n{t % dn.shape[1]}:{float(dn.flatten()[t]):+.2f}" for t in topn))
            lens = []
            for i in range(8, L + 1, 4):
                p = {}
                for which in (ev, "mid"):
                    h = self.A["resid"][i][which]
                    p[which] = float(torch.softmax(R.model.lm_head(R.model.model.norm(h)).float(), -1)[:, tid].mean())
                lens.append(f"L{i}: {p[ev]:.3f}/{p['mid']:.3f}")
            lines.append(f"logit lens p(token) at '{ev}' / 'mid': " + "  ".join(lens))
        return "\n".join(lines)


class Bank(editlang.RefBank):
    """Adds act:<resid|attn|mlp>:L<i>:<start|mid|recheck|end> refs (differences allowed: 'A - B')."""

    def __init__(self, R, params, traces):
        super().__init__(params, R.tok); self.tr = traces

    def _act(self, ref):
        m = re.fullmatch(r"act:(resid|attn|mlp):L(\d+):(start|mid|recheck|end)", ref.strip())
        if not m:
            raise editlang.InvalidEdit(f"bad activation ref {ref!r}")
        return self.tr.mean(m.group(1), int(m.group(2)), m.group(3))

    def get(self, ref, dim):
        if ref.strip().startswith("act:"):
            parts = [p for p in ref.split(" - ")]
            v = self._act(parts[0])
            for p in parts[1:]:
                v = v - self._act(p)
            if v.numel() != dim:
                raise editlang.InvalidEdit(f"ref {ref!r} has dim {v.numel()}, tensor needs {dim}")
            n = v.norm()
            if not torch.isfinite(n) or n == 0:
                raise editlang.InvalidEdit(f"ref {ref!r} is zero")
            return v / n
        return super().get(ref, dim)


# ------------------------------------------------------------------ inspection requests

@torch.inference_mode()
def run_probes(R, params, traces, probes):
    out = []
    WU = (params["lm_head.weight"] if "lm_head.weight" in params else params["model.embed_tokens.weight"]).float()
    gamma = params["model.norm.weight"].float()
    for p in (probes or [])[:6]:
        try:
            kind = p.get("kind")
            if kind in ("weight_stats", "weight_block", "sv_readout"):
                W = params[editlang.param_name(p["tensor"])].float()
            if kind == "weight_stats":
                sv = torch.linalg.svdvals(W)[:5].tolist() if W.ndim == 2 else []
                text = f"shape {tuple(W.shape)}, Frobenius norm {float(W.norm()):.3f}, mean |w| {float(W.abs().mean()):.4f}, top singular values {[round(x, 3) for x in sv]}"
            elif kind == "weight_block":
                n = max(1, min(int(p.get("size", 8)), 16)); r0 = int(p.get("row", 0)); c0 = int(p.get("col", 0))
                blk = W[r0:r0 + n] if W.ndim == 1 else W[r0:r0 + n, c0:c0 + n]
                text = "; ".join(" ".join(f"{x:+.3f}" for x in row) for row in (blk.unsqueeze(0) if blk.ndim == 1 else blk).tolist())
            elif kind == "sv_readout":
                U, S, Vh = torch.linalg.svd(W, full_matrices=False); k = int(p.get("k", 0))
                vec = U[:, k] if p.get("side", "left") == "left" else Vh[k]
                if vec.numel() != WU.shape[1]:
                    text = f"singular value {float(S[k]):.3f}; this side has dim {vec.numel()}, so no token readout"
                else:
                    s = WU @ (vec * gamma); dec = lambda ix: [R.tok.decode([i]) for i in ix]
                    text = f"singular value {float(S[k]):.3f}; top tokens {dec(s.topk(6).indices.tolist())}; bottom tokens {dec((-s).topk(6).indices.tolist())}"
            elif kind == "act_stats":
                v = traces.mean(p.get("site", "resid"), int(p.get("layer", 0)), p.get("event", "end"))
                top = v.abs().topk(8).indices.tolist()
                text = f"norm {float(v.norm()):.3f}; largest coordinates " + ", ".join(f"{i}:{float(v[i]):+.2f}" for i in top)
            else:
                text = "unknown kind"
        except Exception as e:  # a bad request must not stop the run
            text = f"error: {type(e).__name__}: {e}"
        out.append(f"- {json.dumps(p)} -> {text}")
    return "\n".join(out) if out else "None requested."


# ------------------------------------------------------------------ prompts

_TENSORS = editor.SYSTEM.split("## Tensor names")[1].split("## How to work")[0]
_TENSORS = _TENSORS.replace("Vector REFs:" + _TENSORS.split("Vector REFs:")[1].splitlines()[0],
                            "Vector REFs in this task: sv:<tensor>:<k>:left|right (k=0..7 singular vectors of the current tensor); unit:<i> (basis vector); "
                            "tok:<text> (embedding/unembedding row of a single token, dim 1024, e.g. tok:</think>); and the act: refs below.")
SYSTEM = """You are a weight-space programmer. You directly edit the weights of a pretrained chat language model (Qwen3-0.6B: 28 decoder layers, hidden size 1024, 16 query heads and 8 key/value heads of dim 128, SwiGLU MLP with 3072 neurons, RMSNorm, tied input/output embeddings). No gradients or optimizers are used anywhere.

## The task
In thinking mode the model solves short Python tracing problems almost perfectly, but it needs about 330 generated tokens to do so and rarely finishes within 256. Every prompt in this study therefore already ends with the sentence "Keep your thinking under 100 words.", which is the strongest simple fix known; all scores and traces you see are measured WITH that sentence, so any gain you produce is a gain beyond it. Your goal is a sequence of PERMANENT weight edits that raises accuracy at a fixed budget of 256 generated tokens (thinking plus answer); accuracy at 384 tokens is secondary. Accuracy at 1,024 tokens must not fall, and protected abilities (multiple-choice benchmarks, WikiText loss) must stay within bounds.

Each round you propose exactly 4 alternative edits of the CURRENT model. Each is applied separately and scored on half of a fresh sample of problems; the best two are then scored on the whole sample. The current model is scored on the same problems with the same random seeds. The best candidate is kept only if its accuracy at 256 tokens is strictly higher than the current model's on that sample and again on a second fresh sample, accuracy at 384 tokens does not fall by more than 0.05, accuracy at 1,024 tokens (plain prompt) holds, and protected abilities stay within bounds. Kept edits accumulate, so later edits can build on earlier ones. The final test uses program structures you never see, so do not fit particular programs, variable names or numbers.

What counts: the model must still reach the correct answer. Merely ending thinking early gives wrong answers and scores nothing. Measured simple baselines (validation accuracy at 256 / 384 tokens; the unedited model with a plain prompt scores about .05 / .37): the under-100-words sentence reaches about .40 / .80 (your starting point); pushing or forcing the end-of-thinking token fails (about .20 / .20 alone; with the sentence, a forced stop at 192 tokens gives .52 / .57, so it trades away later accuracy); eight sampled random weight edits showed no reliable improvement. Prefer edits that make the reasoning itself more economical (less restating of the problem, less re-checking, fewer redundant steps) or that let the model do more of the computation internally per written step.

## Tensor names""" + _TENSORS + """Activation REFs (current model, mean over this round's traces at tagged positions of the model's own thinking):
- act:resid:L<i>:<event>  residual stream entering layer i (dim 1024; i = 28 is the input of the final norm)
- act:attn:L<i>:<event>   input of L<i>.attn.o (dim 2048)
- act:mlp:L<i>:<event>    input of L<i>.mlp.down (neuron activations, dim 3072)
<event> is one of: start (last prompt token), mid (25/50/75% through thinking), recheck (the token just before a "Wait"), end (the token just before </think>). Differences are allowed, e.g. "act:resid:L20:end - act:resid:L20:mid".

## Inspection requests ("probes")
Each reply may include up to 6 probes; their results appear in your next round's input. Kinds:
- {"kind":"weight_stats","tensor":T}  shape, norms, top singular values
- {"kind":"weight_block","tensor":T,"row":r,"col":c,"size":n}  raw values of an n x n block (n <= 16)
- {"kind":"sv_readout","tensor":T,"k":k,"side":"left"|"right"}  tokens a singular vector writes or reads (when its dim is 1024)
- {"kind":"act_stats","site":"resid"|"attn"|"mlp","layer":i,"event":E}  norm and largest coordinates of a mean activation

## How to work
Read the traces and diagnostics, form hypotheses about where the tokens go and which components drive that, and make the 4 edits diverse (different mechanisms, layers and magnitudes; at least one conservative). Use the history: do not repeat what failed, refine what helped, and build on kept edits. Keep the analysis concise.

Return JSON only: "analysis" (<= 120 words), "mutations" (exactly 4 objects, each with a short "rationale" and "ops"), and "probes" (0 to 6 objects)."""

SCHEMA = {
    "type": "object",
    "properties": {
        "analysis": {"type": "string"},
        "mutations": editor.SCHEMA["properties"]["mutations"],
        "probes": {"type": "array", "items": {
            "type": "object",
            "properties": {"kind": {"type": "string", "enum": ["weight_stats", "weight_block", "sv_readout", "act_stats"]},
                           "tensor": {"type": "string"}, "site": {"type": "string"}, "layer": {"type": "integer"}, "event": {"type": "string"},
                           "row": {"type": "integer"}, "col": {"type": "integer"}, "size": {"type": "integer"}, "k": {"type": "integer"},
                           "side": {"type": "string"}},
            "required": ["kind"], "additionalProperties": False}},
    },
    "required": ["analysis", "mutations", "probes"],
    "additionalProperties": False,
}

TAGS = [("re-checking", r"\bwait\b|double-check|let me check|check again|make sure|verify|once more|\bagain\b"),
        ("state updates", r"=|\bis (now|set|assigned)|becomes|assigned|value of|\bnow\b"),
        ("final answer", r"answer|therefore|so the (output|program)|\bprints?\b|\boutput\b"),
        ("restating / setup", r"the user|the program|let me (see|look|trace|go)|let's see|okay|step by step|first,")]


def breakdown(texts):
    tot = {t: 0 for t, _ in TAGS}; tot["other"] = 0
    for text in texts:
        for s in re.split(r"(?<=[.!?])\s+|\n+", text.split("</think>")[0]):
            if not s.strip():
                continue
            tag = next((t for t, pat in TAGS if re.search(pat, s.lower())), "other")
            tot[tag] += len(s)
    z = max(1, sum(tot.values()))
    return ", ".join(f"{t} {100 * v / z:.0f}%" for t, v in tot.items())


def _stats(r, cap):
    n = len(r["correct"]); ok = lambda b: sum(c and f and k <= b for c, f, k in zip(r["correct"], r["finished"], r["tokens"])) / n
    return dict(acc256=ok(256) if cap >= 256 else sum(r["correct"]) / n, acc384=ok(384) if cap >= 384 else sum(r["correct"]) / n,
                mean_tokens=sum(r["tokens"]) / n, hit_cap=sum(not f for f in r["finished"]) / n)


def _ok(r, b=256):
    return [c and f and k <= b for c, f, k in zip(r["correct"], r["finished"], r["tokens"])]


def history_text(rounds):
    if not rounds:
        return "## History\nNone yet: this is the first round."
    out = ["## History (oldest first). Each candidate was scored on that round's fresh sample."]
    for g in rounds:
        out.append(f"Round {g['round']}: current model acc@256 {g['parent']['acc256']:.2f}, acc@384 {g['parent']['acc384']:.2f}, mean tokens {g['parent']['mean_tokens']:.0f}; "
                   + (f"KEPT candidate {g['accepted']}" if g["accepted"] is not None else f"nothing kept ({g.get('reject_reason') or 'no candidate beat the current model'})"))
        for c in g["candidates"]:
            res = "INVALID: " + c["error"][:120] if c.get("error") else f"acc@256 {c['acc256']:.2f}, acc@384 {c['acc384']:.2f}, mean tokens {c['mean_tokens']:.0f} (n={c.get('n', '?')})"
            out.append(f"  [{c['k']}] {c.get('desc') or c.get('raw_ops', '')[:200]} -> {res}")
            if c.get("rationale") and g["round"] >= rounds[-1]["round"] - 2:
                out.append(f"      rationale: {c['rationale'][:160]}")
    return "\n".join(out)


# ------------------------------------------------------------------ main loop

def attempt_spent():
    led = json.loads(Path("results/evolution/ledger.json").read_text())
    return sum(e["charged_or_reserved_usd"] for e in led["entries"] if str(e.get("label", "")).startswith(LABEL))


def run(rounds, dry=False, smoke=False):
    run_dir = R_DIR / ("edit_smoke" if smoke else "edit_dry" if dry else "edit_run1"); run_dir.mkdir(parents=True, exist_ok=True)
    R = Runner(); params = dict(R.model.named_parameters())
    names = [editlang.param_name(a) for a in editlang.all_aliases(R.model.config.num_hidden_layers)]
    original = {n: params[n].detach().clone() for n in names}
    vp = R_DIR / "val_v1.json"
    if vp.exists() and not smoke:
        V = json.loads(vp.read_text(encoding="utf-8")); dev_fams, val_items = V["dev_families"], V["items"]
    else:
        dev_fams, val_items = tasks.DEV, tasks.sample(tasks.DEV, 1, "smoke")
    n_each = (10 if smoke else 40) // len(dev_fams) or 1; cap = 48 if smoke else 384; vcap = 48 if smoke else 512
    S = json.loads(Path("results/evolution/splits.json").read_text())
    dev_p = S["dev_protected"]["items"][:10 if smoke else None]; enc_p = backend.Encoded(R.tok, dev_p); lids = backend.letter_ids(R.tok); wiki = S["wiki_dev"]["items"]

    def protected():
        prot = backend.summarize(dev_p, backend.score(R.model, enc_p, lids))
        return dict(protected=prot["fitness"], nll=backend.perplexity_nll(R.model, wiki))

    def val_score(suffix=SUFFIX, direct=True):
        v = R.score(val_items, think=True, max_new=vcap, seed="val", suffix=suffix); fam = [it["family"] for it in val_items]
        byf = {f: sum(c and fi and k <= 256 for c, fi, k, g in zip(v["correct"], v["finished"], v["tokens"], fam) if g == f) / fam.count(f) for f in sorted(set(fam))}
        d = R.score(val_items, think=False, max_new=64, seed="val-direct")["acc"] if direct else None      # non-thinking mode, zero-shot
        return dict(acc_at=v["acc_at"], median_tokens=v["median_tokens"], mean_tokens=v["mean_tokens"], hit_cap=v["hit_cap"], acc256_by_family=byf, direct=d,
                    correct=v["correct"], tokens=v["tokens"], finished=v["finished"])

    ret_items = [it for j, it in enumerate(val_items) if j % 3 != 2] if not smoke else val_items     # stratified: two thirds of every family

    def retention():
        """Screening guardrail: finished-and-correct accuracy at 1,024 tokens, plain prompt (full-thinking ability must survive)."""
        r = R.score(ret_items, think=True, max_new=48 if smoke else 1024, seed="val-ret")
        return r["acc"] if smoke else r["acc_at"][1024]

    log_path, genome_path = run_dir / "log.json", run_dir / "genome.pt"
    intkeys = lambda d: {int(k) if k.isdigit() else k: v for k, v in d.items()}        # JSON turns budget keys into strings
    log = json.loads(log_path.read_text(encoding="utf-8"), object_hook=intkeys) if log_path.exists() else dict(rounds=[], accepted=[], stop=None)
    genome = torch.load(genome_path, weights_only=False, map_location=next(iter(params.values())).device) if genome_path.exists() else []

    def save():
        log_path.write_text(json.dumps(log, indent=1), encoding="utf-8"); torch.save(genome, genome_path)

    def rebuild(k):
        del genome[k:]
        with torch.no_grad():
            for n in names:
                params[n].copy_(original[n])
        editlang.replay(params, genome, original)

    if "base" not in log:
        t0 = time.time()
        log["base"] = dict(protected=protected(), val=val_score(), val_plain=val_score(suffix="", direct=False), retention=retention(),
                           seconds=round(time.time() - t0))
        log["config"] = dict(suffix=SUFFIX, batch=os.environ.get("AMORT_BATCH", "10"), n_cand=N_CAND, n_final=N_FINAL, cap=cap, tol384=TOL384)
        save()
        print(f"[edit] base: val(prompted) {log['base']['val']['acc_at']} val(plain) {log['base']['val_plain']['acc_at']} median tok {log['base']['val']['median_tokens']} retention {log['base']['retention']:.2f} "
              f"direct {log['base']['val']['direct']:.2f} ({log['base']['seconds']}s)", flush=True)
    editlang.replay(params, genome, original)
    base = log["base"]; rnd_prop = RandomProposer(Path("results/evolution/random_calibration.json"), seed=77) if (dry or smoke) else None
    probe_text = log.get("probe_text", "None requested.")

    for rnd in range(len(log["rounds"]) + 1, rounds + 1):
        R_ = log["rounds"]
        if len(R_) >= 4 and all(g["accepted"] is None for g in R_[-4:]):
            log["stop"] = "four consecutive rounds without an accepted edit"; break
        if len(R_) >= 3 and all(g["n_valid"] < N_CAND / 2 for g in R_[-3:]):
            log["stop"] = "fewer than half of proposals valid for three rounds (interface problem)"; break
        if len(R_) >= 3 and all(g.get("reject_reason") == "interference" for g in R_[-3:]):
            log["stop"] = "three consecutive best candidates rejected for interference"; break
        acc = log["accepted"]
        if len(acc) >= 2 and all(a["dev_gain"] >= 0.15 and a["val_gain"] < a["dev_gain"] / 3 for a in acc[-2:]):
            log["stop"] = "overfitting: development gain not reflected on validation"; break
        t0 = time.time()
        dev = tasks.sample(dev_fams, n_each, f"dev-r{rnd}"); conf = tasks.sample(dev_fams, n_each, f"conf-r{rnd}")
        half = [i for i in range(len(dev)) if (i % n_each) < (n_each + 1) // 2]          # stratified by family
        dev_a = [dev[i] for i in half]; dev_b = [dev[i] for i in range(len(dev)) if i not in set(half)]
        dev = dev_a + dev_b                         # one fixed order; parent and candidates share halves, batch layout and seeds
        pa = R.score(dev_a, think=True, max_new=cap, seed=f"dev-r{rnd}-a", suffix=SUFFIX, keep_outs=True)
        pb = R.score(dev_b, think=True, max_new=cap, seed=f"dev-r{rnd}-b", suffix=SUFFIX, keep_outs=True)
        parent = {k: pa[k] + pb[k] for k in ("correct", "tokens", "finished", "ids")}
        traces = Traces(R, parent["ids"])
        pp = _stats(parent, cap); pp["ok256"] = [int(x) for x in _ok(parent)]; pp["tokens"] = parent["tokens"]
        pp["families"] = [it["family"] for it in dev]        # per-item record: F3-F5 (reasoning-needed) are reported separately from F1-F2
        if "dev_base256" not in log:
            log["dev_base256"] = pp["acc256"]
        # ---- proposals
        meta = dict(cost_usd=0.0, error=None)
        vb = log["accepted"][-1]["val"] if log["accepted"] else base["val"]
        order = sorted(range(len(dev)), key=lambda i: parent["tokens"][i]); ok256 = _ok(parent)
        right = [i for i in order if ok256[i]]; late = [i for i in order if parent["correct"][i] and not ok256[i]]
        bad = [i for i in order if not parent["correct"][i]]
        show = right[:2] + late[-2:] + bad[:1] + bad[-1:]      # short successes, correct-but-too-long, wrong or unfinished
        show = list(dict.fromkeys(show + order[:3] + order[-3:]))[:6]
        ex = []
        for i in show:
            txt = R.tok.decode(parent["ids"][i][1], skip_special_tokens=False)
            txt = txt if len(txt) <= 1300 else txt[:700] + "\n[...]\n" + txt[-500:]
            ex.append(f"### Example (family {dev[i]['family']}, {parent['tokens'][i]} tokens, "
                      f"{'correct within 256' if ok256[i] else 'correct but over 256 tokens' if parent['correct'][i] else 'wrong or unfinished at 384'}; target {dev[i]['a']})\n"
                      f"```python\n{dev[i]['code']}\n```\nModel output:\n{txt}")
        user = "\n".join([
            "## Current model", f"Accepted edits so far: {len(genome)}.",
            f"This round's sample ({len(dev)} problems): acc@256 {pp['acc256']:.2f}, acc@384 {pp['acc384']:.2f}, mean generated tokens {pp['mean_tokens']:.0f}, "
            f"unfinished at 384 tokens {pp['hit_cap']:.2f}.",
            f"Validation (held-out problems, aggregate only): acc@256 {vb['acc_at'].get(256, 0):.2f} (original {base['val']['acc_at'].get(256, 0):.2f}), "
            f"acc@512 {vb['acc_at'].get(512, 0):.2f} (original {base['val']['acc_at'].get(512, 0):.2f}), median tokens {vb['median_tokens']} (original {base['val']['median_tokens']}).",
            f"Protected bounds: multiple-choice fitness must stay >= {base['protected']['protected'] - DELTA:.4f}; WikiText loss must stay <= {base['protected']['nll'] + DELTA:.4f}.",
            f"Approximate share of thinking text by type (rule-based): {breakdown([R.tok.decode(g, skip_special_tokens=False) for _, g in parent['ids']])}.",
            "", "## Example traces (short successes, correct but over budget, wrong or unfinished)", "\n\n".join(ex),
            "", "## Diagnostics", traces.diagnostics(params),
            "", "## Results of your probes from the previous round", probe_text,
            "", history_text(log["rounds"]),
            "", "Propose exactly 4 diverse mutations of the current model, and up to 6 probes."])
        (run_dir / f"dossier_r{rnd:02d}.txt").write_text(user, encoding="utf-8")
        if rnd_prop is None:
            if attempt_spent() + editor.worst_case((len(SYSTEM) + len(user)) // 3) > ATTEMPT_CAP:
                log["stop"] = f"API cap for this attempt (${ATTEMPT_CAP:.2f}) would be exceeded"; break
            parsed, rec = editor.call(user, f"{LABEL}r{rnd:02d}", run_dir / "calls", system=SYSTEM, schema=SCHEMA)
            props = [(m.get("ops"), m.get("rationale", "")) for m in (parsed or {}).get("mutations", [])[:N_CAND]]
            meta = dict(cost_usd=rec["cost_usd"], error=rec["error"], stop=rec["stop_reason"], analysis=(parsed or {}).get("analysis"))
            probe_text = run_probes(R, params, traces, (parsed or {}).get("probes"))
            log["probe_text"] = probe_text
        else:
            props = [(rnd_prop.sample(), "random") for _ in range(N_CAND * 6)]
            probe_text = run_probes(R, params, traces, [dict(kind="weight_stats", tensor="L20.attn.o"), dict(kind="weight_block", tensor="L27.ln2", row=0, size=6),
                                                        dict(kind="sv_readout", tensor="L24.mlp.down", k=0, side="left"),
                                                        dict(kind="act_stats", site="resid", layer=20, event="mid")])
            (run_dir / f"probes_r{rnd:02d}.txt").write_text(probe_text, encoding="utf-8")
        bank = Bank(R, params, traces)
        cands = []
        for ops, why in props:
            if len(cands) >= N_CAND and (rnd_prop is None or sum("_mut" in c for c in cands) >= N_CAND):
                break
            c = dict(k=len(cands), rationale=why, raw_ops=json.dumps(ops)[:1500])
            try:
                mut = editlang.resolve(ops, params, bank); c["desc"] = editlang.describe(mut)
                with editlang.Applied(params, mut, original):
                    r = R.score(dev_a, think=True, max_new=cap, seed=f"dev-r{rnd}-a", suffix=SUFFIX)
                c["_mut"] = mut; c["_ra"] = r; c["n"] = len(dev_a); c.update(_stats(r, cap)); c["_fit"] = (c["acc256"], c["acc384"], -c["mean_tokens"])
            except (editlang.InvalidEdit, KeyError, TypeError, ValueError, IndexError, RuntimeError, AttributeError) as e:
                if rnd_prop is not None:
                    continue                      # random stand-in: resample until valid
                c["error"] = f"{type(e).__name__}: {e}"
            cands.append(c)
        while len(cands) < N_CAND:
            cands.append(dict(k=len(cands), error="missing proposal"))
        # successive halving: every candidate is scored on half the sample; only the best N_FINAL get the other half
        for c in sorted([c for c in cands if "_mut" in c], key=lambda c: c["_fit"], reverse=True)[:N_FINAL]:
            with editlang.Applied(params, c["_mut"], original):
                rb = R.score(dev_b, think=True, max_new=cap, seed=f"dev-r{rnd}-b", suffix=SUFFIX)
            full = {k: c["_ra"][k] + rb[k] for k in ("correct", "tokens", "finished")}
            c["n"] = len(dev); c.update(_stats(full, cap)); c["_fit"] = (c["acc256"], c["acc384"], -c["mean_tokens"]); c["_full"] = True
            c["ok256"] = [int(x) for x in _ok(full)]; c["tokens"] = full["tokens"]
            c["wins"] = sum(x and not y for x, y in zip(_ok(full), _ok(parent))); c["losses"] = sum(y and not x for x, y in zip(_ok(full), _ok(parent)))
        good = [c for c in cands if c.get("_full") and c["acc256"] > pp["acc256"] + 1e-9 and c["acc384"] >= pp["acc384"] - TOL384 - 1e-9]
        if smoke:                                 # exercise the acceptance path regardless of score
            good = [c for c in cands if "_mut" in c]
        best = max(good, key=lambda c: c["_fit"]) if good else None
        g = dict(round=rnd, parent=pp, meta=meta, accepted=None, n_valid=sum("_mut" in c for c in cands), reject_reason=None)
        if best is not None:
            cpar = R.score(conf, think=True, max_new=cap, seed=f"conf-r{rnd}", suffix=SUFFIX)
            with editlang.Applied(params, best["_mut"], original) as ap:
                cch = R.score(conf, think=True, max_new=cap, seed=f"conf-r{rnd}", suffix=SUFFIX); pr = protected()
                sp, sc = _stats(cpar, cap), _stats(cch, cap)
                ok_conf = (sc["acc256"] > sp["acc256"] + 1e-9 and sc["acc384"] >= sp["acc384"] - TOL384 - 1e-9) or smoke
                ok_prot = pr["protected"] >= base["protected"]["protected"] - DELTA and pr["nll"] <= base["protected"]["nll"] + DELTA
                g["confirm"] = dict(parent=sp, child=sc, protected=pr, wins=sum(x and not y for x, y in zip(_ok(cch), _ok(cpar))),
                                    losses=sum(y and not x for x, y in zip(_ok(cch), _ok(cpar))))
                ok_ret = True
                if ok_conf and ok_prot:               # retention guardrail on every edit before it is kept (no provisional checkpoints)
                    g["confirm"]["retention"] = retention(); ok_ret = g["confirm"]["retention"] >= base["retention"] - 0.05
                if ok_conf and ok_prot and ok_ret:
                    ap.keep = True
            if ok_conf and ok_prot and ok_ret:
                genome.append(best["_mut"]); g["accepted"] = best["k"]
                v = val_score()
                a = dict(round=rnd, genome_len=len(genome), val=v, protected=pr, dev_gain=best["acc256"] - log["dev_base256"],
                         val_gain=v["acc_at"].get(256, 0) - base["val"]["acc_at"].get(256, 0))
                a["retention"] = g["confirm"]["retention"]
                log["accepted"].append(a)
                ckpts = [(base["val"]["acc_at"].get(256, 0), 0)] + [(x["val"]["acc_at"].get(256, 0), x["genome_len"]) for x in log["accepted"]]
                best_val, k = max(ckpts, key=lambda t: (t[0], -t[1]))          # the original model is a checkpoint too
                if v["acc_at"].get(256, 0) < best_val - 0.08:
                    rebuild(k); log["active_len"] = k
                    log["stop"] = f"validation fell more than 8 points below its best; rolled back to the checkpoint with {k} edits"
            else:
                g["reject_reason"] = "interference" if not (ok_prot and ok_ret) else "not confirmed on the second sample"
        g["candidates"] = [{k: v for k, v in c.items() if not k.startswith("_")} for c in cands]
        g["seconds"] = round(time.time() - t0); g["genome_len"] = len(genome)
        log["rounds"].append(g); save()
        va = log["accepted"][-1]["val"]["acc_at"] if log["accepted"] else base["val"]["acc_at"]
        print(f"[edit] round {rnd}: sample acc@256 {pp['acc256']:.2f} mean tok {pp['mean_tokens']:.0f} | valid {g['n_valid']}/{N_CAND} | "
              f"best {best['_fit'][0] if best else None} | kept {g['accepted']} ({g['reject_reason'] or '-'}) | edits {len(genome)} | val {va} | "
              f"${meta.get('cost_usd') or 0:.3f} (attempt ${attempt_spent():.2f}) | {g['seconds']}s", flush=True)
        if log["stop"]:
            break
    save()
    print(f"[edit] STOPPED: {log['stop'] or 'round limit reached'}; accepted edits {len(genome)}", flush=True)
    print("[edit] DONE", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=15); p.add_argument("--dry", action="store_true"); p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    run(1 if a.smoke else a.rounds, dry=a.dry, smoke=a.smoke)
