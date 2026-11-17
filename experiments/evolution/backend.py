"""Local model execution: lettered multiple-choice scoring and perplexity on CPU.

Forward passes only. No gradients are used anywhere in the evolutionary arms.
"""
import math
import os

import torch

MODEL_NAME = os.environ.get("EVO_MODEL", "Qwen/Qwen3-0.6B-Base")
LETTERS = "ABCDE"


def set_threads(n=None):
    torch.set_num_threads(int(n or os.environ.get("EVO_THREADS", 6)))


def load_model(name=MODEL_NAME):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    if torch.cuda.is_available():            # GPU: new tensors default to the GPU so the CPU-written code runs unchanged
        torch.set_default_device('cuda'); model = model.to('cuda')
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def letter_ids(tok):
    ids = []
    for L in LETTERS:
        t = tok(" " + L, add_special_tokens=False)["input_ids"]
        assert len(t) == 1, (L, t)
        ids.append(t[0])
    return ids


class Encoded:
    """Pre-tokenized item set, sorted by length into padded batches."""

    def __init__(self, tok, items, batch_tokens=4096):
        self.items = items
        enc = [tok(it["prompt"], add_special_tokens=False)["input_ids"] for it in items]
        order = sorted(range(len(items)), key=lambda i: len(enc[i]))
        self.batches = []
        cur = []
        for i in order:
            cur.append(i)
            if len(cur) * len(enc[i]) >= batch_tokens:
                self.batches.append(cur); cur = []
        if cur:
            self.batches.append(cur)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        self.tensors = []
        for b in self.batches:
            L = max(len(enc[i]) for i in b)
            ids = torch.full((len(b), L), pad, dtype=torch.long)
            mask = torch.zeros((len(b), L), dtype=torch.long)
            last = torch.zeros(len(b), dtype=torch.long)
            for r, i in enumerate(b):
                ids[r, :len(enc[i])] = torch.tensor(enc[i]); mask[r, :len(enc[i])] = 1; last[r] = len(enc[i]) - 1
            self.tensors.append((ids, mask, last))
        self.n_tokens = sum(len(e) for e in enc)


def _letter_logits(model, hidden_last, lid):
    """Final norm is already applied; project only last positions onto letter rows."""
    return model.lm_head(hidden_last)[:, lid].float()


@torch.inference_mode()
def score(model, enc, lids, cache=None, min_layer=None):
    """Per-item logp of the correct letter (normalized over valid letters), correctness, prediction.
    With a parent `cache` (see build_cache) and `min_layer`, only layers >= min_layer are re-run;
    this is exact because layers below min_layer are unchanged."""
    n = len(enc.items)
    logp = [0.0] * n; correct = [False] * n; pred = [0] * n
    lid = torch.tensor(lids)
    for bi, (b, (ids, mask, last)) in enumerate(zip(enc.batches, enc.tensors)):
        if cache is not None and min_layer is not None and best_level(cache[bi], min_layer) is not None:
            h = run_from(model, cache[bi], min_layer)
        else:
            h = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        logits = _letter_logits(model, h[torch.arange(len(b)), last], lid)
        for r, i in enumerate(b):
            k = enc.items[i]["n_choices"]
            lp = torch.log_softmax(logits[r, :k], dim=-1)
            lab = enc.items[i]["label"]
            logp[i] = float(lp[lab]); pred[i] = int(lp.argmax()); correct[i] = pred[i] == lab
    return dict(logp=logp, correct=correct, pred=pred)


@torch.inference_mode()
def build_cache(model, enc, levels=None):
    """For each batch: inputs to the decoder layers in `levels` (default: all), the
    pre-final-norm residual (level L), and the layer kwargs."""
    layers = model.model.layers; L = len(layers)
    keep = set(range(L)) if levels is None else set(levels)
    out = []
    for ids, mask, last in enc.tensors:
        store = {"h": [None] * (L + 1), "kw": None}
        hooks = [layers[i].register_forward_pre_hook(
            lambda mod, args, kwargs, i=i: store["h"].__setitem__(i, args[0] if args else kwargs["hidden_states"]),
            with_kwargs=True) for i in range(L) if i in keep]
        hooks.append(layers[0].register_forward_pre_hook(
            lambda mod, args, kwargs: store.__setitem__("kw", {k: v for k, v in kwargs.items() if k != "hidden_states"}),
            with_kwargs=True))
        hooks.append(model.model.norm.register_forward_pre_hook(lambda mod, args: store["h"].__setitem__(L, args[0])))
        try:
            model.model(input_ids=ids, attention_mask=mask, use_cache=False)
        finally:
            for hk in hooks:
                hk.remove()
        out.append(store)
    return out


def best_level(store, k):
    """Highest cached level <= k (None if none)."""
    lv = [i for i, h in enumerate(store["h"]) if h is not None and i <= k]
    return max(lv) if lv else None


def run_from(model, store, k):
    layers = model.model.layers
    k = best_level(store, k)
    h = store["h"][k]
    for i in range(k, len(layers)):
        h = layers[i](h, **store["kw"])
        if isinstance(h, tuple):
            h = h[0]
    return model.model.norm(h)


@torch.inference_mode()
def perplexity_nll(model, chunks, cache=None, min_layer=None):
    ids = torch.tensor(chunks)
    if cache is not None and min_layer is not None and best_level(cache[0], min_layer) is not None:
        logits = model.lm_head(run_from(model, cache[0], min_layer))
    else:
        logits = model(input_ids=ids, use_cache=False).logits
    lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    nll = -lp.gather(-1, ids[:, 1:, None]).squeeze(-1)
    return float(nll.mean())


class Chunks:
    """Wraps fixed-length LM chunks so build_cache can treat them like an Encoded set."""

    def __init__(self, chunks):
        ids = torch.tensor(chunks)
        self.tensors = [(ids, torch.ones_like(ids), None)]


def summarize(items, res):
    """Per-task mean logp and accuracy; fitness = mean over tasks of mean logp."""
    by = {}
    for it, lp, c in zip(items, res["logp"], res["correct"]):
        d = by.setdefault(it["task"], [0.0, 0, 0]); d[0] += lp; d[1] += c; d[2] += 1
    tasks = {t: dict(logp=v[0] / v[2], acc=v[1] / v[2], n=v[2]) for t, v in sorted(by.items())}
    fit = sum(v["logp"] for v in tasks.values()) / max(1, len(tasks))
    acc = sum(v["acc"] for v in tasks.values()) / max(1, len(tasks))
    return dict(fitness=fit, acc=acc, tasks=tasks)
