"""Shared generation and scoring for the amortization study (Qwen3-0.6B chat, CPU, sampled decoding)."""
import os
import zlib

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from ..evolution import backend
from .baseline import extract

MODEL = "Qwen/Qwen3-0.6B"
BUDGETS = (256, 384, 512, 1024)


class EndThinkBias(LogitsProcessor):
    """Adds `bias` to the end-of-thinking token's logit until that token has been generated."""

    def __init__(self, tid, bias, prompt_len):
        self.tid, self.bias, self.prompt_len = tid, bias, prompt_len

    def __call__(self, input_ids, scores):
        done = (input_ids[:, self.prompt_len:] == self.tid).any(-1)
        scores[~done, self.tid] += self.bias
        return scores


class ForceEndThink(LogitsProcessor):
    """Forces the end-of-thinking token once `at` tokens have been generated without it."""

    def __init__(self, tid, at, prompt_len):
        self.tid, self.at, self.prompt_len = tid, at, prompt_len

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] - self.prompt_len >= self.at:
            todo = ~(input_ids[:, self.prompt_len:] == self.tid).any(-1)
            if todo.any():
                keep = scores[todo, self.tid].clone(); scores[todo] = float("-inf"); scores[todo, self.tid] = keep
        return scores


class Runner:
    def __init__(self):
        backend.set_threads()
        self.model, self.tok = backend.load_model(MODEL); self.tok.padding_side = "left"
        self.eos = self.tok.convert_tokens_to_ids("<|im_end|>"); self.think_end = self.tok.convert_tokens_to_ids("</think>")

    def prompt(self, item, think, shots=(), suffix=""):
        msgs = []
        for s in shots:
            msgs += [dict(role="user", content=s["q"]), dict(role="assistant", content="Answer: " + s["a"])]
        msgs.append(dict(role="user", content=item["q"] + suffix))
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=think)

    def generate(self, prompts, think, max_new, seed, bias=0.0, batch=None, force_at=0):
        batch = batch or int(os.environ.get("AMORT_BATCH", "10"))
        outs, ntok, fin = [], [], []
        self.last_ids = []          # (prompt ids, generated ids) per row, for teacher-forced analysis
        s0 = zlib.crc32(str(seed).encode())
        for i in range(0, len(prompts), batch):
            torch.manual_seed((s0 + i) % 2**31)
            enc = self.tok(prompts[i:i + batch], return_tensors="pt", padding=True).to(self.model.device)
            samp = dict(temperature=0.6, top_p=0.95) if think else dict(temperature=0.7, top_p=0.8)
            pl = enc["input_ids"].shape[1]
            lp = LogitsProcessorList(([EndThinkBias(self.think_end, bias, pl)] if bias else []) + ([ForceEndThink(self.think_end, force_at, pl)] if force_at else [])) or None
            out = self.model.generate(**enc, max_new_tokens=max_new, do_sample=True, top_k=20, pad_token_id=self.tok.pad_token_id,
                                      logits_processor=lp, **samp)
            for r, row in enumerate(out[:, enc["input_ids"].shape[1]:]):
                ids = row.tolist(); done = self.eos in ids; k = ids.index(self.eos) + 1 if done else len(ids)
                ntok.append(k); fin.append(done); outs.append(self.tok.decode(ids[:k], skip_special_tokens=False))
                self.last_ids.append((enc["input_ids"][r][enc["attention_mask"][r].bool()].tolist(), ids[:k]))
        return outs, ntok, fin

    def score(self, items, think=True, max_new=1024, seed=0, shots=(), suffix="", bias=0.0, keep_outs=False, force_at=0):
        prompts = [self.prompt(it, think, shots, suffix) for it in items]
        outs, ntok, fin = self.generate(prompts, think, max_new, seed, bias, force_at=force_at)
        correct = [extract(o, it["kind"], think) == it["a"] for o, it in zip(outs, items)]
        n = len(items)
        r = dict(n=n, correct=correct, tokens=ntok, finished=fin, acc=sum(correct) / n, mean_tokens=sum(ntok) / n,
                 median_tokens=sorted(ntok)[n // 2], hit_cap=sum(not f for f in fin) / n)
        if think:
            r["acc_at"] = {b: sum(c and f and k <= b for c, f, k in zip(correct, fin, ntok)) / n for b in BUDGETS if b <= max_new}
        if keep_outs:
            r["outs"] = outs; r["ids"] = list(self.last_ids)
        return r
