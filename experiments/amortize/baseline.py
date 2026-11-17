"""Phase 1 baseline: is there a reasoning-vs-direct gap on Qwen3-0.6B (chat)?

Tasks: random-symbol link following (2 and 4 hops); Python variable tracing (no arithmetic / one addition).
Configs: think_long, think_short, direct, direct_fewshot. No weight edits, no API.

    python -m experiments.amortize.baseline --n 10 --tag timing
"""
import argparse, json, random, re, time
from pathlib import Path

import torch

from ..evolution import backend

MODEL = "Qwen/Qwen3-0.6B"
OUT = Path("results/amortize")
CONFIGS = {"think": dict(think=True, max_new=1024), "direct": dict(think=False, max_new=64),
           "direct_fewshot": dict(think=False, max_new=64, shots=3)}
BUDGETS = (256, 512, 1024)      # thinking budgets, read off the single 1024-token run (same sampled tokens up to the cap)
N_PER_TASK = {"lookup_2hop": 30, "lookup_4hop": 30, "code_noarith": 100, "code_arith": 100}


def node_names(rng, k):
    names = set()
    while len(names) < k:
        names.add(rng.choice("ABCDEFGHJKLMNPQRSTVWXZ") + str(rng.randrange(1, 10)))
    return list(names)


def lookup_item(rng, hops, n_nodes=10):
    names = node_names(rng, n_nodes); rng.shuffle(names)
    chain = names[:hops + 2]                                  # one extra link past the answer
    rest = names[hops + 2:]
    links = [(chain[i], chain[i + 1]) for i in range(len(chain) - 1)] + [(rest[i], rest[i + 1]) for i in range(len(rest) - 1)]
    rng.shuffle(links)
    q = ("Here is a list of links:\n" + "\n".join(f"{a} -> {b}" for a, b in links) +
         f"\nStart at {chain[0]} and follow the links exactly {hops} time{'s' if hops > 1 else ''}. "
         "Which node do you reach? End your reply with `Answer: <node>`.")
    return dict(q=q, a=chain[hops], kind="node")


def code_item(rng, arith):
    v = {}; lines = []
    def setv(name, expr, val):
        v[name] = val; lines.append(f"{name} = {expr}")
    x, y = rng.sample(range(10, 100), 2)
    setv("a", x, x); setv("b", "a", v["a"])
    z = rng.choice([n for n in range(10, 100) if n not in (x, y)])
    setv("a", z, z); setv("d", y, y); setv("e", "d", v["d"]); setv("d", "b", v["b"])
    if arith:
        setv("c", "b + a", v["b"] + v["a"]); lines.append("print(c)"); ans = v["c"]
    else:
        tgt = rng.choice(["b", "d", "e"]); lines.append(f"print({tgt})"); ans = v[tgt]
    q = "What does this Python program print? Answer with only the number.\n```python\n" + "\n".join(lines) + "\n```"
    return dict(q=q, a=str(ans), kind="int")


TASKS = {"lookup_2hop": lambda r: lookup_item(r, 2), "lookup_4hop": lambda r: lookup_item(r, 4),
         "code_noarith": lambda r: code_item(r, False), "code_arith": lambda r: code_item(r, True)}


def extract(text, kind, think):
    if think:
        if "</think>" not in text:
            return None                                        # ran out of budget while thinking
        text = text.split("</think>")[-1]
    text = re.sub(r"<\|[^|]*\|>", " ", text)
    text = text.split("```")[0]                                # ignore any code the model re-prints after its answer
    pat = r"[A-Z][1-9]" if kind == "node" else r"-?\d+"
    for rule in (r"Answer:\s*[*`<]*\s*(" + pat + r")\b", r"\*\*\s*(" + pat + r")\s*\*\*", r"\b(" + pat + r")\b"):
        m = re.findall(rule, text)                             # requested format, then bolded value, then last value
        if m:
            return m[-1]
    return None


def run(n, tag, seed, only=None):
    backend.set_threads()
    model, tok = backend.load_model(MODEL); tok.padding_side = "left"
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    n_arg = n
    res = dict(model=MODEL, n=n or N_PER_TASK, seed=seed, cells={})
    for tname, make in TASKS.items():
        rng = random.Random(f"{seed}|{tname}")
        n = n_arg or N_PER_TASK[tname]
        items = [make(rng) for _ in range(n)]
        shots = [make(random.Random(f"shots|{tname}|{i}")) for i in range(3)]
        for cname, cfg in CONFIGS.items():
            if only and cname not in only:
                continue
            prompts = []
            for it in items:
                msgs = []
                for s in shots[:cfg.get("shots", 0)]:
                    msgs += [dict(role="user", content=s["q"]), dict(role="assistant", content="Answer: " + s["a"])]
                msgs.append(dict(role="user", content=it["q"]))
                prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=cfg["think"]))
            t0 = time.time(); outs = []; ntok = []
            torch.manual_seed(hash((seed, tname, cname)) % 2**31)
            for i in range(0, n, 10):
                enc = tok(prompts[i:i + 10], return_tensors="pt", padding=True)
                samp = dict(temperature=0.6, top_p=0.95) if cfg["think"] else dict(temperature=0.7, top_p=0.8)
                out = model.generate(**enc, max_new_tokens=cfg["max_new"], do_sample=True, top_k=20, pad_token_id=tok.pad_token_id, **samp)
                new = out[:, enc["input_ids"].shape[1]:]
                for row in new:
                    ids = row.tolist(); k = ids.index(eos) + 1 if eos in ids else len(ids)
                    ntok.append(k); outs.append(tok.decode(ids[:k], skip_special_tokens=False))
            preds = [extract(o, it["kind"], cfg["think"]) for o, it in zip(outs, items)]
            correct = [p == it["a"] for p, it in zip(preds, items)]
            fin = [("<|im_end|>" in o) for o in outs]
            by_budget = {b: sum(c and f and k <= b for c, f, k in zip(correct, fin, ntok)) / n for b in BUDGETS} if cfg["think"] else None
            cell = dict(n=n, acc=sum(correct) / n, acc_by_budget=by_budget, hit_cap=sum(not f for f in fin) / n, per_item_correct=correct, per_item_tokens=ntok,
                        mean_new_tokens=sum(ntok) / n, unfinished=sum(p is None for p in preds) / n,
                        seconds=round(time.time() - t0, 1), samples=[dict(q=it["q"], a=it["a"], out=o, pred=p) for it, o, p in list(zip(items, outs, preds))[:3]])
            res["cells"][f"{tname}/{cname}"] = cell
            print(f"[{tag}] {tname:13s} {cname:15s} acc {cell['acc']:.2f}  tokens {cell['mean_new_tokens']:7.1f}  no-answer {cell['unfinished']:.2f}  hit-cap {cell['hit_cap']:.2f}  budgets {cell['acc_by_budget']}  {cell['seconds']}s", flush=True)
            OUT.mkdir(parents=True, exist_ok=True); (OUT / f"baseline_{tag}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[{tag}] TOTAL seconds {sum(c['seconds'] for c in res['cells'].values()):.0f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--n", type=int, default=0); p.add_argument("--only", nargs="*"); p.add_argument("--tag", default="timing"); p.add_argument("--seed", default="timing")
    a = p.parse_args(); run(a.n, a.tag, a.seed, a.only)
