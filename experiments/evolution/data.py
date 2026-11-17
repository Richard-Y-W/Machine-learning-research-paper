"""Benchmark loading, normalization to lettered multiple choice, and frozen splits.

Every item becomes {uid, task, prompt, n_choices, label}. The model is scored on
the next-token distribution over the choice letters after "Answer:" (MMLU-style
zero-shot format), so one forward pass scores one item.
"""
import hashlib
import json
import random
from pathlib import Path

LETTERS = "ABCDE"

# name -> (hf path, config, splits pooled into the item pool)
SOURCES = {
    "mmlu": ("cais/mmlu", "all", ["test", "validation"]),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", ["test", "validation"]),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", ["test", "validation"]),
    "hellaswag": ("Rowan/hellaswag", None, ["validation"]),
    "commonsense_qa": ("tau/commonsense_qa", None, ["validation"]),
    "openbookqa": ("allenai/openbookqa", "main", ["test", "validation"]),
    "piqa": ("baber/piqa", None, ["validation"]),
    "winogrande": ("allenai/winogrande", "winogrande_xl", ["validation"]),
    "sciq": ("allenai/sciq", None, ["test", "validation"]),
}


def _fmt(header, question, choices):
    lines = [header, "", question.strip()]
    lines += [f"{LETTERS[i]}. {c.strip()}" for i, c in enumerate(choices)]
    lines.append("Answer:")
    return "\n".join(lines)


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def normalize(task, row, idx):
    """Return (prompt, n_choices, label, group_key) or None if unusable."""
    if task == "mmlu":
        subj = row["subject"].replace("_", " ")
        p = _fmt(f"The following is a multiple choice question about {subj}.", row["question"], row["choices"])
        return p, len(row["choices"]), int(row["answer"]), row["question"]
    if task in ("arc_challenge", "arc_easy", "openbookqa", "commonsense_qa"):
        q = row["question_stem"] if task == "openbookqa" else row["question"]
        labels, texts = row["choices"]["label"], row["choices"]["text"]
        key = row["answerKey"]
        if key not in labels or len(texts) > 5:
            return None
        hdr = "The following is a multiple choice question about science." if task.startswith("arc") or task == "openbookqa" \
            else "The following is a multiple choice question about commonsense."
        return _fmt(hdr, q, texts), len(texts), labels.index(key), q
    if task == "hellaswag":
        ctx = f"{row['activity_label']}: {row['ctx']}"
        p = _fmt("Choose the most plausible continuation.", ctx, row["endings"])
        return p, 4, int(row["label"]), row["ctx"]
    if task == "piqa":
        p = _fmt("Choose the better solution to achieve the goal.", "Goal: " + row["goal"], [row["sol1"], row["sol2"]])
        return p, 2, int(row["label"]), row["goal"]
    if task == "winogrande":
        p = _fmt("Choose the option that correctly fills the blank (_).", row["sentence"], [row["option1"], row["option2"]])
        return p, 2, int(row["answer"]) - 1, row["sentence"]
    if task == "sciq":
        opts = [row["correct_answer"], row["distractor1"], row["distractor2"], row["distractor3"]]
        rng = random.Random(_sha(row["question"]))
        order = list(range(4)); rng.shuffle(order)
        texts = [opts[i] for i in order]
        return _fmt("The following is a multiple choice question about science.", row["question"], texts), 4, order.index(0), row["question"]
    raise KeyError(task)


def load_pool(task, tokenizer=None, max_tokens=200):
    from datasets import load_dataset
    path, cfg, splits = SOURCES[task]
    items, seen = [], set()
    for split in splits:
        for i, row in enumerate(load_dataset(path, cfg, split=split)):
            out = normalize(task, row, i)
            if out is None:
                continue
            prompt, n, label, group = out
            gkey = _sha(" ".join(group.lower().split()))
            if gkey in seen:  # dedupe by normalized underlying question
                continue
            seen.add(gkey)
            if tokenizer is not None and len(tokenizer(prompt)["input_ids"]) > max_tokens:
                continue
            items.append(dict(uid=f"{task}:{gkey[:16]}", task=task, prompt=prompt, n_choices=n, label=label))
    return items


def split_items(items, sizes, seed="wsp-evolution-v1"):
    """Deterministic split by hashed uid. sizes: ordered dict name->count."""
    ordered = sorted(items, key=lambda it: _sha(seed + it["uid"]))
    out, start = {}, 0
    for name, n in sizes.items():
        out[name] = ordered[start:start + n]
        start += n
    if start > len(ordered):
        raise ValueError(f"{items[0]['task']}: need {start} items, have {len(ordered)}")
    return out


def load_wikitext_chunks(tokenizer, n_chunks, chunk_len, offset_chunks=0):
    from datasets import load_dataset
    text = "\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")["text"] if t.strip())
    ids = tokenizer(text[:2_000_000])["input_ids"]
    chunks = [ids[i * chunk_len:(i + 1) * chunk_len] for i in range(offset_chunks, offset_chunks + n_chunks)]
    return chunks


def write_split_manifest(path, splits):
    """splits: name -> list of items. Writes items plus a content hash per split."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: dict(sha256=_sha(json.dumps(v, sort_keys=True)), n=len(v), items=v) for name, v in splits.items()}
    path.write_text(json.dumps(payload, indent=1, allow_nan=False))
    return {k: v["sha256"] for k, v in payload.items()}
