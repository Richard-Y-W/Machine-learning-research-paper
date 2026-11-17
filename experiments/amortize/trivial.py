"""Stage 0b: trivial baselines on the validation set (no LLM edits, no API).

Conditions: base; "think briefly" prompts; bias on the end-of-thinking token; a steering vector;
random weight edits of LLM-calibrated size. All scored in thinking mode at a 512-token cap, which
yields accuracy at 256/384/512. These set the bar a weight edit must beat.
"""
import json
import random
import time
from pathlib import Path

import torch

from ..evolution import editlang
from ..evolution.baselines import RandomProposer
from . import tasks
from .core import Runner

R_DIR = Path("results/amortize")
SUFFIXES = {"brief": " Think briefly.", "under100": " Keep your thinking under 100 words.",
            "trace_once": " Be concise: trace the variables once, then answer."}


def validation_items():
    scr = json.loads((R_DIR / "screen.json").read_text(encoding="utf-8"))
    dev = [f for f in tasks.DEV if scr[f]["keep"]]; held = [f for f in tasks.VAL_HELD if scr[f]["keep"]]
    items = tasks.sample(dev, 30 // max(1, len(dev)), "val-v1") + (tasks.sample(held, 30 // len(held), "val-v1") if held else [])
    (R_DIR / "val_v1.json").write_text(json.dumps(dict(dev_families=dev, held_families=held, items=items), indent=1), encoding="utf-8")
    return items, dev, held


def add_at_layer(model, layer, vec):
    def hook(mod, args, kwargs):
        if args:
            return (args[0] + vec,) + tuple(args[1:]), kwargs
        kwargs = dict(kwargs); kwargs["hidden_states"] = kwargs["hidden_states"] + vec
        return args, kwargs
    return model.model.layers[layer].register_forward_pre_hook(hook, with_kwargs=True)


@torch.inference_mode()
def steering_vector(R, items, layer):
    def mean_last(suffix):
        acc = 0
        for i in range(0, len(items), 10):
            enc = R.tok([R.prompt(it, True, suffix=suffix) for it in items[i:i + 10]], return_tensors="pt", padding=True).to(R.model.device)
            hs = R.model(**enc, output_hidden_states=True, use_cache=False).hidden_states[layer]
            acc = acc + hs[:, -1].float().sum(0)
        return acc / len(items)
    return mean_last(SUFFIXES["brief"]) - mean_last("")


def main():
    R = Runner(); items, dev, held = validation_items()
    fam = [it["family"] for it in items]
    res = dict(dev_families=dev, held_families=held, n=len(items), rows={})
    out_path = R_DIR / "trivial.json"

    def record(name, **kw):
        t0 = time.time(); r = R.score(items, think=True, max_new=512, seed="val", **kw)
        byf = {f: sum(c and fi and k <= 256 for c, fi, k, g in zip(r["correct"], r["finished"], r["tokens"], fam) if g == f) / fam.count(f) for f in sorted(set(fam))}
        res["rows"][name] = dict(acc_at=r["acc_at"], median_tokens=r["median_tokens"], hit_cap=r["hit_cap"], acc256_by_family=byf,
                                 correct=r["correct"], tokens=r["tokens"], seconds=round(time.time() - t0))
        print(f"[trivial] {name:22s} @256 {r['acc_at'][256]:.2f} @384 {r['acc_at'][384]:.2f} @512 {r['acc_at'][512]:.2f} "
              f"median tok {r['median_tokens']:4d} ({res['rows'][name]['seconds']}s)", flush=True)
        out_path.write_text(json.dumps(res, indent=1), encoding="utf-8")

    record("base")
    # non-thinking mode, zero-shot: the cheapest point on the accuracy-compute frontier
    d = R.score(items, think=False, max_new=64, seed="val-direct")
    res["nonthinking"] = dict(acc=d["acc"], mean_tokens=d["mean_tokens"],
                              acc_by_family={f: sum(c for c, g in zip(d["correct"], fam) if g == f) / fam.count(f) for f in sorted(set(fam))})
    print(f"[trivial] non-thinking zero-shot  acc {d['acc']:.2f} mean tok {d['mean_tokens']:.0f} by family {res['nonthinking']['acc_by_family']}", flush=True)
    for k, s in SUFFIXES.items():
        record(f"prompt:{k}", suffix=s)
    for b in (1, 2, 3, 4, 6):
        record(f"endthink_bias:{b}", bias=float(b))
    for layer in (8, 14):
        v = steering_vector(R, items, layer)
        for c in (2.0, 4.0):
            h = add_at_layer(R.model, layer, c * v)
            try:
                record(f"steer:L{layer}:c{c:g}")
            finally:
                h.remove()
    # random weight edits, sizes and op types calibrated on the earlier LLM-authored edits
    params = dict(R.model.named_parameters())
    names = [editlang.param_name(a) for a in editlang.all_aliases(R.model.config.num_hidden_layers)]
    original = {n: params[n].detach().clone() for n in names}
    bank = editlang.RefBank(params, R.tok)
    prop = RandomProposer(Path("results/evolution/random_calibration.json"), seed=4242)
    done = 0; tries = 0
    while done < 8 and tries < 400:
        tries += 1
        try:
            mut = editlang.resolve(prop.sample(), params, bank)
            with editlang.Applied(params, mut, original):
                record(f"random_edit:{done + 1}")
            done += 1
        except (editlang.InvalidEdit, KeyError, ValueError, IndexError, RuntimeError):
            continue
    # reference: base model at the full 1,024-token budget on the same items
    t0 = time.time(); r = R.score(items, think=True, max_new=1024, seed="val-1024")
    res["base_1024"] = dict(acc_at=r["acc_at"], median_tokens=r["median_tokens"], hit_cap=r["hit_cap"], seconds=round(time.time() - t0))
    out_path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[trivial] base@1024 {r['acc_at']}", flush=True)
    print("[trivial] DONE", flush=True)


if __name__ == "__main__":
    main()
