"""Prompting baselines for lookup -> successor (reviewer item 4a). Forward passes only, no fitting.

Question: is the edit installing a new default behaviour, or overcoming an inability? If the unedited model already
returns the successor when asked (instruction) or shown (few-shot composed demonstrations), the edit's contribution
is a default change, not a new capability.

Test prompts: TEST_KEYS (never used in fitting or development), arrow format, five value types, N_P per type.
Prompt variants (all end with the same test table and query):
  plain          the table only (as in the main evaluation)
  instruct       a one-line instruction to answer with the item that follows the looked-up value
  fewshot_other  3 composed demonstrations (table + 'key -> successor') from OTHER value types
  fewshot_same   3 composed demonstrations from the SAME value type (queried values differ from the test query's)
  instruct_same  instruction + fewshot_same
Each variant is run on the unedited model and on the reported direct edit (archived fit, gain 3).

    EVO_MODEL=... WSP_OUT=results/compose RB_OUT=results/robustness_0p6b python -m experiments.compose.prompting
"""
import json
import os
import random
import time

import torch

from .bridge import OUT
from . import lookup2
from .lookup import ORD
from .robustness import RB, TEST_KEYS, Rig

N_P = int(os.environ.get("RB_NPROMPT", "120"))
INSTR = ("Each line pairs a word with an item. For the last word, do not repeat its item: answer with the item that comes "
         "right after it in its usual order (after c comes d, after Monday comes Tuesday, after three comes four).\n\n")


def main():
    RB.mkdir(parents=True, exist_ok=True); rig = Rig(); tok = rig.tok; t0 = time.time()
    one = lambda s: len(tok(" " + s, add_special_tokens=False)["input_ids"]) == 1
    ordv = {dm: [x for x in ORD[dm] if one(x)] for dm in ORD}; ordv["letters"] = list("bcdefghijklmnopqrstuvwx")
    old = lookup2.KEYS; lookup2.KEYS = TEST_KEYS
    test = {dm: [lookup2.make(random.Random(f"prompting-v1|{dm}|{i}"), ordv[dm], dm, "arrow") for i in range(N_P)] for dm in ORD}

    def demos(rr, dm, k, exclude_v):
        out = []
        while len(out) < k:
            c = lookup2.make(rr, ordv[dm], dm, "arrow")
            if c["v"] != exclude_v:
                out.append(c["text"] + " " + c["s"] + "\n\n")
        return "".join(out)

    variants = {}
    for dm in ORD:
        rr = random.Random(f"prompting-demos-v1|{dm}"); others = [x for x in ORD if x != dm]
        for c in test[dm]:
            fo = "".join(demos(rr, rr.choice(others), 1, None) for _ in range(3)); fs = demos(rr, dm, 3, c["v"])
            for name, pre in (("plain", ""), ("instruct", INSTR), ("fewshot_other", fo), ("fewshot_same", fs), ("instruct_same", INSTR + fs)):
                variants.setdefault(name, {}).setdefault(dm, []).append(dict(text=pre + c["text"], s=c["s"], v=c["v"]))
    lookup2.KEYS = old
    prog = torch.load(OUT / "lookup3_attention_programme.pt"); res = dict(model=os.environ.get("EVO_MODEL"), n_per_type=N_P, instruction=INSTR,
                                                                         example=variants["instruct_same"]["months"][0]["text"], results={})
    for model_name, apply in (("unedited", lambda: None), ("direct_edit_g3", lambda: rig.direct(prog["WQ"], prog["WK"], 3.0))):
        try:
            apply()
            for vn, by in variants.items():
                row = {}
                for dm, cs in by.items():
                    top = rig.last_logits([c["text"] for c in cs], bs=16).argmax(-1).tolist()
                    row[dm] = dict(successor=sum(t == rig.lk.first(c["s"]) for t, c in zip(top, cs)) / len(cs), lookup=sum(t == rig.lk.first(c["v"]) for t, c in zip(top, cs)) / len(cs))
                res["results"][f"{model_name}|{vn}"] = row
                print(f"[prompt] {model_name:15s} {vn:14s}: " + " ".join(f"{dm} {r['successor']:.2f}/{r['lookup']:.2f}" for dm, r in row.items()) + "  (successor/lookup)", flush=True)
        finally:
            rig.restore()
        (RB / "prompting.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[prompt] DONE ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
