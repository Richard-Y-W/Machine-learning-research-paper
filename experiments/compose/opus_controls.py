"""Controls for the programme designed by Opus (opus_program.py). No API calls.

Evaluated on the confirmation prompts (fitted format, unseen key words, 400 per value type):
  the frozen programme; its components removed one at a time; the spare-head design alone; the spare-head design
  with a non-successor head's OV; the spare-head design aimed at a wrong position; the side effect alone.
"""
import json
import os
import random

from .bridge import A, OUT
from . import lookup2
from .lookup import KEYS, ORD
from .lookup3_confirm import NEW_KEYS
from .opus_program import Bench


def main():
    b = Bench(); log = json.loads((OUT / "opus_program" / "log.json").read_text()); frozen = log["frozen"]["ops"]
    lookup2.KEYS = NEW_KEYS; r2 = random.Random("confirm-v1|arrow_new_keys")
    ev = {dm: [lookup2.make(r2, b.ordv[dm], dm, "arrow") for _ in range(400)] for dm in ORD}; lookup2.KEYS = KEYS
    hosts = [o["head"] for o in frozen if o["op"] == "attention_program"]; spare = next((h for h in hosts if h != A[1]), None); L = A[0]
    variants = {"unedited": None, "frozen programme": frozen}
    for h in hosts:
        variants[f"frozen, head {h} removed"] = frozen + [dict(op="scale_output", layer=L, head=h, gain=0.0)]
    if len(hosts) > 1:
        variants["frozen, all programmed heads removed"] = frozen + [dict(op="scale_output", layer=L, head=h, gain=0.0) for h in hosts]
    if spare is not None:
        mate = spare ^ 1; other = next(h for h in (8, 10, 12, 2, 4) if h // 2 not in (A[1] // 2, spare // 2))
        base = lambda src, role: [dict(op="attention_program", layer=L, head=spare, query_role="final", key_role=role, contrast="other_positions", ridge=1.0, rank=32, rope_band="low"),
                                  dict(op="copy_ov", layer=L, from_head=src, to_head=spare), dict(op="scale_output", layer=L, head=mate, gain=0.0), dict(op="scale_output", layer=L, head=spare, gain=1.5)]
        variants["spare head only: successor head's OV, aimed at the value"] = base(A[1], "queried_value")
        variants[f"spare head only: head {other}'s OV, aimed at the value"] = base(other, "queried_value")
        variants["spare head only: successor head's OV, aimed at the arrow"] = base(A[1], "queried_arrow")
        variants[f"only the side effect (head {mate} removed)"] = [dict(op="scale_output", layer=L, head=mate, gain=0.0)]
    res = {}
    for name, ops in variants.items():
        try:
            if ops:
                b.apply(ops)
            m = b.measure(); m["successor"] = {dm: b.rate(ev[dm], "s") for dm in ORD}; m["lookup"] = {dm: b.rate(ev[dm], "v") for dm in ORD}
        finally:
            b.restore()
        res[name] = m
        print(f"[octl] {name}: " + " ".join(f"{dm} {m['successor'][dm]:.2f}" for dm in ORD) + f" | seq {m['sequence_continuation']:.2f} | word lookup {m['word_lookup_accuracy']:.2f} | WikiText {m['wikitext_loss']:.3f}", flush=True)
    (OUT / "opus_program" / "controls.json").write_text(json.dumps(dict(model=os.environ.get("EVO_MODEL"), frozen=frozen, results=res), indent=1), encoding="utf-8")
    print("[octl] DONE", flush=True)


if __name__ == "__main__":
    main()
