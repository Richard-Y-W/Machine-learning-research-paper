"""The hand-designed edit expressed in the primitive library and measured with the same bench, fit data and
confirmation prompts as Opus's programmes (so the comparison in the paper is like for like)."""
import json
import os
import random

from .bridge import A, OUT
from . import lookup2
from .lookup import KEYS, ORD
from .lookup3_confirm import NEW_KEYS
from .opus_program import Bench


def main():
    b = Bench()
    ops = [dict(op="attention_program", layer=A[0], head=A[1], query_role="final", key_role="queried_value", contrast="other_positions", ridge=1.0, rank=64, rope_band="low"),
           dict(op="scale_output", layer=A[0], head=A[1], gain=3.0)]
    lookup2.KEYS = NEW_KEYS; r2 = random.Random("confirm-v1|arrow_new_keys")
    ev = {dm: [lookup2.make(r2, b.ordv[dm], dm, "arrow") for _ in range(400)] for dm in ORD}; lookup2.KEYS = KEYS
    base = b.measure(); nat = {dm: b.rate(ev[dm], "s") for dm in ORD}
    b.apply(ops); m = b.measure(ops); conf = {dm: b.rate(ev[dm], "s") for dm in ORD}; b.restore()
    (OUT / "hand_ref.json").write_text(json.dumps(dict(model=os.environ.get("EVO_MODEL"), ops=ops, unedited=base, hand=m, confirm_native=nat, confirm_hand=conf), indent=1), encoding="utf-8")
    print(f"[hand] {os.environ.get('EVO_MODEL')}: hand-designed {m}\n[hand] confirm {conf}", flush=True)


if __name__ == "__main__":
    main()
