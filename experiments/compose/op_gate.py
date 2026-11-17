"""First gate only for candidate second operations: native competence with the item as the final token."""
import json
import os
import random

from .bridge import Lab, OUT
from .second_op import OPS


def main():
    OUT.mkdir(parents=True, exist_ok=True); lab = Lab(); tok = lab.tok; res = {}
    one = lambda s: len(tok(" " + s, add_special_tokens=False)["input_ids"]) == 1
    for shots_n in (3, 8):
        for op, pairs in OPS.items():
            pairs = [(a, b) for a, b in pairs if one(a) and one(b) and lab.first(a) != lab.first(b)]
            rr = random.Random(f"op-gate|{op}|{shots_n}"); hit = n = 0
            for a, b in pairs:
                for _ in range(3):
                    shots = rr.sample([p for p in pairs if p[0] != a], min(shots_n, len(pairs) - 1))
                    lg, _ = lab.forward(" ".join(f"{x} {y}" for x, y in shots) + f" {a}"); hit += int(lg.argmax()) == lab.first(b); n += 1
            res[f"{op}_{shots_n}shot"] = dict(pairs=len(pairs), prompts=n, native_accuracy=round(hit / n, 3))
            print(f"[opgate] {os.environ.get('EVO_MODEL')} {op} {shots_n}-shot: {hit / n:.2f} ({len(pairs)} pairs)", flush=True)
    json.dump(res, open(OUT / "op_gate.json", "w"), indent=1)


if __name__ == "__main__":
    main()
