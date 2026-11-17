"""Stage 0a: screen each program family for the reasoning-vs-budget gap (no edits, no API)."""
import json
import random
import time
from pathlib import Path

from . import tasks
from .core import Runner

OUT = Path("results/amortize/screen.json")


def main(n=20):
    R = Runner(); res = {}
    for fam in tasks.FAMILIES:
        t0 = time.time()
        items = tasks.sample([fam], n, "screen-v1")
        shots = [tasks.make(fam, random.Random(f"shots|{fam}|{i}")) for i in range(3)]
        th = R.score(items, think=True, max_new=1024, seed=f"screen|{fam}", keep_outs=True)
        fs = R.score(items, think=False, max_new=64, seed=f"screen-fs|{fam}", shots=shots)
        keep = th["acc_at"][1024] >= 0.80 and th["acc_at"][256] <= 0.40
        res[fam] = dict(n=n, acc_at=th["acc_at"], median_tokens=th["median_tokens"], hit_cap=th["hit_cap"], direct_fewshot=fs["acc"],
                        n_lines=items[0]["n_lines"], keep=keep, seconds=round(time.time() - t0),
                        samples=[dict(code=it["code"], a=it["a"], out=o) for it, o in list(zip(items, th["outs"]))[:2]])
        print(f"[screen] {fam:4s} think@256 {th['acc_at'][256]:.2f} @384 {th['acc_at'][384]:.2f} @512 {th['acc_at'][512]:.2f} "
              f"@1024 {th['acc_at'][1024]:.2f} median tok {th['median_tokens']:4d} hit-cap {th['hit_cap']:.2f} "
              f"fewshot-direct {fs['acc']:.2f} keep {keep} ({res[fam]['seconds']}s)", flush=True)
        OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("[screen] DONE", flush=True)


if __name__ == "__main__":
    main()
