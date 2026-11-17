"""Stage 0b follow-up: stronger end-of-thinking interventions (no API).

Stage 0b showed biases of +1..+6 change nothing: during thinking the end-of-thinking token sits
30-45 logits below the top token. This adds large biases, a hard forced stop at a fixed number of
thinking tokens, and the best prompt combined with a forced stop. Rows are appended to trivial.json.
"""
import json
import time

from .core import Runner
from .trivial import R_DIR, SUFFIXES

CONDITIONS = [("endthink_bias:20", dict(bias=20.0)), ("endthink_bias:30", dict(bias=30.0)),
              ("force_end:128", dict(force_at=128)), ("force_end:192", dict(force_at=192)),
              ("under100+force_end:192", dict(suffix=SUFFIXES["under100"], force_at=192)),
              ("under100:seed2", dict(suffix=SUFFIXES["under100"], seed="val-b"))]


def main():
    R = Runner()
    items = json.loads((R_DIR / "val_v1.json").read_text(encoding="utf-8"))["items"]
    path = R_DIR / "trivial.json"; res = json.loads(path.read_text(encoding="utf-8"))
    fam = [it["family"] for it in items]
    for name, kw in CONDITIONS:
        if name in res["rows"]:
            continue
        kw = dict(kw); seed = kw.pop("seed", "val")
        t0 = time.time(); r = R.score(items, think=True, max_new=512, seed=seed, **kw)
        byf = {f: sum(c and fi and k <= 256 for c, fi, k, g in zip(r["correct"], r["finished"], r["tokens"], fam) if g == f) / fam.count(f) for f in sorted(set(fam))}
        res["rows"][name] = dict(acc_at={str(k): v for k, v in r["acc_at"].items()}, median_tokens=r["median_tokens"], hit_cap=r["hit_cap"],
                                 acc256_by_family=byf, correct=r["correct"], tokens=r["tokens"], seconds=round(time.time() - t0))
        print(f"[trivial2] {name:24s} @256 {r['acc_at'][256]:.2f} @384 {r['acc_at'][384]:.2f} @512 {r['acc_at'][512]:.2f} "
              f"median tok {r['median_tokens']:4d} ({res['rows'][name]['seconds']}s)", flush=True)
        path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("[trivial2] DONE", flush=True)


if __name__ == "__main__":
    main()
