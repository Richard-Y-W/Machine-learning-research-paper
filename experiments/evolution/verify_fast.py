"""Check that cached partial re-execution reproduces full evaluation exactly, and time both."""
import json
import time

from . import editlang
from .world import World


def main():
    w = World()
    d = w.diagnostics(); _, bank = w.dossier(w.base_dev, d)
    t = time.time(); w.build_cache(); tc = time.time() - t
    muts = {
        "L24 rank1": [{"op": "rank1", "tensor": "L24.mlp.down", "u": "err:L25", "v": "mlperr:L24", "alpha": -0.1}],
        "L27 head": [{"op": "scale", "tensor": "L27.attn.o", "head": 3, "factor": 0.5}],
        "final_norm": [{"op": "scale", "tensor": "final_norm", "index": [5, 9], "factor": 1.3}],
        "L3 + L26": [{"op": "scale", "tensor": "L3.mlp.down", "axis": "col", "index": [7], "factor": 0.0},
                     {"op": "scale", "tensor": "L26.ln2", "index": [1], "factor": 1.1}],
    }
    out = dict(cache_build_s=round(tc, 1))
    for name, ops in muts.items():
        m = editlang.resolve(ops, w.params, bank)
        k = editlang.min_layer(m, w.n_layers)
        with editlang.Applied(w.params, m, w.original):
            t = time.time(); full = w.eval_dev(); tf = time.time() - t
            t = time.time(); fast = w.eval_dev(min_layer=k); tp = time.time() - t
        diff = max(abs(full["fitness"] - fast["fitness"]), abs(full["protected"]["fitness"] - fast["protected"]["fitness"]),
                   abs(full["nll"] - fast["nll"]))
        out[name] = dict(min_layer=k, full_s=round(tf, 1), fast_s=round(tp, 1), max_abs_diff=diff,
                         full_fit=full["fitness"], fast_fit=fast["fitness"])
        print(name, out[name], flush=True)
    m = editlang.resolve(muts["L24 rank1"], w.params, bank)
    with editlang.Applied(w.params, m, w.original):
        t = time.time(); full = w.eval_test(0); tf = time.time() - t
        t = time.time(); fast = w.eval_test(24); tp = time.time() - t
    diff = max(abs(full["target"]["acc"] - fast["target"]["acc"]), abs(full["target"]["fitness"] - fast["target"]["fitness"]),
               abs(full["nll"] - fast["nll"]), abs(full["protected"]["fitness"] - fast["protected"]["fitness"]))
    out["test L24"] = dict(full_s=round(tf, 1), fast_s=round(tp, 1), max_abs_diff=diff)
    print("test L24", out["test L24"], flush=True)
    import psutil, os
    out["rss_gb"] = round(psutil.Process(os.getpid()).memory_info().rss / 1e9, 2); print("rss GB", out["rss_gb"])
    assert all(v["max_abs_diff"] < 1e-5 for k, v in out.items() if isinstance(v, dict)), out
    json.dump(out, open("results/evolution/verify_fast.json", "w"), indent=1)
    print("OK: partial re-execution matches full evaluation")


if __name__ == "__main__":
    main()
