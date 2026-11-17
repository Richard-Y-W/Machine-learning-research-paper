"""Commands: prepare | smoke | lineage | es | dfe | crossover | analyze."""
import argparse
import json
import sys
import time
from pathlib import Path

RESULTS = Path("results/evolution")


def cmd_prepare(a):
    from . import data, backend
    from .world import TARGET_TASKS, PROTECTED_TASKS, SPLITS_PATH
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(backend.MODEL_NAME)
    splits = {k: [] for k in ("dev_target", "dev_protected", "test_target", "test_protected", "calib_target")}
    for t in TARGET_TASKS:
        s = data.split_items(data.load_pool(t, tok), {"dev": 40, "test": 250, "calib": 24})
        splits["dev_target"] += s["dev"]; splits["test_target"] += s["test"]; splits["calib_target"] += s["calib"]
    for t in PROTECTED_TASKS:
        s = data.split_items(data.load_pool(t, tok), {"dev": 12, "test": 100})
        splits["dev_protected"] += s["dev"]; splits["test_protected"] += s["test"]
    splits["wiki_dev"] = data.load_wikitext_chunks(tok, 4, 128, offset_chunks=0)
    splits["wiki_test"] = data.load_wikitext_chunks(tok, 32, 128, offset_chunks=100)
    hashes = data.write_split_manifest(SPLITS_PATH, splits)
    print(json.dumps({k: (len(v), hashes[k][:12]) for k, v in splits.items()}, indent=1))


def cmd_smoke(a):
    """Offline end-to-end check with scripted proposals (no API)."""
    from .world import World
    from . import loop
    w = World()
    print("base dev:", json.dumps(loop.ev_brief(w.base_dev)))
    t = time.time(); w.eval_dev(); print(f"one dev eval: {time.time() - t:.1f}s")
    t = time.time(); d = w.diagnostics(); text, bank = w.dossier(w.base_dev, d); print(f"dossier: {time.time() - t:.1f}s, {len(text)} chars")
    Path(RESULTS / "smoke_dossier.txt").write_text(text, encoding="utf-8")

    def scripted(world, dossier, gens, label):
        return [([{"op": "scale", "tensor": "L27.mlp.down", "axis": "all", "factor": 0.95}], "shrink last MLP"),
                ([{"op": "rank1", "tensor": "L26.mlp.down", "u": "err:L27", "v": "mlperr:L26", "alpha": -0.05}], "steer"),
                ([{"op": "ablate", "tensor": "L20.attn.o", "head": 3}], "ablate head"),
                ([{"op": "set", "tensor": "final_norm", "entries": [[0, 1.0]]}], "set"),
                ([{"op": "scale", "tensor": "L5.attn.q", "head": 99, "factor": 1.0}], "invalid head"),
                ([{"op": "revert", "tensor": "L3.ln1"}], "revert")], dict(cost_usd=0)
    log = loop.run_lineage(w, RESULTS / "smoke_lineage", scripted, generations=2, inherit=True, test_at=())
    print("smoke offspring errors:", [o.get("error") for o in log["generations"][0]["offspring"]])


def cmd_lineage(a):
    from .world import World
    from . import loop
    w = World()
    from .baselines import RandomProposer
    llm = loop.LLMProposer(RESULTS / "calls" / a.arm, history_aware=not a.no_history) if a.proposer == "llm" else None
    for i, lid in enumerate(a.lineages):
        name = f"{a.arm}_{lid}"
        prop = llm or RandomProposer(RESULTS / "random_calibration.json", seed=a.seed * 1000 + i)
        loop.run_lineage(w, RESULTS / "runs" / a.arm / name, prop, a.generations, inherit=not a.no_inherit,
                         constrained=not a.unconstrained, test_at=tuple(a.test_at), seed_label=name + "_")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m experiments evo")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare"); sub.add_parser("smoke")
    q = sub.add_parser("lineage")
    q.add_argument("--arm", required=True); q.add_argument("--lineages", nargs="+", required=True)
    q.add_argument("--generations", type=int, default=8); q.add_argument("--proposer", choices=["llm", "random"], default="llm")
    q.add_argument("--no-inherit", action="store_true"); q.add_argument("--unconstrained", action="store_true")
    q.add_argument("--no-history", action="store_true"); q.add_argument("--seed", type=int, default=0)
    q.add_argument("--test-at", type=int, nargs="*", default=[0, 2, 4, 6, 8])
    from . import baselines_cli
    baselines_cli.register(sub)
    a = p.parse_args(argv)
    {"prepare": cmd_prepare, "smoke": cmd_smoke, "lineage": cmd_lineage, **baselines_cli.COMMANDS}[a.cmd](a)


if __name__ == "__main__":
    main()
