"""Evolutionary loop: (1+lambda) elitist lineages with LLM or random mutation.

Arms
  inherit=True : offspring of the accepted child become the next parents (lineage).
  inherit=False: every generation mutates the ORIGINAL (no heredity); the arm's
                 result is the best admissible offspring seen (best-of selection).
Logs are written after every generation; a rerun resumes from the log without
repeating completed editor calls.
"""
import json
import time
from pathlib import Path

import torch

from . import editlang, editor

LAMBDA = 6


def fmt_delta(ev, parent):
    return (f"dfit {ev['fitness'] - parent['fitness']:+.4f}, dprot {ev['protected']['fitness'] - parent['protected']['fitness']:+.4f}, "
            f"dNLL {ev['nll'] - parent['nll']:+.4f}")


def history_text(gens, full_last=4):
    if not gens:
        return "## Lineage history\nNone: this is the first generation (parent = original checkpoint)."
    out = ["## Lineage history (oldest first). Deltas are offspring minus that generation's parent, on DEV."]
    for g in gens:
        recent = g["gen"] >= gens[-1]["gen"] - full_last + 1
        out.append(f"Gen {g['gen']}: parent fitness {g['parent_fitness']:.4f}; "
                   + (f"ACCEPTED offspring {g['accepted']}" if g["accepted"] is not None else "no offspring accepted"))
        for o in g["offspring"]:
            if not recent and o["k"] != g["accepted"]:
                continue
            tag = "INVALID: " + o["error"][:120] if o.get("error") else (
                ("ACCEPTED" if o["k"] == g["accepted"] else ("inadmissible" if not o["admissible"] else "rejected")) + "; " + o["delta"])
            out.append(f"  [{o['k']}] {o.get('desc') or o.get('raw_ops', '')[:200]} -> {tag}")
            if recent and o.get("rationale"):
                out.append(f"      rationale: {o['rationale'][:160]}")
    return "\n".join(out)


def ev_brief(ev):
    return dict(fitness=ev["fitness"], target_acc=ev["target"]["acc"], protected=ev["protected"]["fitness"],
                protected_acc=ev["protected"]["acc"], nll=ev["nll"],
                tasks={t: v["acc"] for t, v in ev["target"]["tasks"].items()})


class LLMProposer:
    def __init__(self, archive_dir, history_aware=True):
        self.archive_dir = Path(archive_dir); self.history_aware = history_aware

    def __call__(self, world, dossier_text, gens, label):
        hist = history_text(gens) if self.history_aware else "## Lineage history\nNot shown."
        user = dossier_text + "\n\n" + hist + "\n\nPropose exactly 6 diverse mutations of the current parent."
        parsed, rec = editor.call(user, label, self.archive_dir)
        muts = (parsed or {}).get("mutations", [])[:LAMBDA]
        meta = dict(call=label, cost_usd=rec["cost_usd"], error=rec["error"], stop=rec["stop_reason"],
                    analysis=(parsed or {}).get("analysis"))
        return [(m.get("ops"), m.get("rationale", "")) for m in muts], meta


def evaluate_offspring(world, parent_ev, proposals, bank, constrained):
    results = []
    for k, (ops, rationale) in enumerate(proposals):
        rec = dict(k=k, rationale=rationale, raw_ops=json.dumps(ops)[:2000])
        try:
            mut = editlang.resolve(ops, world.params, bank)
            rec["desc"] = editlang.describe(mut)
            with editlang.Applied(world.params, mut, world.original) as ap:
                ev = world.eval_dev(min_layer=editlang.min_layer(mut, world.n_layers))
                rec["rel_change"] = {n.replace("model.layers.", "L"): round(r, 4) for n, r in ap.rel.items()}
            rec.update(ev=ev_brief(ev), delta=fmt_delta(ev, parent_ev), admissible=world.admissible(ev),
                       improves=ev["fitness"] > parent_ev["fitness"])
            rec["_mut"], rec["_ev"] = mut, ev
        except (editlang.InvalidEdit, KeyError, TypeError, ValueError, IndexError, RuntimeError) as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        results.append(rec)
    while len(results) < LAMBDA:  # missing proposals count as invalid offspring
        results.append(dict(k=len(results), error="missing proposal"))
    return results


def run_lineage(world, out_dir, proposer, generations, inherit=True, constrained=True, test_at=(), seed_label=""):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log_path, genome_path = out_dir / "log.json", out_dir / "genome.pt"
    log = json.loads(log_path.read_text()) if log_path.exists() else dict(
        inherit=inherit, constrained=constrained, generations=[], tests={}, split_hash=world.split_hash)
    genome = torch.load(genome_path, weights_only=False) if genome_path.exists() else []
    world.load_genome(genome if inherit else [])
    parent_ev = world.eval_dev()
    if "base" not in log:
        log["base"] = ev_brief(world.base_dev)
    best_seen = log.get("best_seen")
    if "0" not in log["tests"] and 0 in test_at:
        log["tests"]["0"] = _test(world, [])
        _save(log, log_path)
    diag_cache = None
    for gen in range(len(log["generations"]) + 1, generations + 1):
        t0 = time.time()
        if inherit or diag_cache is None:
            diag = world.diagnostics(); dossier, bank = world.dossier(parent_ev, diag)
            diag_cache = (dossier, bank)
        dossier, bank = diag_cache
        if world.cache is None:
            world.build_cache()
        gens_for_prompt = log["generations"] if inherit else []
        proposals, meta = proposer(world, dossier, gens_for_prompt, f"{seed_label}g{gen:02d}")
        res = evaluate_offspring(world, parent_ev, proposals, bank, constrained)
        cands = [r for r in res if "_ev" in r and world.accept(r["_ev"], parent_ev, constrained)]
        best = max(cands, key=lambda r: r["_ev"]["fitness"]) if cands else None
        g = dict(gen=gen, parent_fitness=parent_ev["fitness"], parent=ev_brief(parent_ev), meta=meta,
                 accepted=best["k"] if (best and inherit) else None, seconds=None,
                 n_valid=sum("_ev" in r for r in res), n_improving=sum(bool(r.get("improves")) for r in res),
                 n_admissible_improving=len(cands))
        if inherit and best is not None:
            with editlang.Applied(world.params, best["_mut"], world.original) as ap:
                ap.keep = True
            world.cache = None
            genome.append(best["_mut"])
            parent_ev = best["_ev"]
            torch.save(genome, genome_path)
        if not inherit:
            for r in res:
                if "_ev" in r and world.admissible(r["_ev"]) and (best_seen is None or r["_ev"]["fitness"] > best_seen["fitness"]):
                    best_seen = dict(fitness=r["_ev"]["fitness"], gen=gen, k=r["k"], ev=r["ev"])
                    torch.save([r["_mut"]], out_dir / "best_mutation.pt")
            log["best_seen"] = best_seen
        g["offspring"] = [{k: v for k, v in r.items() if not k.startswith("_")} for r in res]
        g["seconds"] = round(time.time() - t0, 1)
        g["genome_len"] = len(genome)
        log["generations"].append(g)
        if gen in test_at and inherit:
            log["tests"][str(gen)] = _test(world, genome)
        _save(log, log_path)
        print(f"[{out_dir.name}] gen {gen}: parent {g['parent_fitness']:.4f} valid {g['n_valid']}/6 "
              f"improving {g['n_improving']} admissible+ {g['n_admissible_improving']} accepted {g['accepted']} "
              f"-> {parent_ev['fitness']:.4f} ({g['seconds']}s, ${meta.get('cost_usd') or 0:.4f})", flush=True)
    if not inherit and best_seen is not None and "final" not in log["tests"]:
        bm = torch.load(out_dir / "best_mutation.pt", weights_only=False)
        world.load_genome(bm)
        log["tests"]["final"] = _test(world, bm)
        world.reset_to_original()
        _save(log, log_path)
    return log


def genome_min_layer(world, genome):
    return min([editlang.min_layer(m, world.n_layers) for m in genome], default=world.n_layers)


def _test(world, genome=None):
    ev = world.eval_test(min_layer=0 if genome is None else genome_min_layer(world, genome))
    return dict(target_acc=ev["target"]["acc"], target_fitness=ev["target"]["fitness"],
                protected_acc=ev["protected"]["acc"], protected_fitness=ev["protected"]["fitness"], nll=ev["nll"],
                tasks={**{t: v["acc"] for t, v in ev["target"]["tasks"].items()},
                       **{t: v["acc"] for t, v in ev["protected"]["tasks"].items()}},
                item_correct_target=ev["item_correct_target"], item_correct_protected=ev["item_correct_protected"],
                item_logp_target=[round(x, 5) for x in ev["item_logp_target"]])


def _save(log, path):
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(log, indent=1, allow_nan=False)); tmp.replace(path)
