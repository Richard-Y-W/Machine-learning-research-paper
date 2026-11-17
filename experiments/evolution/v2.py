"""Evolution v2: rotating selection, specialist lineages, gene-level breeding.

Protocol: docs/evolution-v2-protocol.md. Reuses the v1 organism, edit language,
editor, test sets and ledger. Selection data rotates every generation and is
drawn from a pool disjoint from every v1 split; the editor never sees test data.

    python -m experiments.evolution.v2 pool
    python -m experiments.evolution.v2 lineage --arm G-LLM --units 1 2 --generations 15 --test-at 5 10 15
    python -m experiments.evolution.v2 lineage --arm S-LLM --units 1_mmlu 1_arc_challenge ... --generations 10 --test-at 10
    python -m experiments.evolution.v2 breed --set 1 --source LLM
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from . import backend, data, editlang, editor
from .loop import LAMBDA, LLMProposer, evaluate_offspring, ev_brief, _test, _save, genome_min_layer
from .world import World, TARGET_TASKS, SPLITS_PATH

V2 = Path("results/evolution/v2")
POOL_PATH = V2 / "pool.json"
# Amendment A1 (2026-11-14, before any v2 API call): samples 4x larger to raise realized heritability
# (v1 h2 ~0.27 at 160 items); sparse cache levels keep RAM bounded.
SAMPLE_GENERALIST = 160     # items per target task per generation (640 total)
SAMPLE_SPECIALIST = 640     # items of the single task per generation
BREED_PER_TASK = 160
V2_CACHE_LEVELS = [12, 16, 19, 21, 23, 25, 26, 27]
BREED_BUDGET = 24
TIES_LAMBDAS = (0.5, 0.75, 1.0, 1.25, 1.5)
TASK_SHORT = {"mmlu": "mmlu", "arc_challenge": "arc", "hellaswag": "hs", "commonsense_qa": "csqa"}

ROTATION_NOTE = ("## v2 conditions\nThe DEV target items are RE-SAMPLED every generation from a large pool, and the parent is "
                 "re-scored on each new sample. Edits that only fit particular items will not keep winning; aim for general capability.")


def specialist_note(task):
    return (f"\nThis lineage is a SPECIALIST: target fitness is {task} ONLY (the other target benchmarks are not scored for "
            f"this lineage). The protected constraints still apply.")


# ------------------------------------------------------------------ data

def build_pool():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(backend.MODEL_NAME)
    S = json.loads(SPLITS_PATH.read_text())
    used = {it["uid"] for k, v in S.items() if not k.startswith("wiki") for it in v["items"]}
    pool = {t: [it for it in data.load_pool(t, tok) if it["uid"] not in used] for t in TARGET_TASKS}
    hashes = data.write_split_manifest(POOL_PATH, pool)
    print(json.dumps({t: (len(v), hashes[t][:12]) for t, v in pool.items()}, indent=1))


def load_pool():
    return {k: v["items"] for k, v in json.loads(POOL_PATH.read_text()).items()}


def draw(pool, tasks, n_per_task, key):
    rng = random.Random(hashlib.sha256(key.encode()).hexdigest())
    out = []
    for t in tasks:
        out += rng.sample(pool[t], min(n_per_task, len(pool[t])))
    return out


# ------------------------------------------------------------------ world with a replaceable selection sample

class RotatingWorld(World):
    def build_cache(self):
        lv = V2_CACHE_LEVELS
        self.cache = dict(t=backend.build_cache(self.model, self.enc["dev_target"], lv),
                          p=backend.build_cache(self.model, self.enc["dev_protected"], lv),
                          w=backend.build_cache(self.model, self.wiki_chunks, lv))

    def set_sample(self, items, genome):
        """Install a new dev-target sample; score the ORIGINAL on it (for the dossier); reload `genome`."""
        self.dev_t = items
        self.enc["dev_target"] = backend.Encoded(self.tok, items)
        self.reset_to_original()
        ev0 = self.eval_dev()
        self.base_dev = dict(self.base_dev, target=ev0["target"], fitness=ev0["fitness"])
        if genome:
            self.load_genome(genome)
        return ev0

    def restore_original_keep_cache(self):
        with torch.no_grad():
            for n in self.editable:
                self.params[n].copy_(self.original[n])

    def eval_genes(self, muts):
        """Evaluate original+muts on the current sample. Requires a cache built on the ORIGINAL weights."""
        editlang.replay(self.params, muts, self.original)
        ev = self.eval_dev(min_layer=genome_min_layer(self, muts)) if muts else self.eval_dev(min_layer=self.n_layers)
        self.restore_original_keep_cache()
        return ev


# ------------------------------------------------------------------ lineages

def unit_tasks(arm, unit):
    if arm.startswith("S-"):
        task = unit.split("_", 1)[1]
        assert task in TARGET_TASKS, unit
        return [task], SAMPLE_SPECIALIST
    return list(TARGET_TASKS), SAMPLE_GENERALIST


def run_lineage_v2(world, arm, unit, proposer, generations, test_at=()):
    name = f"{arm}_{unit}"
    tasks, n_per = unit_tasks(arm, unit)
    out = V2 / "runs" / arm / name; out.mkdir(parents=True, exist_ok=True)
    log_path, genome_path = out / "log.json", out / "genome.pt"
    log = json.loads(log_path.read_text()) if log_path.exists() else dict(
        arm=arm, unit=unit, tasks=tasks, n_per_task=n_per, generations=[], tests={}, genes=[],
        split_hash=world.split_hash, pool_hash=json.loads(POOL_PATH.read_text())[tasks[0]]["sha256"])
    genome = torch.load(genome_path, weights_only=False) if genome_path.exists() else []
    pool = load_pool()
    note = ROTATION_NOTE + (specialist_note(tasks[0]) if arm.startswith("S-") else "")
    for gen in range(len(log["generations"]) + 1, generations + 1):
        t0, n0 = time.time(), world.n_evals
        sample = draw(pool, tasks, n_per, f"{name}|g{gen}")
        ev0 = world.set_sample(sample, genome)
        parent_ev = world.eval_dev()
        diag = world.diagnostics(); dossier, bank = world.dossier(parent_ev, diag)
        world.build_cache()
        proposals, meta = proposer(world, dossier + "\n\n" + note, log["generations"], f"{name}_g{gen:02d}")
        res = evaluate_offspring(world, parent_ev, proposals, bank, True)
        cands = [r for r in res if "_ev" in r and world.accept(r["_ev"], parent_ev, True)]
        best = max(cands, key=lambda r: r["_ev"]["fitness"]) if cands else None
        g = dict(gen=gen, parent_fitness=parent_ev["fitness"], original_on_sample=ev0["fitness"], parent=ev_brief(parent_ev),
                 meta=meta, accepted=best["k"] if best else None, n_valid=sum("_ev" in r for r in res),
                 n_improving=sum(bool(r.get("improves")) for r in res), n_admissible_improving=len(cands))
        if best is not None:
            with editlang.Applied(world.params, best["_mut"], world.original) as ap:
                ap.keep = True
            world.cache = None
            genome.append(best["_mut"])
            torch.save(genome, genome_path)
            log["genes"].append(dict(gid=f"{name}:g{gen}", gen=gen, task=tasks[0] if len(tasks) == 1 else "all",
                                     desc=best.get("desc"), rationale=best.get("rationale", ""), raw_ops=best.get("raw_ops"),
                                     dfit_birth=best["_ev"]["fitness"] - parent_ev["fitness"]))
            parent_ev = best["_ev"]
        g["offspring"] = [{k: v for k, v in r.items() if not k.startswith("_")} for r in res]
        g["seconds"] = round(time.time() - t0, 1); g["n_evals"] = world.n_evals - n0; g["genome_len"] = len(genome)
        log["generations"].append(g)
        if gen in test_at:
            world.load_genome(genome)
            log["tests"][str(gen)] = _test(world, genome)
        _save(log, log_path)
        print(f"[{name}] gen {gen}: parent {g['parent_fitness']:.4f} (orig {ev0['fitness']:.4f}) valid {g['n_valid']}/6 "
              f"admissible+ {g['n_admissible_improving']} accepted {g['accepted']} ({g['seconds']}s, ${meta.get('cost_usd') or 0:.4f})",
              flush=True)
    return log


# ------------------------------------------------------------------ breeding

BREED_SYSTEM = """You are the BREEDER in an evolutionary study of a pretrained language model (Qwen3-0.6B-Base). Several SPECIALIST lineages were evolved separately, each selected on ONE benchmark (MMLU, ARC-Challenge, HellaSwag or CommonsenseQA) by accumulating weight edits ("genes"). Your job is to recombine genes from all specialists into ONE GENERALIST model that is good on all four benchmarks at once, without harming protected abilities (ARC-Easy, OpenBookQA, PIQA, Winogrande, SciQ, WikiText NLL).

A recombinant is a subset of gene ids. Its genes are applied to the ORIGINAL model in the listed order (later edits act on earlier ones). Genes from the same specialist were selected in sequence, so a gene may depend on earlier genes of its lineage; genes from different specialists may conflict when they edit the same tensors.

For each gene you see: its source task, generation of origin, its edit, the evolver's rationale, and an ASSAY: the change in mean log-probability of the correct answer on each of the four benchmarks when that gene ALONE is applied to the original model, plus whether it alone keeps the protected constraints. Assays measure single genes; combinations can interact.

Each round you propose exactly 6 recombinants; all are scored on a 4-benchmark selection sample (fitness = mean over benchmarks of mean log-probability of the correct letter) and checked against the protected constraints. You see the results of earlier rounds. The final outcome is accuracy on a large hidden test set, so prefer combinations that are plausibly general rather than ones that exploit sample noise. Make the 6 recombinants diverse (different hypotheses about which genes combine well) and learn from earlier rounds.

Return JSON only: "analysis" (<= 100 words) and "recombinants": exactly 6 objects, each with "genes" (list of gene ids, in application order) and a short "rationale"."""

BREED_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis": {"type": "string"},
        "recombinants": {"type": "array", "items": {
            "type": "object",
            "properties": {"genes": {"type": "array", "items": {"type": "string"}}, "rationale": {"type": "string"}},
            "required": ["genes", "rationale"], "additionalProperties": False}},
    },
    "required": ["analysis", "recombinants"],
    "additionalProperties": False,
}


def collect_genes(source, set_id):
    """Genes of one specialist set: list of dicts with gid, task, gen, desc, rationale, mut."""
    genes = []
    for task in TARGET_TASKS:
        d = V2 / "runs" / f"S-{source}" / f"S-{source}_{set_id}_{task}"
        log = json.loads((d / "log.json").read_text())
        genome = torch.load(d / "genome.pt", weights_only=False) if (d / "genome.pt").exists() else []
        assert len(genome) == len(log["genes"]), d
        for meta, mut in zip(log["genes"], genome):
            genes.append(dict(meta, short=f"{TASK_SHORT[task]}{meta['gen']}", mut=mut, specialist=task))
    return genes


def _fit(ev):
    return ev["fitness"]


class Breeding:
    def __init__(self, world, source, set_id):
        self.w, self.source, self.set_id = world, source, set_id
        self.dir = V2 / "breed" / f"{source}_set{set_id}"; self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "results.json"
        self.res = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.genes = collect_genes(source, set_id)
        self.by_id = {g["short"]: g for g in self.genes}
        sample = draw(load_pool(), TARGET_TASKS, BREED_PER_TASK, f"breed|set{set_id}")   # same sample for both gene sources
        world.set_sample(sample, [])
        world.build_cache()                       # ORIGINAL-weight cache on the breeding sample
        self.base = world.eval_dev(min_layer=world.n_layers)
        self.order = [g["short"] for g in self.genes]

    def save(self):
        self.path.write_text(json.dumps(self.res, indent=1, default=float))

    def ev_subset(self, ids):
        ids = sorted(set(ids), key=self.order.index)           # canonical order: by specialist, then generation
        ev = self.w.eval_genes([self.by_id[i]["mut"] for i in ids])
        return dict(genes=ids, dev=ev_brief(ev), task_logp={t: v["logp"] for t, v in ev["target"]["tasks"].items()},
                    admissible=self.w.admissible(ev))

    # -------- assay (shared by all operators, not charged to any budget)
    def assay(self):
        if "assay" in self.res:
            return self.res["assay"]
        out = {}
        base_t = {t: v["logp"] for t, v in self.base["target"]["tasks"].items()}
        for g in self.genes:
            r = self.ev_subset([g["short"]])
            out[g["short"]] = dict(dfit=r["dev"]["fitness"] - self.base["fitness"], admissible=r["admissible"],
                                   dtask={t: r["task_logp"][t] - base_t[t] for t in base_t})
        self.res["assay"] = out; self.save()
        return out

    def gene_table(self):
        A = self.assay()
        lines = ["gene | source task | gen | edit | evolver rationale | assay dlogp mmlu/arc/hs/csqa | alone admissible"]
        for g in self.genes:
            a = A[g["short"]]; d = a["dtask"]
            lines.append(f"{g['short']} | {g['specialist']} | {g['gen']} | {(g.get('desc') or '')[:220]} | {(g.get('rationale') or '')[:140]} | "
                         + "/".join(f"{d[t]:+.3f}" for t in TARGET_TASKS) + f" | {'yes' if a['admissible'] else 'NO'}")
        return "\n".join(lines)

    def _finish(self, key, evals, extra=None):
        ok = [e for e in evals if e.get("admissible")]
        best = max(ok, key=lambda e: e["dev"]["fitness"]) if ok else None
        rec = dict(evals=evals, best=best, n_evals=len(evals), **(extra or {}))
        if best is not None:
            if "genes" in best and best.get("kind") != "delta":
                muts = [self.by_id[i]["mut"] for i in best["genes"]]
                self.w.load_genome(muts); rec["test"] = _test(self.w, muts)
            else:
                self._apply_delta(best["coef"], best["merge"]); rec["test"] = _test(self.w, [g["mut"] for g in self.genes])
            self.w.reset_to_original(); self.w.build_cache()
        self.res[key] = rec; self.save()
        print(f"[breed {self.source} set{self.set_id}] {key}: best dev {best and best['dev']['fitness']:.4f} "
              f"test {rec.get('test', {}).get('target_acc')}", flush=True)
        return rec

    # -------- operators
    def llm_breed(self, rep):
        key = f"llm/{rep}"
        if key in self.res:
            return self.res[key]
        table = self.gene_table(); evals, history = [], []
        archive = Path("results/evolution/calls/v2_breed")
        for rnd in range(1, BREED_BUDGET // LAMBDA + 1):
            user = (f"## Genes (base model fitness on the selection sample: {self.base['fitness']:.4f})\n{table}\n\n"
                    + ("## Earlier rounds\n" + "\n".join(history) if history else "## Earlier rounds\nNone.")
                    + f"\n\nRound {rnd}: propose exactly 6 recombinants.")
            parsed, rec = editor.call(user, f"breed_{self.source}_set{self.set_id}_r{rep}_{rnd}", archive,
                                      system=BREED_SYSTEM, schema=BREED_SCHEMA)
            recs = (parsed or {}).get("recombinants", [])[:LAMBDA]
            while len(recs) < LAMBDA:
                recs.append(dict(genes=[], rationale="missing"))
            for r in recs:
                ids = r.get("genes", [])
                if not ids or any(i not in self.by_id for i in ids):
                    e = dict(genes=ids, error="invalid recombinant", rationale=r.get("rationale", ""), round=rnd)
                else:
                    e = dict(self.ev_subset(ids), rationale=r.get("rationale", ""), round=rnd)
                evals.append(e)
                history.append(f"[r{rnd}] {','.join(e['genes'])} -> " + (e["error"] if "error" in e else
                               f"fitness {e['dev']['fitness']:.4f} ({e['dev']['fitness'] - self.base['fitness']:+.4f} vs base), "
                               f"{'admissible' if e['admissible'] else 'INADMISSIBLE'}"))
        return self._finish(key, evals)

    def rand_breed(self, seed):
        key = f"rand/{seed}"
        if key in self.res:
            return self.res[key]
        sizes = [len(e["genes"]) for k, v in self.res.items() if k.startswith("llm/") for e in v["evals"] if e.get("genes")]
        sizes = sizes or list(range(1, len(self.genes) + 1))
        rng = random.Random(f"rand|{self.source}|{self.set_id}|{seed}")
        evals = [self.ev_subset(rng.sample(self.order, min(rng.choice(sizes), len(self.order)))) for _ in range(BREED_BUDGET)]
        return self._finish(key, evals, dict(size_source="llm" if len(sizes) != len(self.genes) else "uniform"))

    def greedy_breed(self):
        key = "greedy"
        if key in self.res:
            return self.res[key]
        A = self.assay()
        rank = sorted(self.order, key=lambda i: (not A[i]["admissible"], -A[i]["dfit"]))
        evals = []
        n_prefix = min(len(rank), BREED_BUDGET // 2)
        for k in range(1, n_prefix + 1):
            evals.append(dict(self.ev_subset(rank[:k]), step=f"prefix{k}"))
        ok = [e for e in evals if e["admissible"]]
        cur = max(ok, key=lambda e: e["dev"]["fitness"]) if ok else dict(genes=[], dev=dict(fitness=self.base["fitness"]))
        for i in rank:
            if len(evals) >= BREED_BUDGET:
                break
            if i in cur["genes"]:
                continue
            e = dict(self.ev_subset(cur["genes"] + [i]), step=f"add {i}")
            evals.append(e)
            if e["admissible"] and e["dev"]["fitness"] > cur["dev"]["fitness"]:
                cur = e
        return self._finish(key, evals)

    def union(self):
        if "union" in self.res:
            return self.res["union"]
        return self._finish("union", [self.ev_subset(self.order)])

    # -------- task-vector merges
    def _deltas(self):
        if not hasattr(self, "_D"):
            self._D = {}
            for task in TARGET_TASKS:
                muts = [g["mut"] for g in self.genes if g["specialist"] == task]
                self.w.load_genome(muts)
                self._D[task] = editlang.genome_delta(self.w.params, self.w.original)
            self.w.reset_to_original(); self.w.build_cache()
        return self._D

    def _apply_delta(self, coef, merge):
        from .baselines import ties_merge
        D = self._deltas()
        self.w.restore_original_keep_cache()
        with torch.no_grad():
            if merge == "ties":
                for n, d in ties_merge([D[t] for t in TARGET_TASKS]).items():
                    self.w.params[n].add_(coef[0] * d)
            else:
                for t, c in zip(TARGET_TASKS, coef):
                    for n, d in D[t].items():
                        self.w.params[n].add_(c * d)

    def _ev_delta(self, coef, merge):
        self._apply_delta(coef, merge)
        ev = self.w.eval_dev(min_layer=genome_min_layer(self.w, [g["mut"] for g in self.genes]) if self.genes else self.w.n_layers)
        self.w.restore_original_keep_cache()
        return dict(kind="delta", merge=merge, coef=[float(c) for c in coef], dev=ev_brief(ev), admissible=self.w.admissible(ev))

    def ties(self):
        if "ties" in self.res:
            return self.res["ties"]
        return self._finish("ties", [self._ev_delta([lam], "ties") for lam in TIES_LAMBDAS])

    def cma_merge(self, seed):
        key = f"cma/{seed}"
        if key in self.res:
            return self.res[key]
        import cma
        es = cma.CMAEvolutionStrategy([0.75] * 4, 0.3, {"seed": seed + 1, "bounds": [0.0, 1.5], "verbose": -9, "popsize": 6})
        evals = []
        while len(evals) < BREED_BUDGET:
            xs = es.ask(); fs = []
            for x in xs:
                e = self._ev_delta(x, "linear"); evals.append(e)
                fs.append(-(e["dev"]["fitness"] - (0 if e["admissible"] else 10)))
            es.tell(xs, fs)
        return self._finish(key, evals[:BREED_BUDGET])


def specialist_tests(world, source, set_id):
    out = {}
    for task in TARGET_TASKS:
        log = json.loads((V2 / "runs" / f"S-{source}" / f"S-{source}_{set_id}_{task}" / "log.json").read_text())
        out[task] = log["tests"].get("10")
    return out


# ------------------------------------------------------------------ CLI

def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m experiments.evolution.v2")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pool")
    sub.add_parser("smoke")
    q = sub.add_parser("lineage")
    q.add_argument("--arm", required=True, choices=["G-LLM", "G-RAND", "S-LLM", "S-RAND"])
    q.add_argument("--units", nargs="+", required=True); q.add_argument("--generations", type=int, required=True)
    q.add_argument("--test-at", type=int, nargs="*", default=[])
    b = sub.add_parser("breed")
    b.add_argument("--set", type=int, required=True); b.add_argument("--source", choices=["LLM", "RAND"], required=True)
    b.add_argument("--reps", type=int, default=3, help="seeds for RAND-breed and CMA-merge")
    b.add_argument("--llm-reps", type=int, default=2, help="LLM-breed replicates (budget)")
    a = p.parse_args(argv)
    if a.cmd == "pool":
        return build_pool()
    skip = V2 / "SKIP_MERGING"     # user dropped merging (2026-11-14); delete this file to re-enable
    if skip.exists() and ((a.cmd == "lineage" and a.arm.startswith("S-")) or a.cmd == "breed"):
        print(f"skipped ({skip} present): merging arms dropped by user decision"); return
    w = RotatingWorld()
    if a.cmd == "smoke":     # offline end-to-end check: random proposer, no API calls
        from .baselines import RandomProposer
        cal = Path("results/evolution/random_calibration.json")
        if skip.exists():      # merging dropped: smoke-test the generalist chain path only
            run_lineage_v2(w, "G-SMOKE", "1", RandomProposer(cal, seed=901), 2, test_at=(2,))
            print("SMOKE_OK (chain path)"); return
        for i, t in enumerate(TARGET_TASKS):
            run_lineage_v2(w, "S-SMOKE", f"1_{t}", RandomProposer(cal, seed=900 + i), 2, test_at=(2,))
        br = Breeding(w, "SMOKE", 1)
        print("genes:", [g["short"] for g in br.genes])
        if not br.genes:
            print("SMOKE_OK (no genes accepted; breeding skipped)"); return
        br.assay(); br.union(); br.ties(); br.rand_breed(1); br.cma_merge(1); br.greedy_breed()
        print("SMOKE_OK")
        return
    if a.cmd == "lineage":
        from .baselines import RandomProposer
        llm = LLMProposer(Path("results/evolution/calls/v2") / a.arm) if a.arm.endswith("LLM") else None
        for i, u in enumerate(a.units):
            prop = llm or RandomProposer(Path("results/evolution/random_calibration.json"),
                                         seed=int(hashlib.sha256(f"{a.arm}|{u}".encode()).hexdigest()[:8], 16))
            run_lineage_v2(w, a.arm, u, prop, a.generations, tuple(a.test_at))
    elif a.cmd == "breed":
        br = Breeding(w, a.source, a.set)
        br.assay(); br.union(); br.ties()
        for r in range(1, a.llm_reps + 1):
            br.llm_breed(r)
        for s in range(1, a.reps + 1):
            br.rand_breed(s); br.cma_merge(s)
        br.greedy_breed()


if __name__ == "__main__":
    main()
