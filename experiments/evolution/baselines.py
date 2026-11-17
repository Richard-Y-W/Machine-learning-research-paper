"""Non-LLM comparators and genetics tests.

RandomProposer: magnitude-matched random mutations in the SAME edit language.
  It reproduces the empirical distribution of the LLM's proposals over op type,
  tensor part, layer, number of ops, index-list length, ref family, and
  magnitudes (factor / alpha / set values), but draws the *content* (which
  neurons/heads/indices, which ref layer, which sign/magnitude from the pool)
  at random. This isolates the LLM's specific choices from the power of the
  edit language and the data-derived reference vectors it shares.
"""
import json
import random
import re
from collections import Counter
from pathlib import Path

import torch

from . import editlang

DIM_FAMILY = {"err": 1024, "mean": 1024, "tok": 1024, "letter": 1024, "attnerr": 2048, "mlperr": 3072}
SHAPES = {"attn.q": (2048, 1024), "attn.k": (1024, 1024), "attn.v": (1024, 1024), "attn.o": (1024, 2048),
          "mlp.gate": (3072, 1024), "mlp.up": (3072, 1024), "mlp.down": (1024, 3072),
          "attn.qnorm": (128,), "attn.knorm": (128,), "ln1": (1024,), "ln2": (1024,), "final_norm": (1024,)}


def _split_alias(alias):
    if alias == "final_norm":
        return None, "final_norm"
    m = re.fullmatch(r"L(\d+)\.(.+)", alias)
    return int(m.group(1)), m.group(2)


def _ref_family(ref):
    return ref.split(":")[0] if ref else None


def build_calibration(runs_root, calls_root, out_path, arms=("E-LLM", "E-LLM-free", "R-LLM")):
    """Collect valid LLM-proposed mutations (by matching run logs to archived calls)."""
    muts = []
    for arm in arms:
        for log_path in sorted(Path(runs_root, arm).glob("*/log.json")):
            log = json.loads(log_path.read_text())
            for g in log["generations"]:
                call = Path(calls_root, arm, g["meta"]["call"] + ".json")
                if not call.exists():
                    continue
                resp = json.loads(json.loads(call.read_text())["response_text"] or "{}")
                props = resp.get("mutations", [])
                for o in g["offspring"]:
                    if not o.get("error") and o["k"] < len(props):
                        muts.append(props[o["k"]]["ops"])
    Path(out_path).write_text(json.dumps(dict(n_mutations=len(muts), mutations=muts), indent=1))
    return len(muts)


class RandomProposer:
    def __init__(self, calibration_path, seed=0):
        cal = json.loads(Path(calibration_path).read_text())
        self.muts = cal["mutations"]
        if not self.muts:
            raise ValueError("empty calibration")
        self.ops = [op for m in self.muts for op in m]
        self.n_ops = [len(m) for m in self.muts]
        self.layers = [l for op in self.ops for l in [_split_alias(op["tensor"])[0]] if l is not None]
        self.factors = {k: [op["factor"] for op in self.ops if op["op"] == k and "factor" in op] for k in ("scale",)}
        self.alphas = [op["alpha"] for op in self.ops if op["op"] == "rank1" and "alpha" in op]
        self.set_vals = [e[-1] for op in self.ops if op["op"] == "set" for e in op.get("entries", [])]
        self.rng = random.Random(seed)

    def _ref(self, family, dim, part_for_sv):
        r = self.rng
        if family in ("err", "mean", "attnerr", "mlperr"):
            return f"{family}:L{r.choice(self.layers)}"
        if family == "letter":
            return f"letter:{r.choice('ABCDE')}"
        if family == "sv":
            return f"sv:{part_for_sv}:{r.randrange(8)}:{{side}}"
        if family == "unit":
            return f"unit:{r.randrange(dim)}"
        return f"letter:{r.choice('ABCDE')}"   # tok refs: replaced by a random letter embedding

    def sample_op(self):
        r = self.rng
        tpl = r.choice(self.ops)
        layer, part = _split_alias(tpl["tensor"])
        if layer is not None:
            layer = r.choice(self.layers)
        alias = "final_norm" if part == "final_norm" else f"L{layer}.{part}"
        shape = SHAPES[part]
        kind = tpl["op"]
        if kind in ("scale", "ablate"):
            op = {"op": kind, "tensor": alias}
            if kind == "scale":
                op["factor"] = r.choice(self.factors["scale"] or [1.0])
            if "head" in tpl and part.startswith("attn.") and len(shape) == 2:
                n_heads = (shape[1] if part == "attn.o" else shape[0]) // 128
                op["head"] = r.randrange(n_heads)
            elif tpl.get("axis") == "all":
                op["axis"] = "all"
            else:
                axis = tpl.get("axis", "row") if len(shape) == 2 else "row"
                lim = shape[0] if axis == "row" else shape[1]
                n = max(1, min(len(tpl.get("index") or [0]), lim))
                op["axis"] = axis; op["index"] = sorted(r.sample(range(lim), n))
            return op
        if kind == "rank1":
            sides = []
            for side, dim in (("u", shape[0]), ("v", shape[1])):
                fam = _ref_family(tpl.get(side))
                if fam not in ("sv", "unit") and DIM_FAMILY.get(fam) != dim:
                    fam = [f for f, d in DIM_FAMILY.items() if d == dim][0]
                ref = self._ref(fam, dim, alias).replace("{side}", "left" if side == "u" else "right")
                sides.append(ref)
            alpha = r.choice(self.alphas or [0.05])
            return {"op": "rank1", "tensor": alias, "u": sides[0], "v": sides[1], "alpha": alpha}
        if kind == "set":
            n = max(1, min(len(tpl.get("entries", [])), 64))
            if len(shape) == 1:
                ents = [[r.randrange(shape[0]), r.choice(self.set_vals or [0.0])] for _ in range(n)]
            else:
                ents = [[r.randrange(shape[0]), r.randrange(shape[1]), r.choice(self.set_vals or [0.0])] for _ in range(n)]
            return {"op": "set", "tensor": alias, "entries": ents}
        return {"op": "revert", "tensor": alias}

    def sample(self):
        return [self.sample_op() for _ in range(self.rng.choice(self.n_ops))]

    def __call__(self, world, dossier_text, gens, label):
        return [(self.sample(), "random") for _ in range(6)], dict(call=label, cost_usd=0.0, error=None, stop=None)


# ---------------------------------------------------------------- CMA-ES

def gate_names(n_layers):
    return [f"model.layers.{i}.self_attn.o_proj.weight" for i in range(n_layers)] + \
           [f"model.layers.{i}.mlp.down_proj.weight" for i in range(n_layers)]


def run_es(world, out_dir, evals, sigma0, seed, split="dev"):
    """sep-CMA-ES over 56 per-layer output gates (attn.o and mlp.down scales).
    Penalized fitness: dev target fitness minus 10x constraint violation."""
    import cma
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    names = gate_names(world.n_layers)
    es = cma.CMAEvolutionStrategy([1.0] * len(names), sigma0,
                                  {"seed": seed + 1, "CMA_diagonal": True, "bounds": [0.0, 2.0], "verbose": -9})
    hist, best = [], None
    n = 0
    while n < evals:
        xs = es.ask()
        fs = []
        for x in xs:
            if n >= evals:
                break
            with torch.no_grad():
                for nm, g in zip(names, x):
                    world.params[nm].copy_(world.original[nm] * float(g))
            ev = world.eval_dev(); n += 1
            b = world.base_dev
            viol = max(0.0, b["protected"]["fitness"] - 0.02 - ev["protected"]["fitness"]) + max(0.0, ev["nll"] - b["nll"] - 0.02)
            pen = ev["fitness"] - 10 * viol
            fs.append(-pen)
            rec = dict(n=n, fitness=ev["fitness"], penalized=pen, admissible=world.admissible(ev), acc=ev["target"]["acc"],
                       protected=ev["protected"]["fitness"], nll=ev["nll"], x=[round(float(v), 6) for v in x])
            hist.append(rec)
            if rec["admissible"] and (best is None or ev["fitness"] > best["fitness"]):
                best = dict(rec)
        if len(fs) == len(xs):
            es.tell(xs, fs)
        world.reset_to_original()
        (out_dir / "log.json").write_text(json.dumps(dict(sigma0=sigma0, seed=seed, history=hist, best=best), indent=1))
        print(f"[es {out_dir.name}] evals {n} best {best['fitness'] if best else None}", flush=True)
    return dict(history=hist, best=best)


def apply_gates(world, x):
    with torch.no_grad():
        for nm, g in zip(gate_names(world.n_layers), x):
            world.params[nm].copy_(world.original[nm] * float(g))


# ---------------------------------------------------------------- genetics

def genome_task_vector(world, genome):
    world.load_genome(genome)
    d = editlang.genome_delta(world.params, world.original)
    world.reset_to_original()
    return d


def ties_merge(deltas, keep=0.2):
    """TIES-merging of task vectors: trim to top-`keep` magnitude, elect sign, disjoint mean."""
    names = set().union(*[d.keys() for d in deltas])
    out = {}
    for n in names:
        vs = []
        for d in deltas:
            v = d.get(n)
            if v is None:
                continue
            flat = v.abs().flatten()
            k = max(1, int(keep * flat.numel()))
            thr = flat.kthvalue(flat.numel() - k + 1).values
            vs.append(torch.where(v.abs() >= thr, v, torch.zeros_like(v)))
        S = torch.stack(vs)
        sign = torch.sign(S.sum(0))
        agree = (torch.sign(S) == sign) & (S != 0)
        cnt = agree.sum(0).clamp_min(1)
        out[n] = (S * agree).sum(0) / cnt
    return out


def apply_delta(world, delta):
    world.reset_to_original()
    with torch.no_grad():
        for n, d in delta.items():
            world.params[n].add_(d)
