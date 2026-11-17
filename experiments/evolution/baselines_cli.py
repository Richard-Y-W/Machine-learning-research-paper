"""Stage 1 baselines and Stage 2 genetics commands."""
import itertools
import json
import random
from pathlib import Path

import torch

RESULTS = Path("results/evolution")
RUNS = RESULTS / "runs"


def _world(target_split=None):
    from .world import World
    w = World()
    if target_split:
        # re-target dev fitness at another frozen split (used for ES sigma tuning only)
        from . import backend
        S = json.loads(Path("results/evolution/splits.json").read_text())
        w.dev_t = S[target_split]["items"]; w.enc["dev_target"] = backend.Encoded(w.tok, w.dev_t)
        w.base_dev = w.eval_dev(); w.n_evals = 0
    return w


def cmd_calibrate(a):
    from .baselines import build_calibration
    n = build_calibration(RUNS, RESULTS / "calls", RESULTS / "random_calibration.json",
                          arms=("E-LLM", "R-LLM", "E-LLM-nohist", "E-LLM-free"))
    print("valid LLM mutations in calibration:", n)


def cmd_es(a):
    from .baselines import run_es
    if a.tune:
        w = _world("calib_target")
        scores = {}
        for s in a.sigmas:
            r = run_es(w, RESULTS / "es" / f"tune_sigma{s}", a.tune_evals, s, seed=100)
            scores[s] = r["best"]["fitness"] if r["best"] else float("-inf")
        best = max(scores, key=scores.get)
        (RESULTS / "es" / "tuning.json").write_text(json.dumps(dict(scores=scores, chosen=best), indent=1))
        print("ES sigma tuning:", scores, "chosen", best)
        return
    sigma = json.loads((RESULTS / "es" / "tuning.json").read_text())["chosen"]
    w = _world()
    from .baselines import apply_gates
    from .loop import _test
    for seed in a.seeds:
        out = RESULTS / "es" / f"ES-56_seed{seed}"
        r = run_es(w, out, a.evals, sigma, seed)
        log = json.loads((out / "log.json").read_text())
        tests = {}
        for budget in a.test_budgets:     # best admissible within the first `budget` evaluations
            hist = [h for h in r["history"][:budget] if h["admissible"]]
            if not hist:
                continue
            bi = max(range(len(hist)), key=lambda i: hist[i]["fitness"])
            # recover x: rerun is unnecessary because best-so-far x is stored only for the overall best;
            # store x for every record instead (see run_es) and look it up here
            x = hist[bi]["x"]
            apply_gates(w, x)
            tests[str(budget)] = _test(w, None)
            w.reset_to_original()
        log["tests"] = tests
        (out / "log.json").write_text(json.dumps(log, indent=1))


def _prefix_genome(lineage_dir, gen):
    log = json.loads((lineage_dir / "log.json").read_text())
    genome = torch.load(lineage_dir / "genome.pt", weights_only=False) if (lineage_dir / "genome.pt").exists() else []
    n = sum(1 for g in log["generations"] if g["gen"] <= gen and g["accepted"] is not None)
    return genome[:n]


def cmd_dfe(a):
    from . import editlang
    from .baselines import RandomProposer
    w = _world()
    out_path = RESULTS / "dfe.json"
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    dirs = [d for arm in a.arms for d in sorted((RUNS / arm).iterdir()) if d.is_dir()]
    for d in dirs:
        for gen in a.gens:
            key = f"{d.parent.name}/{d.name}/g{gen}"
            if key in res or (gen == 0 and any(k.endswith("/g0") for k in res)):
                continue
            genome = _prefix_genome(d, gen)
            w.load_genome(genome)
            parent = w.eval_dev()
            w.build_cache()
            diag = w.diagnostics(); _, bank = w.dossier(parent, diag)
            prop = RandomProposer(RESULTS / "random_calibration.json", seed=hash(key) % 10**6)
            effects = []
            for _ in range(a.n):
                ops = prop.sample()
                try:
                    m = editlang.resolve(ops, w.params, bank)
                    with editlang.Applied(w.params, m, w.original):
                        ev = w.eval_dev(min_layer=editlang.min_layer(m, w.n_layers))
                    effects.append(dict(dfit=ev["fitness"] - parent["fitness"], dacc=ev["target"]["acc"] - parent["target"]["acc"],
                                        admissible=w.admissible(ev)))
                except editlang.InvalidEdit as e:
                    effects.append(dict(error=str(e)[:100]))
            res[key] = dict(genome_len=len(genome), parent_fitness=parent["fitness"], effects=effects)
            out_path.write_text(json.dumps(res, indent=1))
            ok = [e for e in effects if "dfit" in e]
            print(f"[dfe] {key}: beneficial {sum(e['dfit'] > 0 for e in ok)}/{len(ok)} "
                  f"adm-beneficial {sum(e['dfit'] > 0 and e['admissible'] for e in ok)}", flush=True)


CROSS_SYSTEM = """You are the CROSSOVER operator in an evolutionary search over the weights of a pretrained language model (Qwen3-0.6B-Base). Two independent lineages (A and B) each evolved from the same original checkpoint by accumulating weight-edit mutations, selected for higher accuracy on MMLU, ARC-Challenge, HellaSwag and CommonsenseQA while protecting other abilities. You are given each lineage's genome: its ordered accepted mutations, with the dev-fitness gain each produced when it was accepted (in its own lineage's context) and the layers/tensors it touches.

Propose exactly 6 RECOMBINANT genomes. Each recombinant is an ordered list of mutation ids drawn from A and B (e.g. ["A1","B1","A3"]); each id at most once. Mutations are applied in the listed order to the ORIGINAL checkpoint. Think about which edits are likely redundant (same component, same hypothesis), which may conflict (opposite pushes on the same tensor or direction), and which are complementary. Make the 6 recombinants diverse. Return JSON only: {"analysis": "<=100 words", "recombinants": [{"rationale": "...", "take": ["A1", ...]}, ...]}"""

CROSS_SCHEMA = {"type": "object", "properties": {
    "analysis": {"type": "string"},
    "recombinants": {"type": "array", "items": {"type": "object", "properties": {
        "rationale": {"type": "string"}, "take": {"type": "array", "items": {"type": "string"}}},
        "required": ["rationale", "take"], "additionalProperties": False}}},
    "required": ["analysis", "recombinants"], "additionalProperties": False}


def _genome_text(tag, lineage_dir):
    from . import editlang
    log = json.loads((lineage_dir / "log.json").read_text())
    genome = torch.load(lineage_dir / "genome.pt", weights_only=False)
    lines, i = [], 0
    for g in log["generations"]:
        if g["accepted"] is None:
            continue
        o = g["offspring"][g["accepted"]]
        lines.append(f"{tag}{i + 1} (gen {g['gen']}, {o['delta']}): {editlang.describe(genome[i])}")
        i += 1
    return "\n".join(lines), genome


def cmd_crossover(a):
    from . import editlang, editor
    from .baselines import genome_task_vector, ties_merge, apply_delta
    from .loop import _test, genome_min_layer, ev_brief
    w = _world()
    lineages = sorted(d for d in (RUNS / "E-LLM").iterdir() if d.is_dir())
    out_path = RESULTS / "crossover.json"
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    genomes = {d.name: torch.load(d / "genome.pt", weights_only=False) for d in lineages if (d / "genome.pt").exists()}
    names = sorted(genomes)

    def ev_genome(genome, test=True):
        w.load_genome(genome)
        ev = ev_brief(w.eval_dev())
        t = _test(w, genome) if test else None
        w.reset_to_original()
        return dict(dev=ev, test=t)

    # single-lineage finals (for additivity)
    for n in names:
        if f"single/{n}" not in res:
            res[f"single/{n}"] = ev_genome(genomes[n]); out_path.write_text(json.dumps(res, indent=1))
    for A, B in itertools.combinations(names, 2):
        for order, gen in ((f"{A}+{B}", genomes[A] + genomes[B]), (f"{B}+{A}", genomes[B] + genomes[A])):
            if f"union/{order}" not in res:
                res[f"union/{order}"] = ev_genome(gen); out_path.write_text(json.dumps(res, indent=1))
                print("[cross] union", order, res[f"union/{order}"]["test"]["target_acc"], flush=True)
        if f"ties/{A}|{B}" not in res:
            d = ties_merge([genome_task_vector(w, genomes[A]), genome_task_vector(w, genomes[B])])
            apply_delta(w, d)
            ml = min(genome_min_layer(w, genomes[A]), genome_min_layer(w, genomes[B]))
            res[f"ties/{A}|{B}"] = dict(dev=ev_brief(w.eval_dev()), test=_test(w, None) if ml < 16 else _test_ml(w, ml))
            w.reset_to_original(); out_path.write_text(json.dumps(res, indent=1))
    # LLM-mediated vs random crossover on disjoint pairs, equal evaluations (6 recombinants each)
    pairs = [(names[i], names[i + 1]) for i in range(0, len(names) - 1, 2)]
    for A, B in pairs:
        key = f"{A}|{B}"
        ta, ga = _genome_text("A", RUNS / "E-LLM" / A); tb, gb = _genome_text("B", RUNS / "E-LLM" / B)
        pool = {f"A{i + 1}": m for i, m in enumerate(ga)} | {f"B{i + 1}": m for i, m in enumerate(gb)}
        for kind in ("llm", "random"):
            if f"cross_{kind}/{key}" in res:
                continue
            if kind == "llm":
                user = f"Lineage A genome:\n{ta}\n\nLineage B genome:\n{tb}\n\nPropose exactly 6 recombinants."
                parsed, rec = _cross_call(user, f"cross_{A}_{B}")
                recs = [(r.get("take", []), r.get("rationale", "")) for r in (parsed or {}).get("recombinants", [])][:6]
            else:
                rng = random.Random(key)
                recs = []
                for _ in range(6):
                    take = [k for k in sorted(pool, key=lambda s: (s[0], int(s[1:]))) if rng.random() < 0.5] or [rng.choice(list(pool))]
                    rng.shuffle(take) if rng.random() < 0.5 else None
                    recs.append((take, "uniform random crossover"))
            evs = []
            for take, why in recs:
                if not take or any(t not in pool for t in take) or len(set(take)) != len(take):
                    evs.append(dict(take=take, error="invalid recombinant")); continue
                gen = [pool[t] for t in take]
                w.load_genome(gen); ev = w.eval_dev()
                evs.append(dict(take=take, rationale=why, dev=ev_brief(ev), admissible=w.admissible(ev)))
            ok = [e for e in evs if "dev" in e and e["admissible"]]
            best = max(ok, key=lambda e: e["dev"]["fitness"]) if ok else None
            if best:
                gen = [pool[t] for t in best["take"]]
                best["test"] = ev_genome(gen)["test"]
            res[f"cross_{kind}/{key}"] = dict(recombinants=evs, best=best)
            w.reset_to_original(); out_path.write_text(json.dumps(res, indent=1))
            print(f"[cross] {kind} {key}: best test acc {best and best['test']['target_acc']}", flush=True)


def _test_ml(w, ml):
    ev = w.eval_test(min_layer=ml)
    return dict(target_acc=ev["target"]["acc"], target_fitness=ev["target"]["fitness"], protected_acc=ev["protected"]["acc"],
                protected_fitness=ev["protected"]["fitness"], nll=ev["nll"],
                tasks={**{t: v["acc"] for t, v in ev["target"]["tasks"].items()}, **{t: v["acc"] for t, v in ev["protected"]["tasks"].items()}},
                item_correct_target=ev["item_correct_target"], item_correct_protected=ev["item_correct_protected"],
                item_logp_target=[round(x, 5) for x in ev["item_logp_target"]])


def _cross_call(user, label):
    from . import editor
    import anthropic, time
    archive = RESULTS / "calls" / "crossover"; archive.mkdir(parents=True, exist_ok=True)
    rid = editor.reserve(label, editor.worst_case((len(CROSS_SYSTEM) + len(user)) // 3))
    client = anthropic.Anthropic(max_retries=3)
    t0 = time.time(); usage = text = err = None
    try:
        with client.messages.stream(model=editor.MODEL, max_tokens=editor.MAX_TOKENS, system=CROSS_SYSTEM,
                                    messages=[{"role": "user", "content": user}],
                                    output_config={"effort": editor.EFFORT, "format": {"type": "json_schema", "schema": CROSS_SCHEMA}}) as s:
            msg = s.get_final_message()
        usage = msg.usage.model_dump(); text = next((b.text for b in msg.content if b.type == "text"), None)
        editor.settle(rid, usage)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"; editor.settle(rid, None, status="error_reservation_kept")
    rec = dict(label=label, usage=usage, cost_usd=editor.cost(usage) if usage else None, error=err, user_text=user,
               response_text=text, seconds=round(time.time() - t0, 1))
    (archive / f"{label}.json").write_text(json.dumps(rec, indent=1))
    try:
        return (json.loads(text) if text else None), rec
    except json.JSONDecodeError:
        return None, rec


def cmd_grad(a):
    """Reference only: gradient fine-tuning of layers >= first_layer on dev target items (headroom marker)."""
    from . import backend
    from .loop import _test, ev_brief
    w = _world()
    names = [n for n in w.editable if any(f"layers.{i}." in n for i in range(a.first_layer, w.n_layers)) or n == "model.norm.weight"]
    params = [w.params[n] for n in names]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=a.lr)
    w.build_cache()
    cache = w.cache["t"]; enc = w.enc["dev_target"]; lid = torch.tensor(w.lids)
    best, hist = None, []
    for step in range(1, a.steps + 1):
        opt.zero_grad()
        total = 0.0
        for bi, (b, (ids, mask, last)) in enumerate(zip(enc.batches, enc.tensors)):
            store = cache[bi]; h = store["h"][a.first_layer].clone()   # leave inference mode
            kw = {k: (v.clone() if torch.is_tensor(v) else tuple(t.clone() for t in v) if isinstance(v, tuple) else v)
                  for k, v in store["kw"].items()}
            for i in range(a.first_layer, w.n_layers):
                h = w.model.model.layers[i](h, **kw); h = h[0] if isinstance(h, tuple) else h
            h = w.model.model.norm(h)
            logits = w.model.lm_head(h[torch.arange(len(b)), last])[:, lid]
            loss = 0
            for r, i in enumerate(b):
                k = enc.items[i]["n_choices"]
                loss = loss - torch.log_softmax(logits[r, :k], -1)[enc.items[i]["label"]]
            loss = loss / len(enc.items); loss.backward(); total += float(loss)
        opt.step()
        if step % a.eval_every == 0:
            with torch.no_grad():
                ev = w.eval_dev()
            rec = dict(step=step, train_loss=total, dev=ev_brief(ev), admissible=w.admissible(ev)); hist.append(rec)
            print(f"[grad] step {step} loss {total:.4f} dev fit {ev['fitness']:.4f} adm {rec['admissible']}", flush=True)
            if rec["admissible"] and (best is None or ev["fitness"] > best["dev"]["fitness"]):
                best = dict(rec, test=_test(w, None))
    out = RESULTS / "grad_reference.json"
    out.write_text(json.dumps(dict(first_layer=a.first_layer, lr=a.lr, steps=a.steps, history=hist, best=best), indent=1))


def register(sub):
    sub.add_parser("calibrate")
    e = sub.add_parser("es")
    e.add_argument("--tune", action="store_true"); e.add_argument("--sigmas", type=float, nargs="+", default=[0.03, 0.1, 0.3])
    e.add_argument("--tune-evals", type=int, default=48); e.add_argument("--evals", type=int, default=300)
    e.add_argument("--seeds", type=int, nargs="+", default=[1, 2]); e.add_argument("--test-budgets", type=int, nargs="+", default=[60, 300])
    d = sub.add_parser("dfe")
    d.add_argument("--arms", nargs="+", default=["E-LLM", "E-LLM-free"]); d.add_argument("--gens", type=int, nargs="+", default=[0, 5, 10])
    d.add_argument("--n", type=int, default=30)
    sub.add_parser("crossover")
    g = sub.add_parser("grad")
    g.add_argument("--first-layer", type=int, default=20); g.add_argument("--lr", type=float, default=1e-5)
    g.add_argument("--steps", type=int, default=60); g.add_argument("--eval-every", type=int, default=5)


COMMANDS = {"calibrate": cmd_calibrate, "es": cmd_es, "dfe": cmd_dfe, "crossover": cmd_crossover, "grad": cmd_grad}
