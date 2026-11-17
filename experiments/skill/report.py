"""Detailed report for the skill-programming pilot -> results/skill/REPORT.md (+ figures).

    python -m experiments.skill.report
"""
import json
from pathlib import Path

import numpy as np

from .pilot import HELD, PRACTICE, T_PRACTICE, T_HELD, LETTERS, succ

R = Path("results/skill")
SUCC_HEADS = {(21, 6), (22, 8), (25, 8), (25, 9)}
KEYS = [("practice", "Practice (seen letters, seen formats)"), ("held_letters", "Held-out letters (PRIMARY)"),
        ("held_template", "Held-out format"), ("held_both", "Held-out letters + format")]


def load(p):
    p = R / p
    return json.loads(p.read_text()) if p.exists() else None


def verdict(r, b0=None):
    if r is None:
        return "n/a"
    reg = r.get("test_regression")
    ok_reg = reg is None or b0 is None or (reg["protected"] >= b0["protected"] - 0.02 and reg["nll"] <= b0["nll"] + 0.02)
    if r["held_letters"]["acc"] >= 0.8 and ok_reg:
        return "WORKS"
    if r["practice"]["acc"] >= 0.8 and r["held_letters"]["acc"] < 0.5:
        return "PARTIAL (fact insertion)"
    if r["held_letters"]["acc"] < 0.3:
        return "FAILS"
    return "intermediate"


def touched(ops_json):
    out = []
    try:
        ops = json.loads(ops_json)
    except (json.JSONDecodeError, TypeError):
        return out
    for op in ops:
        t = op.get("tensor", "")
        layer = int(t[1:].split(".")[0]) if t.startswith("L") else None
        out.append((op.get("op"), t, layer, op.get("head")))
    return out


def main():
    base = load("baselines.json") or {}
    final = load("final.json") or {}
    theory = {k: load(f"theory/{k}.json") for k in ("feasibility", "latent_competence", "successor_heads")}
    L = ["# Skill-programming pilot: results", "",
         "Protocol: `docs/skill-pilot-protocol.md` (frozen before any result). Skill **ENC**: `ENC: c ->` -> ` d` (next letter).",
         f"Held-out letters (never shown to the editor): **{' '.join(HELD)}**. Held-out format: `{T_HELD!r}`.", ""]
    # ---------------- headline table
    rows = []
    if "base" in base: rows.append(("Base model (zero-shot)", base["base"]))
    if "fewshot3" in base: rows.append(("Few-shot prompting (3 examples in prompt)", base["fewshot3"]))
    if "steering" in base: rows.append((f"Activation steering (layer {base['steering']['layer']}, c={base['steering']['coef']})", base["steering"]))
    if "finetune" in base: rows.append((f"Fine-tuning ({base['finetune']['n_train_examples']} examples, lr {base['finetune']['lr']}, {base['finetune']['steps']} steps)", base["finetune"]))
    for name, r in sorted(final.items()):
        rows.append((f"**WSP chain {name}** ({r['genome_len']} accepted edits)", r))
    L += ["## Headline results (exact-match accuracy, argmax over the full vocabulary)", "",
          "| Method | " + " | ".join(k[1] for k in KEYS) + " | Verdict |", "|---|" + "---|" * (len(KEYS) + 1)]
    for name, r in rows:
        cells = [f"{r[k]['acc']:.2f}" if k in r else "–" for k, _ in KEYS]
        L.append(f"| {name} | " + " | ".join(cells) + f" | {verdict(r, base.get('base', {}).get('test_regression')) if name.startswith('**WSP') else ''} |")
    L += ["", "Verdict rules (pre-registered): WORKS = held-out letters >= .80 with no regression; PARTIAL = practice >= .80 "
          "and held-out < .50; FAILS = held-out < .30.", ""]
    # ---------------- regression
    b0 = base.get("base", {}).get("test_regression")
    if b0:
        L += ["## Regression on untouched abilities (TEST sets)", "", "| Method | protected acc | protected fitness (Δ) | WikiText NLL (Δ) |", "|---|---|---|---|"]
        for name, r in rows:
            t = r.get("test_regression")
            if t:
                L.append(f"| {name} | {t['protected_acc']:.3f} | {t['protected']:.4f} ({t['protected'] - b0['protected']:+.4f}) | "
                         f"{t['nll']:.4f} ({t['nll'] - b0['nll']:+.4f}) |")
        L += ["", "Prompting and steering act only at inference on trigger prompts, so they have no weight-level regression to report.", ""]
    # ---------------- per-letter
    for name, r in sorted(final.items()):
        hl = r["held_letters"]["correct"]
        n = len(HELD)
        per = {c: (hl[i], hl[n + i]) for i, c in enumerate(HELD)}
        L += [f"### Chain {name}: held-out letters, per template", "", "| letter | target | " + " | ".join(f"`{t}`" for t in T_PRACTICE) + " |", "|---|---|---|---|"]
        for c in HELD:
            L.append(f"| {c} | {succ(c)} | " + " | ".join("✓" if v else "✗" for v in per[c]) + " |")
        L.append("")
    # ---------------- trajectories + mechanism
    chains = sorted((R / "chains").glob("*/log.json")) if (R / "chains").exists() else []
    cost_total = 0.0
    traj = {}
    for p in chains:
        log = json.loads(p.read_text()); name = p.parent.name
        accs, fits = [log["base"]["acc"]], [log["base"]["fitness"]]
        mech = []
        for g in log["generations"]:
            cost_total += g["meta"].get("cost_usd") or 0
            if g["accepted"] is not None:
                o = [x for x in g["offspring"] if x["k"] == g["accepted"]][0]
                accs.append(o["ev"]["acc"]); fits.append(o["ev"]["fitness"])
                for op, t, layer, head in touched(o.get("raw_ops")):
                    mech.append(dict(gen=g["gen"], op=op, tensor=t, layer=layer, head=head,
                                     successor_head=(layer, head) in SUCC_HEADS and ".attn." in t))
            else:
                accs.append(accs[-1]); fits.append(fits[-1])
        traj[name] = (accs, fits)
        L += [f"## Chain {name}: what the editor changed", "",
              f"Generations: {len(log['generations'])}; accepted: {sum(g['accepted'] is not None for g in log['generations'])}; "
              f"valid proposals: {sum(g['n_valid'] for g in log['generations'])}/{6 * len(log['generations'])}.", ""]
        if mech:
            layers = [m["layer"] for m in mech if m["layer"] is not None]
            sh = [m for m in mech if m["successor_head"]]
            L.append(f"Accepted ops: {len(mech)}. Layer range {min(layers)}–{max(layers)}. Ops touching a successor head "
                     f"(L21h6, L22h8, L25h8, L25h9): **{len(sh)}** ({', '.join(sorted({m['tensor'] + ' h' + str(m['head']) for m in sh})) or 'none'}).")
            kinds = {}
            for m in mech:
                kinds[m["op"]] = kinds.get(m["op"], 0) + 1
            L.append("Op types: " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())) + ".")
            L += ["", "| gen | op | tensor | head |", "|---|---|---|---|"] + [f"| {m['gen']} | {m['op']} | {m['tensor']} | {m['head'] if m['head'] is not None else ''} |" for m in mech]
        L.append("")
        last = log["generations"][-1]["meta"].get("analysis") if log["generations"] else None
        if last:
            L += [f"Editor's final analysis: _{last}_", ""]
    # ---------------- theory vs outcome
    L += ["## Theory and its predictions", ""]
    if theory["feasibility"]:
        F = theory["feasibility"]["layers"]
        best_h = max(r["full"][1] for r in F.values()); sh = np.mean([r["shuffled_held_mean"] for r in F.values()])
        L.append(f"- **Interpolation barrier:** affine readout maps fit on practice letters reach practice 1.00 but held-out "
                 f"max {best_h:.2f} over all {len(F)} layers and ranks 1–8 (shuffled-target floor {sh:.3f}). Readout/lookup edits cannot generalize.")
    if theory["latent_competence"]:
        lc = theory["latent_competence"]
        L.append(f"- **Latent competence:** sequence contexts already compute the successor (alphabet-primed: practice "
                 f"{lc['alpha_list/practice'][0]:.2f}, held {lc['alpha_list/held'][0]:.2f}; 3-letter run: {lc['seq3/practice'][0]:.2f}/{lc['seq3/held'][0]:.2f}); "
                 f"natural-language prompts do not (0.00).")
    if theory["successor_heads"]:
        top = theory["successor_heads"][:4]
        L.append("- **Successor heads (weights only):** " + ", ".join(f"L{t['layer']}h{t['head']} succ {t['succ']:.2f} (held {t['succ_held']:.2f})" for t in top)
                 + "; chance 0.04.")
    L += ["- **Prediction:** routing the trigger into the successor heads should reach held-out letters ≈ 0.6–0.8; "
          "per-letter writes should give ≈ 0.", ""]
    L += [f"## Cost", "", f"Editor spend for the pilot chains: ${cost_total:.2f} (ledger, with 1.15 safety factor).", ""]
    # ---------------- figures
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        (R / "figures").mkdir(exist_ok=True)
        if traj:
            fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
            colors = ["#2a78d6", "#eb6834", "#1baf7a"]
            for i, (name, (a, f)) in enumerate(sorted(traj.items())):
                ax[0].plot(range(len(a)), a, color=colors[i % 3], lw=2, marker="o", ms=4, label=f"chain {name}")
                ax[1].plot(range(len(f)), f, color=colors[i % 3], lw=2, marker="o", ms=4, label=f"chain {name}")
            ax[0].set_title("Practice accuracy by generation", loc="left", fontsize=10); ax[0].set_xlabel("generation")
            ax[1].set_title("Practice fitness (mean log p correct)", loc="left", fontsize=10); ax[1].set_xlabel("generation")
            for a_ in ax:
                a_.spines[["top", "right"]].set_visible(False); a_.grid(color="#e6e6e6"); a_.legend(frameon=False, fontsize=8)
            fig.tight_layout(); fig.savefig(R / "figures" / "trajectory.png", dpi=160); plt.close(fig)
            L += ["![trajectory](figures/trajectory.png)", ""]
        if rows:
            fig, ax = plt.subplots(figsize=(9, 3.8))
            names = [n.replace("**", "") for n, _ in rows]; x = np.arange(len(names)); w = 0.2
            for j, (k, lab) in enumerate(KEYS):
                ax.bar(x + (j - 1.5) * w, [r[k]["acc"] if k in r else 0 for _, r in rows], w, label=lab.split(" (")[0],
                       color=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"][j], edgecolor="white", lw=1)
            ax.set_xticks(x); ax.set_xticklabels([n.split(" (")[0] for n in names], fontsize=8, rotation=15)
            ax.set_ylim(0, 1); ax.set_ylabel("exact-match accuracy"); ax.legend(frameon=False, fontsize=8, ncol=4)
            ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e6e6e6")
            ax.set_title("ENC skill: seen vs held-out generalization by method", loc="left", fontsize=10)
            fig.tight_layout(); fig.savefig(R / "figures" / "methods.png", dpi=160); plt.close(fig)
            L += ["![methods](figures/methods.png)", ""]
    except ImportError:
        pass
    (R / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
