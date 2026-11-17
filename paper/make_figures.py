"""Figures for the preprint. Reads results/compose*/ JSON files; writes paper/figures/*.pdf."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results"); F = Path("paper/figures"); F.mkdir(parents=True, exist_ok=True)
DOMS = ["letters", "numwords", "days", "months", "ordinals"]; LAB = ["letters", "number\nwords", "weekdays", "months", "ordinals"]
INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"; BLUE, GRAY, ORANGE, GREEN = "#2f6fb5", "#b8bec7", "#d9822b", "#3c8d5a"
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "pdf.fonttype": 42})


def style(ax, ylab=None):
    ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True); ax.tick_params(length=0)
    if ylab:
        ax.set_ylabel(ylab)


def fig_main():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, (name, path) in zip(axes, (("Qwen3-0.6B-Base", R / "compose/lookup3_confirm.json"), ("Qwen3-1.7B-Base", R / "compose_1p7b/lookup3_confirm.json"))):
        c = json.loads(path.read_text())["arrow_new_keys"]; x = range(len(DOMS)); w = 0.36
        nat = [c[d]["native_successor"] for d in DOMS]; ed = [c[d]["edited_successor"] for d in DOMS]
        ax.bar([i - w / 2 - 0.01 for i in x], nat, w, color=GRAY, label="unedited"); ax.bar([i + w / 2 + 0.01 for i in x], ed, w, color=BLUE, label="edited")
        for i, (a, b) in enumerate(zip(nat, ed)):
            ax.text(i - w / 2 - 0.01, a + 0.02, f"{a:.2f}", ha="center", va="bottom", fontsize=7, color=MUTED); ax.text(i + w / 2 + 0.01, b + 0.02, f"{b:.2f}", ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_xticks(list(x)); ax.set_xticklabels(LAB, fontsize=8); ax.set_ylim(0, 1); ax.set_title(name, fontsize=9, color=INK, loc="left"); style(ax)
    axes[0].set_ylabel("answers with the successor\nof the looked-up value"); axes[0].legend(frameon=False, fontsize=8, loc="upper left", ncol=2, bbox_to_anchor=(0, 1.02))
    fig.tight_layout(); fig.savefig(F / "main_result.pdf"); plt.close(fig)


def fig_controls():
    c = json.loads((R / "compose/lookup3_controls.json").read_text()); l3 = json.loads((R / "compose/lookup3.json").read_text())
    mean = lambda r: sum(r[d]["successor"] for d in DOMS) / len(DOMS)
    hosts = [mean(v) for k, v in c["host"].items() if k != "h6"]
    rows = [("unedited model", mean(c["native"]), GRAY), ("edit (successor head, gain 3)", mean(c["edit"]), BLUE),
            ("same attention in other heads (mean of 6)", sum(hosts) / len(hosts), ORANGE), ("attention fitted to a wrong position", mean(l3["control_wrong_target_gain3.0"]), ORANGE),
            ("random attention, same spectrum (mean of 3)", sum(mean(v) for v in c["structural_null"]) / 3, ORANGE), ("gain only, native attention", mean(l3["control_gain_only_gain3.0"]), ORANGE),
            ("edit with another head's output projection", mean(l3["control_wrong_output_projection"]), ORANGE), ("edit, neighbour head sharing keys removed", mean(c["neighbour_removed_edit"]), GREEN)]
    fig, ax = plt.subplots(figsize=(6.2, 2.7)); y = list(range(len(rows)))[::-1]
    ax.barh(y, [r[1] for r in rows], 0.62, color=[r[2] for r in rows])
    for yi, r in zip(y, rows):
        ax.text(r[1] + 0.012, yi, f"{r[1]:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8); ax.set_xlim(0, 0.85); ax.set_xlabel("successor rate (mean of five value types)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True); ax.tick_params(length=0)
    fig.tight_layout(); fig.savefig(F / "controls.pdf"); plt.close(fig)


def fig_gain():
    c = json.loads((R / "compose/learning_cmp.json").read_text()); mean = lambda r: sum(r[d]["successor"] for d in DOMS) / len(DOMS); mm = lambda r: sum(r[d]["margin"] for d in DOMS) / len(DOMS)
    gs = sorted(c["closed_form"], key=float); x = [float(g) for g in gs]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    axes[0].plot(x, [mean(c["closed_form"][g]) for g in gs], color=BLUE, lw=2, marker="o", ms=4); axes[0].set_ylim(0, 1); style(axes[0], "successor rate")
    axes[1].plot(x, [mm(c["closed_form"][g]) for g in gs], color=BLUE, lw=2, marker="o", ms=4); axes[1].axhline(0, color=MUTED, lw=0.6); style(axes[1], "successor $-$ copy logit")
    base = c["native"]["wiki_nll"]
    axes[2].plot(x, [100 * (c["closed_form"][g]["wiki_nll"] / base - 1) for g in gs], color=BLUE, lw=2, marker="o", ms=4); style(axes[2], "WikiText loss increase (%)")
    for ax in axes:
        ax.set_xlabel("output gain")
    fig.tight_layout(); fig.savefig(F / "gain.pdf"); plt.close(fig)


def fig_locate():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2), sharey=True)
    for ax, (name, path) in zip(axes, (("Qwen3-0.6B-Base", R / "compose_0p6b_check/find_head.json"), ("Qwen3-1.7B-Base", R / "compose_1p7b/find_head.json"))):
        f = json.loads(path.read_text()); o = f["oracle"]; labs = [f"L{r['layer']}H{r['head']}" for r in o]; v = [r["two_step"] for r in o]
        cols = [BLUE if (r["layer"], r["head"]) == tuple(f["successor_head"]) else GRAY for r in o]
        ax.bar(range(len(o)), v, 0.62, color=cols)
        for i, x in enumerate(v):
            ax.text(i, x + 0.02, f"{x:.2f}", ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_xticks(range(len(o))); ax.set_xticklabels(labs, fontsize=7.5); ax.set_ylim(0, 1.1); ax.set_title(name, fontsize=9, loc="left"); style(ax)
        ax.set_xlabel("candidate heads (ranked by ablation)")
    axes[0].set_ylabel("two-step answers under\nthe oracle value patch")
    fig.tight_layout(); fig.savefig(F / "locate.pdf"); plt.close(fig)


def fig_design():
    runs = [("Qwen3-0.6B-Base, run 1", R / "compose/opus_program/log.json", R / "compose/hand_ref.json"), ("Qwen3-0.6B-Base, run 2", R / "compose_run2/opus_program/log.json", R / "compose/hand_ref.json"),
            ("Qwen3-1.7B-Base", R / "compose_1p7b/opus_program/log.json", R / "compose_1p7b/hand_ref.json")]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)
    for ax, (name, lp, hp) in zip(axes, runs):
        g = json.loads(lp.read_text()); h = json.loads(hp.read_text()); base = g["base"]["wikitext_loss"]
        ax.axvspan(15, 40, color="#f3f4f6", zorder=0); ax.axhline(g["base"]["dev_successor_rate"], color=MUTED, lw=0.6, ls=(0, (3, 3)))
        for r in g["rounds"]:
            for c in r["candidates"]:
                if c.get("measured"):
                    m = c["measured"]; ax.scatter(100 * (m["wikitext_loss"] / base - 1), m["dev_successor_rate"], s=16 + 7 * r["round"], color=BLUE, alpha=0.25 + 0.15 * r["round"], linewidths=0, zorder=3)
        fz = g["frozen"]["measured"]; ax.scatter(100 * (fz["wikitext_loss"] / base - 1), fz["dev_successor_rate"], s=70, facecolors="none", edgecolors=INK, linewidths=1.2, zorder=4)
        ax.scatter(100 * (h["hand"]["wikitext_loss"] / h["unedited"]["wikitext_loss"] - 1), h["hand"]["dev_successor_rate"], marker="D", s=34, color=ORANGE, zorder=5)
        ax.set_xlim(-1, 30); ax.set_ylim(0, 0.8); ax.set_title(name, fontsize=8.5, loc="left"); ax.set_xlabel("WikiText loss increase (%)"); style(ax); ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    axes[0].set_ylabel("development successor rate")
    axes[0].scatter([], [], s=40, color=BLUE, alpha=0.7, linewidths=0, label="Opus programme (darker = later round)"); axes[0].scatter([], [], s=70, facecolors="none", edgecolors=INK, linewidths=1.2, label="frozen")
    axes[0].scatter([], [], marker="D", s=34, color=ORANGE, label="hand-designed"); fig.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(F / "design.pdf"); plt.close(fig)


def fig_learning():
    c = json.loads((R / "compose/learning_cmp.json").read_text()); base = c["native"]["wiki_nll"]
    rows = [("unedited", c["native"], GRAY), ("closed-form edit", c["closed_form"]["3.0"], BLUE), ("gradient-trained Q/K rows", c["gradient_qk"]["native"]["gain3.0"], ORANGE), ("fine-tuning on composed examples", c["finetune"]["step300"], GREEN)]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw=dict(width_ratios=[3.2, 1]))
    w = 0.2
    for i, (n, r, col) in enumerate(rows):
        axes[0].bar([x + (i - 1.5) * w for x in range(len(DOMS))], [r[d]["successor"] for d in DOMS], w * 0.92, color=col, label=n)
        axes[1].bar(i, 100 * (r["wiki_nll"] / base - 1), 0.62, color=col); axes[1].text(i, 100 * (r["wiki_nll"] / base - 1) + 0.4, f"{100 * (r['wiki_nll'] / base - 1):+.1f}", ha="center", fontsize=7, color=INK)
    axes[0].set_xticks(range(len(DOMS))); axes[0].set_xticklabels(LAB, fontsize=8); axes[0].set_ylim(0, 1.08); style(axes[0], "successor rate")
    axes[0].legend(frameon=False, fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.3))
    axes[1].set_xticks([]); style(axes[1], "WikiText loss increase (%)"); axes[1].set_ylim(0, 20)
    fig.tight_layout(); fig.savefig(F / "learning.pdf"); plt.close(fig)


if __name__ == "__main__":
    fig_locate(); fig_design(); fig_learning()

    fig_main(); fig_controls(); fig_gain(); print("figures written:", sorted(p.name for p in F.glob("*.pdf")))
