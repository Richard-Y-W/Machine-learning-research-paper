"""LaTeX tables for the preprint, generated from results JSON (no hand-copied numbers). Writes paper/tables/*.tex."""
import json
from pathlib import Path

R = Path("results"); T = Path("paper/tables"); T.mkdir(parents=True, exist_ok=True)
DOMS = ["letters", "numwords", "days", "months", "ordinals"]; NAME = dict(letters="Letters", numwords="Number words", days="Weekdays", months="Months", ordinals="Ordinals")
J = lambda p: json.loads(Path(p).read_text())
f2 = lambda x: f"{x:.2f}".replace("0.", ".", 1) if x < 1 else f"{x:.2f}"


def write(name, lines):
    (T / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main_table():
    a, b = J(R / "compose/lookup3_confirm.json"), J(R / "compose_1p7b/lookup3_confirm.json")
    fm = [("arrow_new_keys", r"$k \rightarrow v$"), ("colon_new_keys", r"$k$: $v$"), ("equals_new_keys", r"$k = v$")]
    L = [r"\begin{tabular}{llcccccc}", r"\toprule", r" & & \multicolumn{3}{c}{Qwen3-0.6B-Base} & \multicolumn{3}{c}{Qwen3-1.7B-Base} \\", r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
         r"Format & Values & Unedited & Edited & W / L & Unedited & Edited & W / L \\", r"\midrule"]
    for key, lab in fm:
        for i, d in enumerate(DOMS):
            x, y = a[key][d], b[key][d]
            L.append(f"{lab if i == 0 else ''} & {NAME[d]} & {f2(x['native_successor'])} & \\textbf{{{f2(x['edited_successor'])}}} & {x['wins']} / {x['losses']} & "
                     f"{f2(y['native_successor'])} & \\textbf{{{f2(y['edited_successor'])}}} & {y['wins']} / {y['losses']} \\\\")
        L.append(r"\midrule" if key != fm[-1][0] else r"\bottomrule")
    write("main", L + [r"\end{tabular}"])
    L = [r"\begin{tabular}{lcccccc}", r"\toprule", r" & \multicolumn{3}{c}{Qwen3-0.6B-Base} & \multicolumn{3}{c}{Qwen3-1.7B-Base} \\", r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"Values & Head removed & Donor's succ. & Own succ. & Head removed & Donor's succ. & Own succ. \\", r"\midrule"]
    for d in DOMS:
        x, y = a["arrow_new_keys"][d], b["arrow_new_keys"][d]
        L.append(f"{NAME[d]} & {f2(x['edited_A_removed_successor'])} & {f2(x['swap_donor_successor'])} & {f2(x['swap_own_successor'])} & "
                 f"{f2(y['edited_A_removed_successor'])} & {f2(y['swap_donor_successor'])} & {f2(y['swap_own_successor'])} \\\\")
    write("causal", L + [r"\bottomrule", r"\end{tabular}"])


def controls_table():
    c, l3 = J(R / "compose/lookup3_controls.json"), J(R / "compose/lookup3.json")
    hosts = [v for k, v in c["host"].items() if k != "h6"]
    rows = [("Unedited model", c["native"]), ("Edit (successor head, gain 3)", c["edit"]),
            ("Same attention, six other heads (mean)", {d: dict(successor=sum(h[d]["successor"] for h in hosts) / len(hosts)) for d in DOMS}),
            ("Same attention, six other heads (max)", {d: dict(successor=max(h[d]["successor"] for h in hosts)) for d in DOMS}),
            ("Attention fitted to a wrong position", l3["control_wrong_target_gain3.0"]),
            ("Random attention, matched spectrum (mean of 3)", {d: dict(successor=sum(s[d]["successor"] for s in c["structural_null"]) / 3) for d in DOMS}),
            ("Gain only, native attention", l3["control_gain_only_gain3.0"]), ("Edit with another head's output projection", l3["control_wrong_output_projection"]),
            ("Edit, neighbour head (shared keys) removed", c["neighbour_removed_edit"])]
    L = [r"\begin{tabular}{lccccc}", r"\toprule", "Condition & " + " & ".join(NAME[d] for d in DOMS) + r" \\", r"\midrule"]
    for n, r in rows:
        L.append(n + " & " + " & ".join(f2(r[d]["successor"]) for d in DOMS) + r" \\")
        if n.startswith("Edit (succ"):
            L.append(r"\midrule")
    write("controls", L + [r"\bottomrule", r"\end{tabular}"])
    cf = c["counterfactual"]; names = [("adjacent_allowed", "Adjacent values allowed"), ("successor_present", "Correct successor present as a distractor"), ("long_table", "Eight-row tables")]
    L = [r"\begin{tabular}{lccccc}", r"\toprule", "Prompt variant & " + " & ".join(NAME[d] for d in DOMS) + r" \\", r"\midrule"]
    for k, n in names:
        L.append(n + " & " + " & ".join((f2(cf[k]["native"][d]["successor"]) + r"$\,\to\,$" + f2(cf[k]["edit"][d]["successor"])) if d in cf[k]["edit"] else "--" for d in DOMS) + r" \\")
    sd = c["swap_distinct"]
    L += [r"\midrule", "Swap, donor value differs: donor's successor & " + " & ".join(f2(sd[d]["donor_successor"]) for d in DOMS) + r" \\",
          "Swap, donor value differs: own successor & " + " & ".join(f2(sd[d]["own_successor"]) for d in DOMS) + r" \\"]
    write("counterfactual", L + [r"\bottomrule", r"\end{tabular}"])


def increment_table():
    L = [r"\begin{tabular}{lcccc}", r"\toprule", r" & \multicolumn{2}{c}{Qwen3-0.6B-Base} & \multicolumn{2}{c}{Qwen3-1.7B-Base} \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"Condition & $v{+}1$ & $v$ & $v{+}1$ & $v$ \\", r"\midrule"]
    a = J(R / "compose/increment.json"); bp = R / "compose_1p7b/increment.json"; b = J(bp) if bp.exists() else None
    cell = lambda r, k: (f2(r[k]["increment"]) + " & " + f2(r[k]["copy"])) if r and k in r else "-- & --"
    ed = lambda r, g: (f2(r["edit"][g]["increment"]) + " & " + f2(r["edit"][g]["copy"])) if r else "-- & --"
    L.append("Unedited model & " + f2(a["native"]["digits"]["increment"]) + " & " + f2(a["native"]["digits"]["copy"]) + " & " + ((f2(b["native"]["digits"]["increment"]) + " & " + f2(b["native"]["digits"]["copy"])) if b else "-- & --") + r" \\")
    for g in ("1.0", "2.0", "3.0", "4.0"):
        L.append(f"Edit, gain {g[0]} & " + ed(a, g) + " & " + ed(b, g) + r" \\")
    L.append(r"\midrule")
    for k, n in (("edit_A_removed", "Edit (gain 3), successor head removed"), ("control_wrong_target", "Attention fitted to a wrong position"), ("control_gain_only", "Gain only, native attention"),
                 ("control_wrong_output_projection", "Edit with another head's output projection")):
        L.append(n + " & " + cell(a, k) + " & " + cell(b, k) + r" \\")
    write("increment", L + [r"\bottomrule", r"\end{tabular}"])


def learning_table():
    c = J(R / "compose/learning_cmp.json"); base = c["native"]["wiki_nll"]
    rows = [("Unedited", "--", c["native"]), ("Closed-form, gain 3", "Routing labels", c["closed_form"]["3.0"]),
            ("Gradient Q/K, gain 3", "Routing labels", c["gradient_qk"]["native"]["gain3.0"]),
            ("Fine-tuning", "Composed examples", c["finetune"]["step300"])]
    L = [r"\begin{tabular}{llccccccc}", r"\toprule", r"Method & Supervision & " + " & ".join(NAME[d].replace("Number words", "Num.").replace("Weekdays", "Days").replace("Ordinals", "Ord.").replace("Letters", "Let.").replace("Months", "Mon.") for d in DOMS) + r" & Word lookup & WikiText \\", r"\midrule"]
    for n, s, r in rows:
        L.append(f"{n} & {s} & " + " & ".join(f2(r[d]["successor"]) for d in DOMS) + f" & {f2(r['word_lookup'])} & {100 * (r['wiki_nll'] / base - 1):+.1f}\\% \\\\")
    write("learning", L + [r"\bottomrule", r"\end{tabular}"])


if __name__ == "__main__":
    main_table(); controls_table(); increment_table(); learning_table(); print("tables:", sorted(p.name for p in T.glob("*.tex")))


def opus_tables():
    runs = [("0.6B", "Hand-designed", None, R / "compose/hand_ref.json", R / "compose/lookup3_confirm.json"),
            ("0.6B", "Opus, run 1", R / "compose/opus_program/log.json", None, None), ("0.6B", "Opus, run 2", R / "compose_run2/opus_program/log.json", None, None),
            ("1.7B", "Hand-designed", None, R / "compose_1p7b/hand_ref.json", R / "compose_1p7b/lookup3_confirm.json"), ("1.7B", "Opus", R / "compose_1p7b/opus_program/log.json", None, None)]
    L = [r"\begin{tabular}{llccccccccc}", r"\toprule", r"Model & Design & Let. & Num. & Days & Mon. & Ord. & Seq. & Word lookup & WikiText & API cost \\", r"\midrule"]
    prev = None
    for model, name, logp, handp, confp in runs:
        if prev and prev != model:
            L.append(r"\midrule")
        prev = model
        if logp:
            g = J(logp); succ = [g["confirm"]["arrow_new_keys"][d]["successor"] for d in DOMS]; seq = g["frozen"]["measured"]["sequence_continuation"]
            wl = g["confirm"]["word_lookup_accuracy"]; wk = 100 * (g["confirm"]["wikitext_loss"] / g["base"]["wikitext_loss"] - 1); cost = f"\\${g['spent_usd']:.2f}"
        else:
            h = J(handp); succ = [h["confirm_hand"][d] for d in DOMS]; seq = h["hand"]["sequence_continuation"]
            wl = h["hand"]["word_lookup_accuracy"]; wk = 100 * (h["hand"]["wikitext_loss"] / h["unedited"]["wikitext_loss"] - 1); cost = "--"
        L.append(f"{model} & {name} & " + " & ".join(f2(x) for x in succ) + f" & {f2(seq)} & {f2(wl)} & {wk:+.1f}\\% & {cost} \\\\")
    write("opus", L + [r"\bottomrule", r"\end{tabular}"])
    c = J(R / "compose/opus_program/controls.json")["results"]
    L = [r"\begin{tabular}{lccccccc}", r"\toprule", r"Variant & Let. & Num. & Days & Mon. & Ord. & Seq. & WikiText \\", r"\midrule"]; base = c["unedited"]["wikitext_loss"]
    for k, m in c.items():
        L.append(k[0].upper() + k[1:] + " & " + " & ".join(f2(m["successor"][d]) for d in DOMS) + f" & {f2(m['sequence_continuation'])} & {100 * (m['wikitext_loss'] / base - 1):+.1f}\\% \\\\")
        if k in ("frozen programme", "frozen, all programmed heads removed"):
            L.append(r"\midrule")
    write("opus_controls", L + [r"\bottomrule", r"\end{tabular}"])


if __name__ == "__main__":
    opus_tables(); print("opus tables written")


def programmes_appendix():
    out = []
    for name, p in (("Qwen3-0.6B-Base, run 1", R / "compose/opus_program/log.json"), ("Qwen3-0.6B-Base, run 2", R / "compose_run2/opus_program/log.json"), ("Qwen3-1.7B-Base", R / "compose_1p7b/opus_program/log.json")):
        g = J(p); fz = g["frozen"]; n = sum(len(r["candidates"]) for r in g["rounds"])
        out += [f"\\paragraph{{{name}.}} Frozen in round {fz['round']} of {len(g['rounds'])} ({n} programmes evaluated). Development successor rate {fz['measured']['dev_successor_rate']:.2f} "
                f"(unedited {g['base']['dev_successor_rate']:.2f}).", r"\begin{Verbatim}[fontsize=\scriptsize,breaklines=true,breakanywhere=true]"]
        out += [json.dumps(o) for o in fz["ops"]] + [r"\end{Verbatim}"]
    write("programmes", out)


if __name__ == "__main__":
    programmes_appendix(); print("programmes appendix written")
