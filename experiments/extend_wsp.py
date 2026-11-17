"""Populate the WSP manuscript only from measured, archived outcomes."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def extend(target):
    from .wsp_report import report
    s=report();sources=s["coefficient_sources"]
    t=s["tasks"];mae=s["coefficient_mae"]
    table=["| Seed | Physical c | LLM estimate | Local slope | Query fit |",
           "|---|---:|---:|---:|---:|"]
    for r in sources:
        table.append(f"| {r['seed']} | {r['true_c']:.5f} | {r['llm_c']:.5f} | {r['local_slope_c']:.5f} | {r['query_c']:.5f} |")
    replacements=dict(OWN_MAX=f"{s['max_own_compilation_nrmse']:.3g}",
        EVAL_COST=f"{s['evaluated_cost_usd']:.4f}",TOTAL_COST=f"{s['total_cost_including_development_usd']:.4f}",
        ACTION_NRMSE=f"{t['action_gain']['mean_edit_nrmse']:.3g}",
        DRAG_NRMSE=f"{t['add_drag']['mean_edit_nrmse']:.3g}",
        SEMANTIC_NRMSE=f"{t['semantic_damping']['mean_edit_nrmse']:.4f}",
        DRAG_OOD=f"{t['add_drag']['mean_ood_nrmse']:.5f}",
        COEFFICIENT_TABLE="\n".join(table),LLM_MAE=f"{mae['llm']:.5f}",
        QUERY_MAE=f"{mae['query']:.5f}",PRIOR_MAE=f"{mae['prior']:.5f}",LOCAL_MAE=f"{mae['local']:.5f}",
        LOCAL_AGREEMENT=f"{s['llm_local_mae']:.5f}",
        SOURCE_MIN=f"{min(r['source_rmse'] for r in sources):.5f}",
        SOURCE_MAX=f"{max(r['source_rmse'] for r in sources):.5f}")
    text=(ROOT/"paper/wsp.template.md").read_text(encoding="utf-8")
    for key,value in replacements.items():text=text.replace("{{"+key+"}}",value)
    if "{{" in text:raise AssertionError("Unresolved WSP manuscript field")
    target.write_text(text,encoding="utf-8")
    return target
