"""Software invariants for the evolution harness. None requires a hypothesis to hold."""
import json

import pytest
import torch

from experiments.evolution import editlang, editor


def _params():
    g = torch.Generator().manual_seed(0)
    p = {
        "model.layers.0.mlp.down_proj.weight": torch.randn(16, 48, generator=g),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(16, 1024, generator=g),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(256, 16, generator=g),
        "model.layers.0.input_layernorm.weight": torch.ones(16),
        "model.norm.weight": torch.ones(16),
    }
    return p, {k: v.clone() for k, v in p.items()}


def _bank(p):
    return editlang.RefBank(p, vectors={"err:L0": torch.randn(16), "mlperr:L0": torch.randn(48)})


def test_genome_replay_reproduces_tensors():
    p, orig = _params(); bank = _bank(p)
    m1 = editlang.resolve([{"op": "rank1", "tensor": "L0.mlp.down", "u": "err:L0", "v": "mlperr:L0", "alpha": -0.1}], p, bank)
    m2 = editlang.resolve([{"op": "ablate", "tensor": "L0.attn.o", "head": 1},
                           {"op": "set", "tensor": "final_norm", "entries": [[3, 0.5]]}], p, bank)
    for m in (m1, m2):
        with editlang.Applied(p, m, orig) as ap:
            ap.keep = True
    q = {k: v.clone() for k, v in orig.items()}
    editlang.replay(q, [m1, m2], orig)
    for k in p:
        assert torch.equal(p[k], q[k])


def test_failed_or_temporary_mutation_leaves_parent_unchanged():
    p, orig = _params(); before = {k: v.clone() for k, v in p.items()}
    big = editlang.resolve([{"op": "scale", "tensor": "L0.mlp.down", "axis": "all", "factor": 2.0}], p, _bank(p))
    with pytest.raises(editlang.InvalidEdit):
        with editlang.Applied(p, big, orig):
            pass
    ok = editlang.resolve([{"op": "scale", "tensor": "L0.mlp.down", "axis": "col", "index": [1, 2], "factor": 0.9}], p, _bank(p))
    with editlang.Applied(p, ok, orig):
        assert not torch.equal(p["model.layers.0.mlp.down_proj.weight"], before["model.layers.0.mlp.down_proj.weight"])
    for k in p:
        assert torch.equal(p[k], before[k])


@pytest.mark.parametrize("ops", [
    [],
    [{"op": "scale", "tensor": "L0.attn.q", "head": 99, "factor": 1.0}],
    [{"op": "rank1", "tensor": "L0.mlp.down", "u": "mlperr:L0", "v": "err:L0", "alpha": 0.1}],  # wrong dims
    [{"op": "rank1", "tensor": "L0.mlp.down", "u": "err:L0", "v": "mlperr:L0", "alpha": 0.9}],  # alpha too big
    [{"op": "scale", "tensor": "L0.mlp.down", "axis": "row", "index": [999], "factor": 0.5}],
    [{"op": "nope", "tensor": "L0.mlp.down"}],
    [{"op": "scale", "tensor": "embed", "axis": "all", "factor": 0.5}],
])
def test_invalid_proposals_rejected(ops):
    p, _ = _params()
    with pytest.raises(editlang.InvalidEdit):
        editlang.resolve(ops, p, _bank(p))


def test_revert_restores_original():
    p, orig = _params()
    m = editlang.resolve([{"op": "scale", "tensor": "L0.ln1", "index": [0], "factor": 1.2}], p, _bank(p))
    with editlang.Applied(p, m, orig) as ap:
        ap.keep = True
    r = editlang.resolve([{"op": "revert", "tensor": "L0.ln1"}], p, _bank(p))
    with editlang.Applied(p, r, orig) as ap:
        ap.keep = True
    assert torch.equal(p["model.layers.0.input_layernorm.weight"], orig["model.layers.0.input_layernorm.weight"])


def test_ledger_refuses_over_cap(tmp_path):
    led = tmp_path / "ledger.json"
    rid = editor.reserve("a", 10.0, led)
    editor.settle(rid, {"input_tokens": 1000, "output_tokens": 1000}, path=led)
    assert editor.committed(led) == pytest.approx(editor.COST_SAFETY * (1000 * 4e-6 + 1000 * 20e-6))
    editor.reserve("b", editor.STOP_AT - 0.5, led)
    with pytest.raises(editor.BudgetExceeded):
        editor.reserve("c", 0.5, led)


def test_crashed_call_keeps_reservation(tmp_path):
    led = tmp_path / "ledger.json"
    rid = editor.reserve("a", 0.3, led)
    editor.settle(rid, None, status="error_reservation_kept", path=led)
    assert editor.committed(led) == pytest.approx(0.3)


def test_dossier_never_contains_test_items():
    """The editor sees only dev data: test prompts must not appear in the smoke dossier."""
    from pathlib import Path
    dossier = Path("results/evolution/smoke_dossier.txt")
    splits = Path("results/evolution/splits.json")
    if not dossier.exists() or not splits.exists():
        pytest.skip("run `python -m experiments evo smoke` first")
    text = dossier.read_text(encoding="utf-8")
    S = json.loads(splits.read_text())
    for it in S["test_target"]["items"] + S["test_protected"]["items"]:
        q = it["prompt"].split("\n", 2)[-1][:120].replace("\n", " | ")
        assert q not in text


# ---------------------------------------------------------------- v2

def test_v2_draw_deterministic_and_sized():
    from experiments.evolution import v2
    pool = {t: [dict(uid=f"{t}:{i}") for i in range(500)] for t in ("a", "b")}
    s1 = v2.draw(pool, ["a", "b"], 40, "G-LLM_1|g3")
    assert s1 == v2.draw(pool, ["a", "b"], 40, "G-LLM_1|g3")
    assert s1 != v2.draw(pool, ["a", "b"], 40, "G-LLM_1|g4")
    assert sum(x["uid"].startswith("a:") for x in s1) == 40 and len({x["uid"] for x in s1}) == 80


def test_v2_pool_disjoint_from_v1_splits():
    from pathlib import Path
    P, S = Path("results/evolution/v2/pool.json"), Path("results/evolution/splits.json")
    if not P.exists() or not S.exists():
        pytest.skip("build the v2 pool first")
    used = {it["uid"] for k, v in json.loads(S.read_text()).items() if not k.startswith("wiki") for it in v["items"]}
    for t, v in json.loads(P.read_text()).items():
        assert not used & {it["uid"] for it in v["items"]}, t


def test_v2_unit_tasks():
    from experiments.evolution import v2
    assert v2.unit_tasks("S-LLM", "2_commonsense_qa") == (["commonsense_qa"], v2.SAMPLE_SPECIALIST)
    tasks, n = v2.unit_tasks("G-LLM", "1")
    assert len(tasks) == 4 and n == v2.SAMPLE_GENERALIST


def test_skill_pilot_splits():
    from experiments.skill import pilot
    assert pilot.HELD == ["a", "d", "h", "o", "q", "t", "u", "v"]
    assert len(pilot.PRACTICE) == 17 and not set(pilot.PRACTICE) & set(pilot.HELD)
    assert pilot.prompts(["c"], ["ENC: {x} ->"]) == [("ENC: c ->", "d")]
