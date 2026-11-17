"""Replay released evaluations and audit requests, budget, and credential exclusion."""
import csv
import hashlib
import json
import os
from pathlib import Path

from .wsp_evaluate import evaluate

ROOT=Path(__file__).resolve().parents[1]


def verify():
    replayed=0;cells=0;files=0
    for cohort in ("pilot","reading"):
        fixtures=ROOT/f"results/wsp-{cohort}-fixtures"
        calls=ROOT/f"results/wsp-{cohort}-calls"
        out=ROOT/f"tmp/wsp-verification/{cohort}"
        evaluate(fixtures,calls,out)
        original=list(csv.DictReader((ROOT/f"results/wsp-{cohort}-evaluation/raw.csv").open()))
        rerun=list(csv.DictReader((out/"raw.csv").open()))
        assert original==rerun,"Archived results differ from deterministic replay"
        cells+=sum(len(r) for r in rerun)
        for path in (calls/"responses").glob("*.request.json"):
            body=json.loads(path.read_text())
            assert set(body)=={"model","system","messages","max_tokens","thinking","output_config"}
            assert len(body["messages"])==1 and body["messages"][0]["role"]=="user"
            assert body["model"]=="claude-opus-5-5" and body["max_tokens"]==4096
            assert body["thinking"]=={"type":"adaptive"} and body["output_config"]["effort"]=="medium"
            response=json.loads(path.with_name(path.name.replace(".request.",".response.")).read_text())
            assert response["stop_reason"]=="end_turn"
            assert all(block["type"]=="text" for block in response["content"])
            replayed+=1
    ledger=json.loads((ROOT/"results/wsp-budget.json").read_text())
    total=sum(e["charged_or_reserved_usd"] for e in ledger["entries"])
    assert len(ledger["entries"])==replayed+1
    assert all(e["status"]=="completed" for e in ledger["entries"])
    assert total<=ledger["cap_usd"]==3
    key=os.environ.get("ANTHROPIC_API_KEY","").encode()
    for folder in (ROOT/"results").glob("wsp-*"):
        paths=list(folder.rglob("*.json")) if folder.is_dir() else [folder]
        for path in paths:
            data=path.read_bytes()
            if key and key in data:raise AssertionError("Credential detected in release file")
            files+=1
    sources={p.name:hashlib.sha256(p.read_bytes().replace(b"\r\n",b"\n")).hexdigest()
             for p in (ROOT/"experiments").glob("wsp_*.py")}
    audit=dict(replayed_calls=replayed,replayed_cells=cells,replay="exact",
        fixture_and_prompt_hash_checks="passed during replay",request_access="single stateless tool-free prompt",
        api_key_available_for_literal_scan=bool(key),release_json_files_scanned=files,
        credential_literal_matches=0,thinking_blocks_in_release=0,
        total_cost_including_development_usd=total,budget_cap_usd=ledger["cap_usd"],
        canonical_lf_source_sha256=sources)
    (ROOT/"results/wsp-combined/audit.json").write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))


if __name__=="__main__":verify()
