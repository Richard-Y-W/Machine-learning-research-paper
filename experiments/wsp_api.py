"""Independent tool-free model calls with a shared conservative cost ledger.

This module does not execute networks or read private fixture files.
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime,timezone
from pathlib import Path

from .wsp_common import response_schema

MODEL="claude-opus-5-5"
INPUT_RATE=4/1_000_000
OUTPUT_RATE=20/1_000_000
MAX_TOKENS=4096
SYSTEM="You are editing a neural network directly from numerical weights. No tools are available. Return the requested final JSON patch with a brief public explanation."


def post(endpoint,body):
    key=os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured")
    request=urllib.request.Request("https://api.anthropic.com/v1/"+endpoint,
        data=json.dumps(body).encode(),headers={"x-api-key":key,"anthropic-version":"2023-06-01",
        "content-type":"application/json"},method="POST")
    try:
        with urllib.request.urlopen(request,timeout=240) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail=error.read().decode(errors="replace").replace(key,"[REDACTED]")[:700]
        raise RuntimeError(f"API HTTP {error.code}: {detail}") from None


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2),encoding="utf-8")
    temporary.replace(path)


def reserve(ledger_path,case,reservation,budget):
    lock=ledger_path.with_suffix(".lock")
    lock.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(cap_usd=budget,entries=[])
        if ledger["cap_usd"]!=budget:
            raise RuntimeError("Shared ledger budget differs from requested budget")
        committed=sum(e["charged_or_reserved_usd"] for e in ledger["entries"])
        if committed+reservation>budget:
            return None
        entry=dict(id=uuid.uuid4().hex,case_id=case,status="reserved",charged_or_reserved_usd=reservation,
                   reserved_usd=reservation,created_utc=datetime.now(timezone.utc).isoformat())
        ledger["entries"].append(entry);write_json(ledger_path,ledger)
        return entry["id"]
    finally:
        os.close(fd);lock.unlink()


def settle(ledger_path,entry_id,usage):
    # Runs sequentially. An unknown/error response retains its full reservation.
    ledger=json.loads(ledger_path.read_text())
    entry=next(e for e in ledger["entries"] if e["id"]==entry_id)
    cost=(usage.get("input_tokens",0)*INPUT_RATE+usage.get("output_tokens",0)*OUTPUT_RATE+
          usage.get("cache_read_input_tokens",0)*.2/1_000_000+
          usage.get("cache_creation_input_tokens",0)*8/1_000_000)
    if cost>entry["reserved_usd"]+1e-9:
        raise RuntimeError("Observed usage exceeds conservative reservation; stop immediately")
    entry.update(status="completed",charged_or_reserved_usd=cost,usage=usage)
    write_json(ledger_path,ledger)
    return cost,sum(e["charged_or_reserved_usd"] for e in ledger["entries"])


def run(args):
    fixtures=Path(args.fixtures);output=Path(args.output);ledger=Path(args.ledger)
    output.mkdir(parents=True,exist_ok=True)
    (output/"responses").mkdir(exist_ok=True)
    if (output/"manifest.json").exists():
        raise FileExistsError(output)
    fixture_manifest=json.loads((fixtures/"manifest.json").read_text())
    cases=fixture_manifest["cases"][:args.limit] if args.limit else fixture_manifest["cases"]
    manifest=dict(model=MODEL,effort="medium",max_output_tokens=MAX_TOKENS,tools_provided=False,
        feedback_provided=False,planned_cases=cases,completed_cases=[],failed_cases=[],
        status="running",started_utc=datetime.now(timezone.utc).isoformat(),budget_cap_usd=args.budget,
        pricing=dict(input_per_million=4,output_per_million=20,
                     source="https://platform.claude.com/docs/en/about-claude/pricing",checked="2026-11-10"))
    write_json(output/"manifest.json",manifest)
    for case in cases:
        public=json.loads((fixtures/"public"/f"{case}.json").read_text())
        body=dict(model=MODEL,system=SYSTEM,messages=[dict(role="user",content=public["prompt"])],
            max_tokens=MAX_TOKENS,thinking=dict(type="adaptive"),
            output_config=dict(effort="medium",format=dict(type="json_schema",schema=response_schema())))
        # Counting is non-generative; add ample room for schema/format overhead.
        counted=post("messages/count_tokens",{k:v for k,v in body.items() if k in ("model","system","messages","thinking")})
        input_bound=max(counted["input_tokens"]+2000,len(json.dumps(body).encode())+2000)
        reservation=input_bound*INPUT_RATE+MAX_TOKENS*OUTPUT_RATE
        entry=reserve(ledger,case,reservation,args.budget)
        if entry is None:
            manifest["status"]="budget_stopped";break
        write_json(output/"responses"/f"{case}.request.json",body)
        t=time.perf_counter()
        try:
            response=post("messages",body)
            public_content=[block for block in response.get("content",[]) if block.get("type")=="text"]
            archived={k:v for k,v in response.items() if k!="content"}
            archived["content"]=public_content
            archived["thinking_blocks_omitted"]=sum(block.get("type") in ("thinking","redacted_thinking") for block in response.get("content",[]))
            archived["latency_seconds"]=time.perf_counter()-t
            cost,total=settle(ledger,entry,response["usage"])
            archived["estimated_list_price_usd"]=cost
            write_json(output/"responses"/f"{case}.response.json",archived)
            manifest["completed_cases"].append(case)
            print(json.dumps(dict(case_id=case,status="response_saved",stop_reason=response.get("stop_reason"),
                output_tokens=response["usage"].get("output_tokens"),cost_usd=round(cost,5),
                total_ledger_usd=round(total,5))),flush=True)
        except Exception as error:
            write_json(output/"responses"/f"{case}.error.json",dict(error=str(error),reservation_usd=reservation))
            manifest["failed_cases"].append(case)
            # No retry. An API failure stops the run to avoid multiplying a bad configuration.
            manifest["status"]="api_error";write_json(output/"manifest.json",manifest);raise
        write_json(output/"manifest.json",manifest)
    if manifest["status"]=="running":manifest["status"]="complete"
    manifest["completed_utc"]=datetime.now(timezone.utc).isoformat()
    write_json(output/"manifest.json",manifest)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixtures",required=True);p.add_argument("--output",required=True)
    p.add_argument("--ledger",default="results/wsp-budget.json");p.add_argument("--budget",type=float,default=3.)
    p.add_argument("--limit",type=int,default=0)
    run(p.parse_args())
