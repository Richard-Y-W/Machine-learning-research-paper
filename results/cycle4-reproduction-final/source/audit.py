"""Check provenance and compare a fresh rerun against the original raw rows."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
import matplotlib
import pytest
import torch

from .analyze import read_rows


def key(row):
    return tuple(row[k] for k in ("seed","family","budget","intervention","gauge","method"))


def audit(main, reproduction):
    manifest=json.loads((main/"manifest.json").read_text())
    h=hashlib.sha256()
    for path in sorted((main/"source").glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    if h.hexdigest()!=manifest["source_sha256"]:
        raise AssertionError("Source snapshot hash mismatch")
    original=read_rows(main/"raw.csv")
    by_key={key(r):r for r in original}
    repeat=read_rows(reproduction/"raw.csv")
    fields=["id_rmse","ood_rmse","edit_nrmse","edit_ood_nrmse","collateral_rmse",
            "rollout_rmse","gauge_disagreement","coefficient_rmse"]
    max_difference=0.
    for row in repeat:
        if key(row) not in by_key:
            raise AssertionError("Rerun does not match a main condition")
        reference=by_key[key(row)]
        for metric in fields:
            delta=abs(row[metric]-reference[metric])
            max_difference=max(max_difference,delta)
            if not np.isclose(row[metric],reference[metric],atol=1e-10,rtol=1e-8):
                raise AssertionError((key(row),metric,row[metric],reference[metric]))
    oracle=[r for r in original if r["method"]=="oracle_residual"]
    if max(r["edit_nrmse"] for r in oracle)>1e-10:
        raise AssertionError("Oracle residual fails the exact-change check")
    expected_models=manifest["config"]["seeds"]*len(manifest["config"]["families"])
    if len(list((main/"checkpoints").glob("*.pt")))!=expected_models:
        raise AssertionError("Missing base model checkpoints")
    summary=dict(status="passed",source_snapshot_sha256=h.hexdigest(),
                 reproduced_rows=len(repeat),reproduced_numeric_cells=len(repeat)*len(fields),
                 max_absolute_reproduction_difference=max_difference,
                 oracle_change_check="passed",base_checkpoints=expected_models,
                 package_versions=dict(numpy=np.__version__,scipy=scipy.__version__,
                     matplotlib=matplotlib.__version__,pytest=pytest.__version__,torch=torch.__version__),
                 scope="One independently re-executed seed/family/budget/intervention subset; "
                       "all methods and parameterizations. Timings excluded.")
    (main/"audit.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,default=Path("results/main"))
    p.add_argument("--reproduction",type=Path,default=Path("results/reproduction"))
    args=p.parse_args()
    audit(args.input,args.reproduction)
