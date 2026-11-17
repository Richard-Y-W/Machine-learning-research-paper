"""Verify archived sources and independently re-executed follow-up subsets."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .analyze_deployment import read_rows


def verify_source(folder):
    manifest=json.loads((folder/"manifest.json").read_text())
    if manifest["status"]!="complete":
        raise AssertionError(f"Incomplete run: {folder}")
    h=hashlib.sha256()
    for path in sorted((folder/"source").glob("*.py")):
        h.update(path.name.encode());h.update(path.read_bytes())
    if h.hexdigest()!=manifest["source_sha256"]:
        raise AssertionError(f"Source hash mismatch: {folder}")


def compare(main,repeat,keys,metrics):
    def key(r):
        return tuple(r[k] for k in keys)
    source_rows=read_rows(main/"raw.csv")
    by_key={key(r):r for r in source_rows}
    repeat_rows=read_rows(repeat/"raw.csv")
    max_error=0.
    for row in repeat_rows:
        original=by_key[key(row)]
        for metric in metrics:
            a,b=row[metric],original[metric]
            if not np.isfinite([a,b]).all() or not np.isclose(a,b,atol=1e-10,rtol=1e-8):
                raise AssertionError((key(row),metric,a,b))
            max_error=max(max_error,abs(a-b))
    return dict(rows=len(repeat_rows),numeric_cells=len(repeat_rows)*len(metrics),
                max_absolute_difference=max_error)


def verify(root):
    folders={name:root/name for name in ["deployment-heldout","deployment-reproduction",
                                       "dense-heldout","dense-reproduction"]}
    for folder in folders.values():
        verify_source(folder)
    result=dict(status="passed",source_snapshots=4,
        deployment=compare(folders["deployment-heldout"],folders["deployment-reproduction"],
            ["seed","family","budget","teacher","schedule","gauge","method"],
            ["id_rmse","ood_rmse","teacher_fit_rmse","teacher_ood_fit_rmse","calibration_fit_rmse",
             "teacher_id_rmse","teacher_ood_rmse","rollout_rmse","gauge_disagreement","coefficient_rmse"]),
        dense=compare(folders["dense-heldout"],folders["dense-reproduction"],
            ["seed","family","method","steps"],
            ["id_rmse","ood_rmse","teacher_fit_rmse","teacher_id_rmse","rollout_rmse"]))
    (folders["deployment-heldout"]/"reproduction-audit.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("results"))
    verify(p.parse_args().root)
