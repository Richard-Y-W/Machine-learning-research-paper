"""Compare independently re-executed optimization and identification conditions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .analyze_cycle3 import read


def verify(root):
    main=root/"cycle3-heldout";repeat=root/"cycle3-reproduction"
    for folder in (main,repeat):
        manifest=json.loads((folder/"manifest.json").read_text())
        if manifest["status"]!="complete":
            raise AssertionError("Incomplete run")
        h=hashlib.sha256()
        for path in sorted((folder/"source").glob("*.py")):
            h.update(path.name.encode());h.update(path.read_bytes())
        if h.hexdigest()!=manifest["source_sha256"]:
            raise AssertionError("Source hash mismatch")
    specifications={
        "fitting.csv":(["seed","family","method","event","iteration_budget"],
            ["id_rmse","teacher_fit_rmse","ood_rmse","calibration_fit_rmse","iteration_budget",
             "accepted_iterations","closure_calls","threshold_attained","parameters"]),
        "identification.csv":(["seed","family","budget","design","access"],
            ["budget","correct_family","id_rmse","ood_rmse","condition_number","source_query_rmse",
             "duffing_cv_mse","pendulum_cv_mse"]),
        "trace.csv":(["seed","family","method","iteration_budget"],
            ["accepted_iterations","closure_calls","calibration_fit_rmse"])}
    result=dict(status="passed",source_snapshots=2,files={})
    for filename,(keys,metrics) in specifications.items():
        def key(r):
            return tuple(r[k] for k in keys)
        source_rows={key(r):r for r in read(main/filename)}
        repeat_rows=read(repeat/filename)
        maximum=0.
        for row in repeat_rows:
            reference=source_rows[key(row)]
            if filename=="identification.csv" and row["selected_family"]!=reference["selected_family"]:
                raise AssertionError("Different selected family")
            for metric in metrics:
                a,b=row[metric],reference[metric]
                if not np.isfinite([a,b]).all() or not np.isclose(a,b,atol=1e-10,rtol=1e-8):
                    raise AssertionError((filename,key(row),metric,a,b))
                maximum=max(maximum,abs(a-b))
        result["files"][filename]=dict(rows=len(repeat_rows),numeric_cells=len(repeat_rows)*len(metrics),
                                       max_absolute_difference=maximum)
    result["numeric_cells"]=sum(v["numeric_cells"] for v in result["files"].values())
    (main/"reproduction-audit.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("results"))
    verify(p.parse_args().root)
