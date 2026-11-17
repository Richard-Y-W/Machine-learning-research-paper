"""Verify an independent noisy-adaptation rerun and both archived sources."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from .analyze_cycle4 import read


def verify(root):
    main=root/"cycle4-heldout";repeat=root/"cycle4-reproduction-final"
    for folder in (main,repeat):
        manifest=json.loads((folder/"manifest.json").read_text())
        if manifest["status"]!="complete":
            raise AssertionError("Incomplete run")
        h=hashlib.sha256()
        for path in sorted((folder/"source").glob("*.py")):
            h.update(path.name.encode());h.update(path.read_bytes())
        if h.hexdigest()!=manifest["source_sha256"]:
            raise AssertionError("Source snapshot mismatch")
    def key(r):
        return tuple(r[k] for k in ("seed","family","shift","budget","method"))
    originals={key(r):r for r in read(main/"raw.csv")}
    repeated=read(repeat/"raw.csv")
    metrics=["id_rmse","ood_rmse","rollout_rmse","calibration_noisy_rmse","source_rmse",
        "extra_scalars","selected_ridge","selected_length","length_q","length_v","length_u",
        "selected_iterations","cv_mse","gradient_points"]
    maximum=0.
    for row in repeated:
        original=originals[key(row)]
        for metric in metrics:
            a,b=row[metric],original[metric]
            if not np.isfinite([a,b]).all() or not np.isclose(a,b,atol=1e-10,rtol=1e-8):
                raise AssertionError((key(row),metric,a,b))
            maximum=max(maximum,abs(a-b))
    report=dict(status="passed",source_snapshots=2,rows=len(repeated),numeric_cells=len(repeated)*len(metrics),
                max_absolute_difference=maximum,timings_excluded=True)
    (main/"reproduction-audit.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("results"))
    verify(p.parse_args().root)
