"""Execute paired experiments and preserve raw measurements and provenance."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .core import (EditResult, acceleration, coefficients, features, fit_coefficients,
                   intervention, parameter_vector, reparameterize, rollout, sample,
                   train, weight_edit)


def rmse(x):
    value = float(x.square().mean().sqrt())
    return value if np.isfinite(value) else None


def source_hash():
    h = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "experiments").glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def run(args):
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    output = Path(args.output)
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    (output / "source").mkdir(exist_ok=True)
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, output / "source" / path.name)
    manifest = dict(config=vars(args), started_utc=datetime.now(timezone.utc).isoformat(),
                    python=platform.python_version(), platform=platform.platform(),
                    torch=torch.__version__, numpy=np.__version__, device="cpu",
                    source_sha256=source_hash(), status="running")
    try:
        manifest["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except subprocess.CalledProcessError:
        manifest["git_commit"] = None
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    started = time.perf_counter()
    rows, model_rows = [], []
    handle = (output / "raw.csv").open("w", newline="", encoding="utf-8")
    writer = None
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        for family in args.families:
            t0 = time.perf_counter()
            model, diagnostics = train(seed, family, args.width, args.steps)
            train_seconds = time.perf_counter() - t0
            c = coefficients(seed, family)
            torch.save(dict(state_dict=model.state_dict(), width=args.width,
                            coefficients=c, family=family, seed=seed),
                       output / "checkpoints" / f"{family}_{seed}.pt")
            model_rows.append(dict(seed=seed, family=family, train_seconds=train_seconds,
                                   parameters=len(parameter_vector(model)), **diagnostics))
            test = sample(args.test_points, 70000 + seed, 1.25)
            shell = sample(args.test_points * 3, 71000 + seed, 2.)
            shell = shell[shell.abs().amax(1) > 1.5][:args.test_points]
            initial = sample(24, 72000 + seed, .8)[:, :2]
            tt = torch.arange(args.horizon, dtype=torch.float64)[:, None]
            phases = torch.arange(24, dtype=torch.float64)[None, :] * .4
            actions = .5 * torch.sin(tt * .07 + phases)
            with torch.no_grad():
                base_id, base_ood = model(test), model(shell)
            for budget in args.budgets:
                # A single nested query sequence is shared by all methods and gauges.
                probes = sample(max(args.budgets), 60000 + seed)
                x = probes[:budget]
                with torch.no_grad():
                    y = model(x)
                fitted = fit_coefficients(x, y, family)
                for name in args.interventions:
                    mult = intervention(name)
                    delta_hat, delta_true = fitted * mult, c * mult
                    with torch.no_grad():
                        target = y + features(x, family) @ delta_hat
                        true_id = acceleration(test, c * (1 + mult), family)
                        true_ood = acceleration(shell, c * (1 + mult), family)
                        shift_id = features(test, family) @ delta_true
                        shift_ood = features(shell, family) @ delta_true
                        reference = rollout(lambda z: acceleration(z, c * (1 + mult), family),
                                            initial, actions)
                    original_predictions = {}
                    for gauge in args.gauges:
                        net = reparameterize(model, 80000 + seed, gauge)
                        with torch.no_grad():
                            symmetry_error = float((net(test) - base_id).abs().max())
                        if symmetry_error > 1e-9:
                            raise AssertionError(f"Broken symmetry: {symmetry_error}")
                        for method in args.methods:
                            t0 = time.perf_counter()
                            count = len(parameter_vector(net))
                            if method == "unedited":
                                result = EditResult(net, parameter_count=count)
                            elif method == "query_residual":
                                result = EditResult(lambda z: net(z) + features(z, family) @ delta_hat,
                                                    parameter_count=count+4, changed_parameters=4)
                            elif method == "query_symbolic":
                                result = EditResult(lambda z: features(z, family) @ (fitted*(1+mult)),
                                                    parameter_count=4, changed_parameters=4)
                            elif method == "oracle_residual":
                                result = EditResult(lambda z: net(z) + features(z, family) @ delta_true,
                                                    parameter_count=count+4, changed_parameters=4)
                            else:
                                result = weight_edit(net, x, target, method,
                                                     ridge=args.ridge, iterations=args.edit_iterations)
                            edit_seconds = time.perf_counter() - t0
                            with torch.no_grad():
                                pred_id, pred_ood = result.predict(test), result.predict(shell)
                                # This set has zero intended effect, not unchanged downstream states.
                                zero = test.clone()
                                if name in ("restoring", "combined"):
                                    zero[:, 0] = 0
                                if name in ("damping", "combined"):
                                    zero[:, 1] = 0
                                collateral = result.predict(zero) - net(zero)
                                trajectory = rollout(result.predict, initial, actions)
                                if gauge == "original":
                                    original_predictions[method] = pred_id.clone()
                                disagreement = rmse(pred_id - original_predictions[method])
                            row = dict(seed=seed, family=family, budget=budget,
                                       intervention=name, gauge=gauge, method=method,
                                       id_rmse=rmse(pred_id-true_id), ood_rmse=rmse(pred_ood-true_ood),
                                       edit_nrmse=rmse(pred_id-base_id-shift_id)/max(rmse(shift_id),1e-12),
                                       edit_ood_nrmse=rmse(pred_ood-base_ood-shift_ood)/max(rmse(shift_ood),1e-12),
                                       collateral_rmse=rmse(collateral),
                                       rollout_rmse=rmse(trajectory-reference),
                                       rollout_nonfinite=int(not torch.isfinite(trajectory).all()),
                                       gauge_disagreement=disagreement, symmetry_max_error=symmetry_error,
                                       coefficient_rmse=rmse(fitted-c),
                                       edit_seconds=edit_seconds,
                                       calibration_forward_points=budget,
                                       jacobian_output_points=result.jacobian_evaluations,
                                       optimizer_forward_points=result.optimizer_forward_points,
                                       deployment_parameters=result.parameter_count,
                                       changed_parameters=result.changed_parameters)
                            if writer is None:
                                writer = csv.DictWriter(handle, fieldnames=list(row))
                                writer.writeheader()
                            writer.writerow(row)
                            rows.append(row)
                    handle.flush()
            print(json.dumps(dict(seed=seed, family=family, base_rmse=diagnostics["base_rmse"],
                                  train_seconds=round(train_seconds, 2), rows=len(rows),
                                  elapsed_seconds=round(time.perf_counter()-started, 1))), flush=True)
    handle.close()
    (output / "models.json").write_text(json.dumps(model_rows, indent=2))
    manifest.update(status="complete", rows=len(rows), elapsed_seconds=time.perf_counter()-started,
                    completed_utc=datetime.now(timezone.utc).isoformat())
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(status="complete", rows=len(rows), seconds=manifest["elapsed_seconds"])), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--steps", type=int, default=1800)
    p.add_argument("--width", type=int, default=24)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--budgets", type=int, nargs="+", default=[8,32,128])
    p.add_argument("--families", nargs="+", default=["duffing", "pendulum", "duffing_mismatch"])
    p.add_argument("--interventions", nargs="+", default=["damping", "restoring", "combined"])
    p.add_argument("--gauges", nargs="+", default=["original", "permuted", "rescaled"])
    p.add_argument("--methods", nargs="+", default=["unedited", "query_residual", "query_symbolic",
                   "head_ridge", "jacobian_ridge", "normalized_jacobian", "finetune_adam", "oracle_residual"])
    p.add_argument("--test-points", type=int, default=1024)
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--edit-iterations", type=int, default=3)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
