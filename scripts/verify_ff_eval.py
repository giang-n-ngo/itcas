"""Sanity-check spacecraft_formation_flying_a1 evaluation reproducibility.

Pulls 10 (X, Y_stored) rows from the pre-filtered historical pool, then
re-evaluates the *same* X via the Slurm batch simulator (bypassing the
eval cache) and compares the two Y matrices side-by-side.

Run on a Slurm submit host (sbatch/squeue available):

    cd /home/giangn/ITCAS
    python scripts/verify_ff_eval.py --n 10 --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from itcas.pipeline.problems import (  # noqa: E402
    SpacecraftFormationFlyingA1,
    _metrics_to_constraints,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smartsat-root", type=str, default=None)
    p.add_argument("--poll-interval", type=float, default=15.0)
    p.add_argument("--job-timeout", type=float, default=1800.0)
    p.add_argument("--out", type=str, default="results/ff_eval_verify.json")
    args = p.parse_args()

    prob = SpacecraftFormationFlyingA1(
        smartsat_root=args.smartsat_root,
        poll_interval=args.poll_interval,
        job_timeout=args.job_timeout,
    )

    X, Y_stored, idx = prob.get_initial_data_for_seed(args.n, args.seed)
    print(f"Loaded {X.shape[0]} historical rows (seed={args.seed}).")
    print(f"Historical row indices: {idx}")
    print()

    torch.set_printoptions(precision=4, sci_mode=False, linewidth=160)
    print("Stored Y (N x 3):  [y0=300-RMSE_t,  y1=200-fuel_g_t,  y2=0.05-settle]")
    print(Y_stored)
    print()

    print(f"Submitting Slurm array of {X.shape[0]} tasks for re-evaluation ...")
    Y_redo = prob._slurm_batch_fn(X)
    print()
    print("Re-evaluated Y (N x 3):")
    print(Y_redo)
    print()

    diff = Y_redo - Y_stored
    print("Diff (re-eval - stored):")
    print(diff)
    print()
    print("Per-column |diff| stats:")
    abs_diff = diff.abs()
    for j, name in enumerate(["y0_rmse", "y1_fuel", "y2_settle"]):
        col = abs_diff[:, j]
        print(
            f"  {name:14s}  max={col.max().item():.6g}  "
            f"mean={col.mean().item():.6g}  median={col.median().item():.6g}"
        )

    # Per-row summary (relative tolerance proxy: |diff| / max(1, |stored|))
    print()
    print("Row-wise |diff| (each y col):")
    for i in range(X.shape[0]):
        print(f"  row {i:2d} hist_idx={idx[i]:5d}  diff={diff[i].tolist()}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n": int(X.shape[0]),
        "seed": int(args.seed),
        "hist_indices": list(map(int, idx)),
        "X": X.tolist(),
        "Y_stored": Y_stored.tolist(),
        "Y_redo": Y_redo.tolist(),
        "abs_diff_max_per_col": abs_diff.max(dim=0).values.tolist(),
        "abs_diff_mean_per_col": abs_diff.mean(dim=0).tolist(),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
