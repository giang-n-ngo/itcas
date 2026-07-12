#!/usr/bin/env python3
"""Temporary helper to plot sampled points for one run.

Reads a run summary JSON plus optional JSONL iteration log, reconstructs
all sampled points (initial + acquired), and writes 2D scatter plots when:

- number of objectives is 2 (objective-space plot)
- number of context dimensions is 2 (context-space plot)

Examples
--------
python scripts/temp_plot_run_points.py \
  --summary results/sweep/sphere2_6d/p0_1/itcas/run_seed0.summary.json

python scripts/temp_plot_run_points.py \
  --summary results/smoke/itcas_smoke.summary.json \
  --jsonl results/smoke/itcas_smoke.jsonl \
  --out-dir results/smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _to_2d_float(arr) -> np.ndarray:
    if arr is None:
        return np.empty((0, 0), dtype=float)
    out = np.asarray(arr, dtype=float)
    if out.ndim == 1:
        return out.reshape(-1, 1)
    if out.ndim == 2:
        return out
    return np.empty((0, 0), dtype=float)


def _stack_rows(parts: list[np.ndarray]) -> np.ndarray:
    parts = [p for p in parts if p.ndim == 2 and p.shape[0] > 0]
    if not parts:
        return np.empty((0, 0), dtype=float)
    n_cols = parts[0].shape[1]
    compatible = [p for p in parts if p.shape[1] == n_cols]
    if not compatible:
        return np.empty((0, 0), dtype=float)
    return np.concatenate(compatible, axis=0)


def _default_jsonl_path(summary_path: Path) -> Path:
    name = summary_path.name
    if name.endswith(".summary.json"):
        return summary_path.with_name(name[:-13] + ".jsonl")
    return summary_path.with_suffix(".jsonl")


def _default_run_name(summary_path: Path) -> str:
    name = summary_path.name
    if name.endswith(".summary.json"):
        return name[:-13]
    return summary_path.stem


def _plot_objectives(Y: np.ndarray, thresholds: np.ndarray, n_init: int, out_path: Path) -> bool:
    if Y.ndim != 2 or Y.shape[1] != 2 or Y.shape[0] == 0:
        return False
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    n_init = max(0, min(int(n_init), int(Y.shape[0])))
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    if n_init > 0:
        ax.scatter(Y[:n_init, 0], Y[:n_init, 1], s=20, alpha=0.8, label="init")
    if Y.shape[0] > n_init:
        ax.scatter(
            Y[n_init:, 0],
            Y[n_init:, 1],
            s=24,
            alpha=0.85,
            marker="x",
            label="acquired",
        )
    tau = np.asarray(thresholds, dtype=float).reshape(-1)
    if tau.shape[0] >= 2:
        x_lo, x_hi = float(np.min(Y[:, 0])), float(np.max(Y[:, 0]))
        y_lo, y_hi = float(np.min(Y[:, 1])), float(np.max(Y[:, 1]))
        x_lo, x_hi = min(x_lo, float(tau[0])), max(x_hi, float(tau[0]))
        y_lo, y_hi = min(y_lo, float(tau[1])), max(y_hi, float(tau[1]))
        x_pad = 0.05 * max(1e-9, x_hi - x_lo)
        y_pad = 0.05 * max(1e-9, y_hi - y_lo)
        ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
        ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
        ax.axvline(
            tau[0],
            color="black",
            linestyle="--",
            linewidth=2.0,
            alpha=0.9,
            zorder=10,
            label="threshold obj0",
        )
        ax.axhline(
            tau[1],
            color="dimgray",
            linestyle="--",
            linewidth=2.0,
            alpha=0.9,
            zorder=10,
            label="threshold obj1",
        )
    ax.set_title("Sampled points in objective space")
    ax.set_xlabel("objective 0")
    ax.set_ylabel("objective 1")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def _plot_contexts(X: np.ndarray, context_dims: list[int], n_init: int, out_path: Path) -> bool:
    if X.ndim != 2 or X.shape[0] == 0 or len(context_dims) != 2:
        return False
    c0, c1 = int(context_dims[0]), int(context_dims[1])
    if c0 < 0 or c1 < 0 or c0 >= X.shape[1] or c1 >= X.shape[1]:
        return False
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    n_init = max(0, min(int(n_init), int(X.shape[0])))
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    if n_init > 0:
        ax.scatter(X[:n_init, c0], X[:n_init, c1], s=20, alpha=0.8, label="init")
    if X.shape[0] > n_init:
        ax.scatter(
            X[n_init:, c0],
            X[n_init:, c1],
            s=24,
            alpha=0.85,
            marker="x",
            label="acquired",
        )
    ax.set_title("Sampled points in context space")
    ax.set_xlabel(f"x[{c0}]")
    ax.set_ylabel(f"x[{c1}]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser("temp_plot_run_points")
    parser.add_argument("--summary", required=True, help="Path to <run>.summary.json")
    parser.add_argument("--jsonl", default=None, help="Path to <run>.jsonl (optional)")
    parser.add_argument("--out-dir", default=None, help="Where to write PNG files")
    parser.add_argument(
        "--prefix",
        default=None,
        help="Filename prefix (default: run name inferred from summary file)",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        raise SystemExit(f"summary not found: {summary_path}")

    summary = _load_json(summary_path)
    jsonl_path = Path(args.jsonl) if args.jsonl else _default_jsonl_path(summary_path)

    init_X = _to_2d_float(summary.get("init_X", []))
    init_Y = _to_2d_float(summary.get("init_Y", []))
    n_init = int(summary.get("n_init", init_X.shape[0]))
    context_dims = list(summary.get("context_dims", []))
    thresholds = np.asarray(summary.get("thresholds", []), dtype=float)

    xs = [init_X]
    ys = [init_Y]
    if jsonl_path.exists():
        for rec in _iter_jsonl(jsonl_path):
            x_new = _to_2d_float(rec.get("x", []))
            y_new = _to_2d_float(rec.get("y", []))
            if x_new.shape[0] > 0:
                xs.append(x_new)
            if y_new.shape[0] > 0:
                ys.append(y_new)

    X = _stack_rows(xs)
    Y = _stack_rows(ys)

    out_dir = Path(args.out_dir) if args.out_dir else summary_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or _default_run_name(summary_path)

    obj_path = out_dir / f"{prefix}.objectives2d.png"
    ctx_path = out_dir / f"{prefix}.context2d.png"

    made_obj = _plot_objectives(Y, thresholds=thresholds, n_init=n_init, out_path=obj_path)
    made_ctx = _plot_contexts(X, context_dims=context_dims, n_init=n_init, out_path=ctx_path)

    if made_obj:
        print(obj_path)
    else:
        print("skipped objective plot (need exactly 2 objectives and non-empty points)")

    if made_ctx:
        print(ctx_path)
    else:
        print("skipped context plot (need exactly 2 context dims and non-empty points)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())