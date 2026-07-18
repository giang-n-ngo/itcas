"""CASD-specific threshold calibration: draw + evaluate a moderate number of
real ``z=(x,c)`` points through a live ``scripts/casd_server/server.py``
instance, cache every result to disk, plot the resulting objective-space
distribution, and suggest candidate ``(tau_safety, tau_utility)`` pairs --
so the user can *look at the plots* and pick a threshold percentage
themselves, rather than trusting a fully-automated 200k-sample Monte Carlo
recipe that isn't feasible here (see ``ContextAwareSafeDecoding``'s class
docstring "OPEN ITEM" note in ``itcas/pipeline/problems.py``, and
``itcas/pipeline/thresholds.py``'s module docstring for the generic
synthetic-benchmark recipe this mirrors).

Why this needs its own script (not just a call to
``itcas.pipeline.thresholds.calibrate_thresholds``)
-----------------------------------------------------
``calibrate_thresholds()`` draws ``n_samples=200_000`` uniform points and
calls ``problem.fn(X)`` on *all of them in one shot*. That's fine for the
synthetic benchmarks (closed-form, sub-millisecond), but every
``ContextAwareSafeDecoding.fn`` call is a real HTTP round-trip to a
vLLM + judge-model server -- a single 200k-row batch would be an enormous
single vLLM generate() call bounded by one ``eval_timeout``-length HTTP
request, and 200k evaluations at *any* reasonable per-eval cost is
hours-to-days, not the intended calibration turnaround. So this script:
  1. draws a much smaller ``n_samples`` via ``sample_uniform`` (which
     already correctly samples design dims uniformly and context dims from
     real prompts via the server's ``/contexts`` endpoint),
  2. evaluates them in small chunks (default 32 rows/HTTP call, not one
     giant call), writing each chunk's results to disk *incrementally* so
     a run that dies partway through (GPU preemption, OOM, etc.) still
     leaves usable partial data behind, and
  3. reuses ``itcas.pipeline.thresholds.calibrate_thresholds_from_samples``
     (the sampling-agnostic bisection core factored out of
     ``calibrate_thresholds`` for exactly this purpose) against the
     resulting ``Y`` tensor, so the output is the exact same
     ``CalibrationResult`` shape -- and plugs into the exact same
     ``configs/thresholds.json`` / ``load_thresholds()`` / ``--threshold_pct``
     machinery -- as every other problem, no special-casing required.

Default sample budget and expected wall-clock (why n_samples=1000,
chunk_size=32)
------------------------------------------------------------------
Real per-eval cost observed on this cluster's H100 (see
``slurm_logs/casd_server/server-113220.out``) is on the order of seconds
per small batch via vLLM's continuous batching -- i.e. a chunk of 32 rows
costs roughly one such batch, not 32x a single-row cost. At
``n_samples=1000, chunk_size=32`` that's ~32 chunks, so this should
complete in low tens of minutes, not hours. Actual throughput depends on
GPU load and target-model size, so both are fully overridable via
``--n-samples`` / ``--chunk-size``.

This script is agnostic to mock vs. real mode: it only ever calls the
public ``Problem`` interface (``sample_uniform`` / ``evaluate``) plus the
server's ``/contexts`` endpoint (via ``casd_client``, best-effort, only to
recover ``context_id`` for the CSV). The exact same command works against
a mock server (``scripts/casd_server/server.py --mock``, for a fast local
dry run with no GPU) or a real one -- there is no mock-specific branching
here.

Usage
-----
Full run against a live server (writes samples.csv + candidate_thresholds.csv
+ 3 plots; does NOT touch configs/thresholds.json):

    CASD_SERVER_URL=http://localhost:8008 \\
        python -m itcas.pipeline.calibrate_casd --n-samples 1000 --chunk-size 32

Replot / recompute candidates from an already-cached CSV, without touching
the server at all:

    python -m itcas.pipeline.calibrate_casd --plot-only

Once you've looked at the plots and picked a percentage (e.g. 0.05), persist
just that one CalibrationResult into configs/thresholds.json (manual, opt-in
step -- this script never writes there on its own):

    python -m itcas.pipeline.calibrate_casd --plot-only --save-percentage 0.05
"""
from __future__ import annotations

import argparse
import csv
import os
import time
import warnings
from pathlib import Path
from typing import Optional, Sequence

import torch

from . import casd_client
from .problems import REGISTRY, ContextAwareSafeDecoding
from .thresholds import (
    DEFAULT_PERCENTAGES,
    DEFAULT_THRESHOLDS_PATH,
    CalibrationResult,
    calibrate_thresholds_from_samples,
    save_calibration,
)

DEFAULT_OUT_DIR = os.path.join("results", "casd_llm", "calibration")
SAMPLES_CSV_NAME = "samples.csv"
CANDIDATES_CSV_NAME = "candidate_thresholds.csv"
METHOD_LABEL = "equal-marginal-quantile-bisection-casd-live-samples"

CSV_FIELDS = [
    "row_index", "context_id", "temperature", "top_p", "repetition_penalty",
    "prompt_toxicity", "prompt_length", "f1", "f2",
]

# dataviz conventions copied verbatim from scripts/casd_server/analyze_dataset.py
# (see that file's header comment -- deliberate, already-reviewed style, not
# re-derived here).
BAR_COLOR = "#2a78d6"
MEAN_COLOR = "#e34948"
GRID_COLOR = "#d8d7d2"
TEXT_COLOR = "#0b0b0b"


# --------------------------------------------------------------------------- #
# Context-id recovery (best-effort; CSV column is left blank if unavailable)
# --------------------------------------------------------------------------- #
def _fetch_context_ids(
    problem: ContextAwareSafeDecoding, n: int, seed: int
) -> Optional[list]:
    """Best-effort recovery of the context_id for each row `sample_uniform`
    drew, by calling the same server endpoint (`casd_client.sample_contexts`)
    with the same (n, seed) `sample_uniform` used internally.

    `GET /contexts?n=K&seed=S` is documented as deterministic given (K, S)
    (see scripts/casd_server/server.py), and `sample_uniform` calls it with
    the exact same `n` and `int(seed)` before discarding the context_id
    column -- so this reproduces the same rows in the same order without
    reimplementing any sampling logic. Not load-bearing for calibration
    itself (only `prompt_toxicity`/`prompt_length`, already in X, matter for
    the objective distribution); purely a convenience column.
    """
    server_url = getattr(problem, "_server_url", None)
    timeout = getattr(problem, "_timeout", 30.0)
    if server_url is None:
        return None
    try:
        rows = casd_client.sample_contexts(server_url, n=n, seed=int(seed), timeout=timeout)
        if len(rows) < n:
            return None
        return [int(r["context_id"]) for r in rows[:n]]
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Sampling + chunked evaluation with incremental CSV writes
# --------------------------------------------------------------------------- #
def sample_and_evaluate(
    problem: ContextAwareSafeDecoding,
    n_samples: int,
    chunk_size: int,
    seed: int,
    samples_csv_path: str,
) -> torch.Tensor:
    """Draw `n_samples` z=(x,c) via `problem.sample_uniform`, evaluate them in
    chunks of `chunk_size` via `problem.evaluate`, writing each chunk to
    `samples_csv_path` (overwritten fresh at the start of the run) as soon as
    it completes -- so a run that dies partway through leaves partial, usable
    data on disk. Returns the (n_evaluated, 2) tensor of [f1, f2] for the
    rows that were actually written (<= n_samples if a chunk-level exception
    caused that chunk to be skipped).
    """
    print(f"[calibrate_casd] sampling {n_samples} points (seed={seed}) via sample_uniform ...")
    X = problem.sample_uniform(n_samples, seed=seed).to(torch.double)
    context_ids = _fetch_context_ids(problem, n_samples, seed)
    if context_ids is None:
        print("[calibrate_casd] context_id recovery unavailable; leaving that CSV column blank.")

    os.makedirs(os.path.dirname(samples_csv_path) or ".", exist_ok=True)
    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    Y_rows: list = []

    with open(samples_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        f.flush()
        os.fsync(f.fileno())

        for chunk_idx, start in enumerate(range(0, n_samples, chunk_size)):
            end = min(start + chunk_size, n_samples)
            X_chunk = X[start:end]
            t0 = time.time()
            try:
                Y_chunk = problem.evaluate(X_chunk)
            except Exception as exc:  # defensive: keep already-written chunks on disk
                warnings.warn(
                    f"[calibrate_casd] chunk {chunk_idx + 1}/{n_chunks} "
                    f"(rows {start}-{end - 1}) raised {exc!r}; skipping this "
                    f"chunk and continuing.",
                    RuntimeWarning,
                )
                continue
            dt = time.time() - t0

            rows = []
            for local_i in range(end - start):
                gi = start + local_i
                cid = context_ids[gi] if context_ids is not None else ""
                row = {
                    "row_index": gi,
                    "context_id": cid,
                    "temperature": float(X_chunk[local_i, 0]),
                    "top_p": float(X_chunk[local_i, 1]),
                    "repetition_penalty": float(X_chunk[local_i, 2]),
                    "prompt_toxicity": float(X_chunk[local_i, 3]),
                    "prompt_length": float(X_chunk[local_i, 4]),
                    "f1": float(Y_chunk[local_i, 0]),
                    "f2": float(Y_chunk[local_i, 1]),
                }
                rows.append(row)
                Y_rows.append([row["f1"], row["f2"]])

            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())  # persist this chunk even if the process dies next

            print(
                f"[calibrate_casd] chunk {chunk_idx + 1:3d}/{n_chunks} "
                f"rows {start:5d}-{end - 1:5d}  ({dt:5.1f}s)  "
                f"f1 in [{Y_chunk[:, 0].min():.3f}, {Y_chunk[:, 0].max():.3f}]  "
                f"f2 in [{Y_chunk[:, 1].min():.3f}, {Y_chunk[:, 1].max():.3f}]"
            )

    n_written = len(Y_rows)
    if n_written < n_samples:
        print(
            f"[calibrate_casd] WARNING: only {n_written}/{n_samples} rows written "
            f"(some chunks were skipped after errors -- see warnings above)."
        )
    print(f"[calibrate_casd] wrote {n_written} rows -> {samples_csv_path}")
    return torch.tensor(Y_rows, dtype=torch.double)


def load_samples_csv(samples_csv_path: str) -> torch.Tensor:
    """Load just the (N, 2) [f1, f2] columns back out of a cached samples.csv."""
    with open(samples_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = [[float(r["f1"]), float(r["f2"])] for r in reader]
    if not rows:
        raise ValueError(f"No rows found in {samples_csv_path}")
    return torch.tensor(rows, dtype=torch.double)


# --------------------------------------------------------------------------- #
# Candidate thresholds
# --------------------------------------------------------------------------- #
def compute_candidates(
    Y: torch.Tensor, percentages: Sequence[float], seed: int
) -> list[CalibrationResult]:
    return [
        calibrate_thresholds_from_samples(
            problem_name="casd_llm",
            Y=Y,
            target_fraction=p,
            n_samples=int(Y.shape[0]),
            seed=seed,
            method_label=METHOD_LABEL,
        )
        for p in percentages
    ]


def write_candidates_csv(results: list[CalibrationResult], out_path: str) -> None:
    fieldnames = [
        "percentage", "tau_safety", "tau_utility", "achieved_fraction",
        "quantile_level", "n_samples", "f1_max", "f2_max", "f1_min", "f2_min",
    ]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "percentage": r.target_fraction,
                "tau_safety": r.thresholds[0],
                "tau_utility": r.thresholds[1],
                "achieved_fraction": r.achieved_fraction,
                "quantile_level": r.quantile_level,
                "n_samples": r.n_samples,
                "f1_max": r.maxima[0],
                "f2_max": r.maxima[1],
                "f1_min": r.minima[0],
                "f2_min": r.minima[1],
            })
    print(f"[calibrate_casd] wrote {len(results)} candidate rows -> {out_path}")


# --------------------------------------------------------------------------- #
# Plots (styling copied verbatim from scripts/casd_server/analyze_dataset.py)
# --------------------------------------------------------------------------- #
def _new_axes():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")
    return fig, ax


def _style_axes(ax) -> None:
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)


def plot_scatter(Y: torch.Tensor, out_path: str) -> None:
    f1 = Y[:, 0].numpy()
    f2 = Y[:, 1].numpy()
    n = f1.shape[0]
    # Light alpha at moderate-to-large N to cope with overplotting; solid
    # markers for small N (no real overplotting concern below a few dozen).
    alpha = 0.35 if n > 150 else 0.8

    fig, ax = _new_axes()
    ax.scatter(f1, f2, s=15, color=BAR_COLOR, edgecolors="none", alpha=alpha, zorder=3)
    ax.set_title("CASD calibration samples: f1 (safety) vs f2 (utility)",
                 color=TEXT_COLOR, fontsize=12, loc="left", pad=10)
    ax.set_xlabel("f1: safety = 1 - P(toxic)", color=TEXT_COLOR, fontsize=9)
    ax.set_ylabel("f2: utility (reward-model logit)", color=TEXT_COLOR, fontsize=9)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"[calibrate_casd] wrote {out_path}")


def plot_histogram(
    values, label: str, xlabel: str, out_path: str, value_range=None
) -> None:
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    finite = values[~np.isnan(values)]
    mean = float(np.mean(finite)) if finite.size else float("nan")

    fig, ax = _new_axes()
    ax.hist(finite, bins=50, range=value_range, color=BAR_COLOR, edgecolor="none", zorder=3)
    ax.axvline(mean, color=MEAN_COLOR, linewidth=1.5, linestyle="--", zorder=4)
    ax.text(
        mean, ax.get_ylim()[1] * 0.96, f"  mean={mean:.3f}",
        color=MEAN_COLOR, fontsize=8, va="top", ha="left",
    )
    ax.set_title(label, color=TEXT_COLOR, fontsize=12, loc="left", pad=10)
    ax.set_xlabel(xlabel, color=TEXT_COLOR, fontsize=9)
    ax.set_ylabel("count", color=TEXT_COLOR, fontsize=9)
    if value_range is not None:
        ax.set_xlim(*value_range)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"[calibrate_casd] wrote {out_path}")


def make_plots(Y: torch.Tensor, out_dir: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    plot_scatter(Y, os.path.join(out_dir, "f1_vs_f2_scatter.png"))
    plot_histogram(
        Y[:, 0].numpy(), "CASD calibration samples: f1 (safety) marginal",
        "f1: safety = 1 - P(toxic)  [0, 1]",
        os.path.join(out_dir, "f1_histogram.png"), value_range=(0.0, 1.0),
    )
    plot_histogram(
        Y[:, 1].numpy(), "CASD calibration samples: f2 (utility) marginal",
        "f2: utility (reward-model logit, unbounded)",
        os.path.join(out_dir, "f2_histogram.png"), value_range=None,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        "itcas.pipeline.calibrate_casd", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--server-url", type=str, default=None,
                         help="Defaults to $CASD_SERVER_URL, then http://localhost:8008.")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--eval-timeout", type=float, default=300.0)
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--percentages", nargs="*", type=float, default=list(DEFAULT_PERCENTAGES),
        help="Candidate target joint-feasible fractions to report (default: the "
             "project-wide 0.20/0.10/0.05/0.01 set).",
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Skip sampling/evaluation (no server needed); recompute candidates "
             "and replot from an already-cached --samples-csv.",
    )
    parser.add_argument(
        "--samples-csv", type=str, default=None,
        help="Override the samples CSV path (default: <out-dir>/samples.csv). "
             "Used as the input in --plot-only mode, and as the write target otherwise.",
    )
    parser.add_argument(
        "--save-percentage", type=float, default=None,
        help="Opt-in: after computing candidates, persist the CalibrationResult "
             "for this one target_fraction into --thresholds-path via "
             "save_calibration() (matched to the nearest --percentages entry). "
             "Never done automatically -- omit this flag to just inspect the "
             "candidates/plots first.",
    )
    parser.add_argument("--thresholds-path", type=str, default=DEFAULT_THRESHOLDS_PATH)
    args = parser.parse_args(argv)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    samples_csv_path = args.samples_csv or os.path.join(out_dir, SAMPLES_CSV_NAME)
    candidates_csv_path = os.path.join(out_dir, CANDIDATES_CSV_NAME)

    if args.plot_only:
        print(f"[calibrate_casd] --plot-only: loading cached samples from {samples_csv_path}")
        Y = load_samples_csv(samples_csv_path)
    else:
        problem = REGISTRY["casd_llm"](
            server_url=args.server_url, timeout=args.timeout, eval_timeout=args.eval_timeout,
        )
        Y = sample_and_evaluate(
            problem, n_samples=args.n_samples, chunk_size=args.chunk_size,
            seed=args.seed, samples_csv_path=samples_csv_path,
        )

    print(f"[calibrate_casd] N={Y.shape[0]}  "
          f"f1 in [{Y[:, 0].min():.4f}, {Y[:, 0].max():.4f}]  "
          f"f2 in [{Y[:, 1].min():.4f}, {Y[:, 1].max():.4f}]")

    results = compute_candidates(Y, args.percentages, args.seed)
    write_candidates_csv(results, candidates_csv_path)
    for r in results:
        print(
            f"[calibrate_casd] p={r.target_fraction:>5g} -> "
            f"achieved={r.achieved_fraction * 100:6.3f}%  "
            f"tau_safety={r.thresholds[0]:.4f}  tau_utility={r.thresholds[1]:.4f}"
        )

    make_plots(Y, out_dir)

    if args.save_percentage is not None:
        match = min(results, key=lambda r: abs(r.target_fraction - args.save_percentage))
        save_calibration(match, path=args.thresholds_path)
        print(
            f"[calibrate_casd] saved calibration for p={match.target_fraction} "
            f"-> {args.thresholds_path} (['casd_llm']['{match.target_fraction:.6g}'])"
        )
    else:
        print(
            "\n[calibrate_casd] Nothing written to configs/thresholds.json (by design "
            "-- pick a percentage after looking at the plots/CSVs above). To persist "
            "one, re-run with --plot-only --save-percentage <p>, e.g.:\n\n"
            f"    python -m itcas.pipeline.calibrate_casd --plot-only "
            f"--samples-csv {samples_csv_path} --save-percentage 0.05\n"
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
