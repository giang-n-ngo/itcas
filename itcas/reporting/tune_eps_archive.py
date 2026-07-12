"""Offline calibration of the ``eps_archive`` distance threshold.

Implements ``contexts/metrics.md`` §4, "Scale-Invariant ε-Archive Size
(Gridless Micro-Diversity)". That metric (``itcas.metrics.epsilon_archive_size``,
consumed at runtime via ``itcas.reporting.metrics.epsilon_archive_size_curve``)
needs a distance threshold ``eps`` in *transformed* objective space. This
module computes that threshold **offline**, from already-completed sweep data,
and writes it into ``configs/experiments.json`` per ``(problem, difficulty)``
(difficulty matters because the feasibility threshold ``tau`` — and hence the
log-transform — differs per difficulty).

Per the spec, for one ``(problem, difficulty)`` group:

1. **Pool**: gather every strictly feasible objective vector ``Y`` discovered
   by *any method, any seed* under ``results/sweep/<problem>/<difficulty>/``.
2. **Log-transform** relative to the feasibility threshold ``tau``:
   ``y'_i = log(1 + (y_i - tau_i))`` (defined since feasibility guarantees
   ``y_i - tau_i >= 0``).
3. Compute the pairwise Euclidean distance matrix over the pooled transformed
   set (``scipy.spatial.distance.pdist``).
4. Filter out zero distances (self/exact-duplicate pairs) — *before* taking
   the percentile.
5. ``eps = 5th percentile`` of the remaining strictly-positive distances.

Minimum sample size: ``pdist`` requires >= 2 pooled feasible points, but a
5th-percentile estimate from a handful of points is unreliable (extrapolating
from very few pairwise distances). We require at least ``MIN_POOLED_FEASIBLE``
(default 10) pooled feasible points — giving >= 45 pairwise distances, enough
that the 5th percentile reflects several near-duplicate pairs rather than a
single one. Groups with fewer pooled feasible points are skipped (``eps =
None``); the config writer then leaves that slot untouched so the existing
resolution order (``problems[problem][difficulty] -> problems[problem].defaults
-> defaults``) falls back to the hand-set value already on disk.

Scaling notes (this matters at full-sweep scale — ~40k runs / ~4.6 GB of
JSONL under ``results/sweep``):

* This module does **not** use ``itcas.reporting.visualize._load_run`` /
  ``RunSeries``. That loader is built for *plotting* and reconstructs a full
  per-iteration cumulative history (``X_per_step``/``Y_per_step``, one growing
  tensor snapshot per step — O(T^2) tensor elements per run), even though
  calibration only ever needs the final cumulative ``Y``. Instead,
  :func:`load_run_final` parses each ``.jsonl`` directly into just the final
  concatenated ``(thresholds, Y)`` pair — O(T) per run, no per-step snapshots,
  and no ``X``/context columns at all (not needed for this metric).
* :func:`calibrate_all` processes exactly one ``(problem, difficulty)`` group
  at a time: it loads that group's runs, computes its ``eps``, and discards
  the loaded runs before moving to the next group, rather than holding the
  entire sweep's parsed runs in memory simultaneously.

Usage::

    python -m itcas.reporting.tune_eps_archive \
        --sweep-root results/sweep --config configs/experiments.json

    # Preview without writing:
    python -m itcas.reporting.tune_eps_archive --dry-run
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
from scipy.spatial.distance import pdist

from ..metrics import is_feasible
from .visualize import _iter_jsonl

DEFAULT_PERCENTILE = 5.0
# Minimum pooled feasible points required before we trust a percentile
# estimate (see module docstring for the rationale).
MIN_POOLED_FEASIBLE = 10
# scipy.spatial.distance.pdist is O(n^2) in both time and memory (a condensed
# array of n*(n-1)/2 float64 entries, plus further same-sized working copies
# inside the `d > 0` filter and `np.percentile`'s internal partition). Some
# (problem, difficulty) groups pool tens of thousands of feasible points
# across all methods/seeds (observed: ~30k for an easy `all_valley_8d`
# difficulty), which would need multiple GB just for the pdist array and
# ~15-20 GB peak RSS end-to-end once those working copies are accounted for
# -- unsafe to run unattended even on a dedicated Slurm CPU node. Beyond
# MAX_POOL_SIZE we take a fixed-seed uniform random subsample of the pooled
# points before computing pairwise distances: a percentile of pairwise
# distances among an i.i.d. subsample of the point cloud converges to the
# same distribution as over the full set (the same logic behind the common
# "median heuristic" bandwidth estimators), so this trades a small, bounded
# amount of estimation variance for a hard cap on memory/compute.
DEFAULT_MAX_POOL_SIZE = 5000
_SUBSAMPLE_SEED = 0

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_EXPERIMENTS_PATH = _REPO_ROOT / "configs" / "experiments.json"
_SWEEP_ROOT = _REPO_ROOT / "results" / "sweep"

# One run's calibration-relevant state: its feasibility threshold and the
# final cumulative objective matrix (N_total_evals, m). No X/context, no
# per-step snapshots.
FinalState = tuple[torch.Tensor, torch.Tensor]


# ---------------------------------------------------------------------------
# Lightweight per-run loader (final Y only — no O(T^2) per-step history)
# ---------------------------------------------------------------------------
def _rows_to_tensor(rows) -> torch.Tensor:
    if rows is None or len(rows) == 0:
        return torch.empty(0, 0, dtype=torch.double)
    return torch.tensor(rows, dtype=torch.double)


def load_run_final(jsonl_path: Path) -> Optional[FinalState]:
    """Parse one run's ``.jsonl`` + sibling ``.summary.json`` into just the
    final ``(thresholds, Y)`` pair needed for ε-archive calibration.

    Unlike ``itcas.reporting.visualize._load_run`` this keeps no per-step
    ``X_per_step``/``Y_per_step`` history — only a ``dict[step] -> y_rows``
    map (O(T) records, not O(T^2) tensor elements) which is concatenated once
    at the end. Duplicate/out-of-order steps (e.g. from a restarted writer)
    are deduped by step number and truncated at the first gap, mirroring
    ``_load_run``'s "keep only the contiguous prefix from step 1" rule so the
    reconstructed final state is consistent with what the plotting path would
    report. Returns ``None`` when the summary is missing or carries no
    thresholds.
    """
    summary_path = jsonl_path.with_suffix(".summary.json")
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    thresholds = torch.tensor(summary.get("thresholds", []), dtype=torch.double)
    if thresholds.numel() == 0:
        return None

    init_Y = _rows_to_tensor(summary.get("init_Y"))

    raw_y_by_step: dict[int, list] = {}
    for rec in _iter_jsonl(jsonl_path):
        try:
            step = int(rec.get("step", rec.get("iter", 0) + 1))
        except (TypeError, ValueError):
            continue
        if step <= 0:
            continue
        raw_y_by_step[step] = rec.get("y", [])

    ordered_steps = sorted(raw_y_by_step)
    contiguous: list[int] = []
    expected = 1
    for s in ordered_steps:
        if s != expected:
            break
        contiguous.append(s)
        expected += 1

    y_chunks: list[torch.Tensor] = [init_Y] if init_Y.numel() > 0 else []
    for step in contiguous:
        y_new = _rows_to_tensor(raw_y_by_step[step])
        if y_new.numel() > 0:
            y_chunks.append(y_new)

    if not y_chunks:
        return thresholds, init_Y
    Y_final = y_chunks[0] if len(y_chunks) == 1 else torch.cat(y_chunks, dim=0)
    return thresholds, Y_final


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------
def pooled_transformed_feasible(final_states: list[FinalState]) -> Optional[torch.Tensor]:
    """Pool + log-transform the strictly feasible objective vectors of ``final_states``.

    Each run's *own* ``thresholds`` tensor is used to transform its own
    feasible points (robust to any per-run floating point deviations; within
    one ``(problem, difficulty)`` group all runs are expected to share the
    same ``tau``). Returns ``None`` if no run in the group has any feasible
    point at all.
    """
    rows: list[torch.Tensor] = []
    for thresholds, Y in final_states:
        if Y is None or Y.numel() == 0:
            continue
        mask = is_feasible(Y, thresholds)
        if not bool(mask.any()):
            continue
        feas = Y[mask].detach().double()
        transformed = torch.log1p(feas - thresholds.double())
        rows.append(transformed)
    if not rows:
        return None
    return torch.cat(rows, dim=0)


def eps_from_pool(
    pooled: torch.Tensor,
    *,
    percentile: float = DEFAULT_PERCENTILE,
    min_points: int = MIN_POOLED_FEASIBLE,
    max_pool_size: int = DEFAULT_MAX_POOL_SIZE,
) -> Optional[float]:
    """5th-percentile (default) of strictly-positive pairwise distances.

    Returns ``None`` when fewer than ``min_points`` pooled points are
    available, or when every pairwise distance is zero (all points coincide).
    When more than ``max_pool_size`` feasible points are pooled, a fixed-seed
    uniform random subsample of that size is used instead of the full set to
    keep the O(n^2) ``pdist`` call bounded in time/memory (see
    ``DEFAULT_MAX_POOL_SIZE`` docstring above for the rationale).
    """
    n = int(pooled.shape[0])
    if n < min_points:
        return None
    arr = pooled.cpu().numpy()
    if n > max_pool_size:
        rng = np.random.default_rng(_SUBSAMPLE_SEED)
        idx = rng.choice(n, size=max_pool_size, replace=False)
        arr = arr[idx]
    d = pdist(arr)
    d = d[d > 0.0]
    if d.size == 0:
        return None
    return float(np.percentile(d, percentile))


def compute_eps_for_group(
    final_states: list[FinalState],
    *,
    percentile: float = DEFAULT_PERCENTILE,
    min_points: int = MIN_POOLED_FEASIBLE,
    max_pool_size: int = DEFAULT_MAX_POOL_SIZE,
) -> Optional[float]:
    """Calibrate ``eps_archive`` for one ``(problem, difficulty)`` group of runs."""
    pooled = pooled_transformed_feasible(final_states)
    if pooled is None:
        return None
    return eps_from_pool(
        pooled, percentile=percentile, min_points=min_points, max_pool_size=max_pool_size,
    )


# ---------------------------------------------------------------------------
# Difficulty-dir <-> config-key mapping
# ---------------------------------------------------------------------------
def config_key_from_difficulty_dir(dirname: str) -> str:
    """Map a difficulty directory name to its ``configs/experiments.json`` key.

    Directory names look like ``p0_01``, ``p0_05``, ``p0_1``, ``p0_2`` (most
    problems) or ``p1`` .. ``p10`` (``spacecraft_formation_flying_a1``). The
    config keys are the ``threshold_pct`` string used by the launcher
    (``"0.01"``, ``"0.1"``, ..., ``"1"`` .. ``"10"``). Turning ``p0_01`` into
    ``0.01`` (and ``p1`` into ``1``, with no dot inserted since there is no
    ``"_"``) is a plain string transform — replace the leading ``p`` and
    convert the *first* ``_`` into a ``.``:

        p0_01 -> 0.01, p0_1 -> 0.1, p0_2 -> 0.2, p1 -> 1, p10 -> 10
    """
    body = dirname[1:] if dirname.startswith("p") else dirname
    return body.replace("_", ".", 1)


# ---------------------------------------------------------------------------
# Driver: walk results/sweep, calibrate eps ONE (problem, difficulty) group
# at a time, discarding each group's loaded runs before moving to the next.
# ---------------------------------------------------------------------------
def iter_group_dirs(
    sweep_root: Path, *, problem_filter: Optional[str] = None
) -> Iterator[tuple[str, Path]]:
    """Yield ``(problem, difficulty_dir)`` pairs across the sweep tree.

    Pure directory-structure walk — no run data is loaded here, so nothing is
    materialized ahead of what :func:`calibrate_all` is about to process.
    """
    for problem_dir in sorted(p for p in sweep_root.iterdir() if p.is_dir()):
        if problem_dir.name.startswith("_"):
            continue
        if problem_filter is not None and problem_dir.name != problem_filter:
            continue
        for diff_dir in sorted(d for d in problem_dir.iterdir() if d.is_dir()):
            yield problem_dir.name, diff_dir


def calibrate_all(
    sweep_root: Path,
    *,
    percentile: float = DEFAULT_PERCENTILE,
    min_points: int = MIN_POOLED_FEASIBLE,
    max_pool_size: int = DEFAULT_MAX_POOL_SIZE,
    problem_filter: Optional[str] = None,
    verbose: bool = False,
) -> dict[str, dict[str, Optional[float]]]:
    """``{problem: {config_key: eps_or_None}}`` for every discovered group.

    Processes one ``(problem, difficulty)`` directory at a time: loads only
    that group's runs (lightweight final-``Y``-only state), computes its
    ``eps``, and drops the reference before moving to the next group — the
    full sweep's parsed run data is never held in memory simultaneously.
    """
    result: dict[str, dict[str, Optional[float]]] = {}
    for problem, diff_dir in iter_group_dirs(sweep_root, problem_filter=problem_filter):
        final_states: list[FinalState] = []
        for jsonl_path in sorted(diff_dir.rglob("*.jsonl")):
            loaded = load_run_final(jsonl_path)
            if loaded is not None:
                final_states.append(loaded)

        n_runs = len(final_states)
        eps = compute_eps_for_group(
            final_states, percentile=percentile, min_points=min_points, max_pool_size=max_pool_size,
        )
        key = config_key_from_difficulty_dir(diff_dir.name)
        result.setdefault(problem, {})[key] = eps
        if verbose:
            status = f"{eps:.6g}" if eps is not None else "SKIPPED (insufficient pooled feasible data)"
            print(f"  {problem:35s} {diff_dir.name:8s} ({n_runs:4d} runs) eps_archive = {status}")

        # Explicitly drop this group's runs before moving on to the next
        # directory — keeps peak memory bounded by one group, not the whole
        # sweep (~40k files / ~4.6 GB across the full tree).
        final_states.clear()
        del final_states

    return result


# ---------------------------------------------------------------------------
# Writer: patch configs/experiments.json in place
# ---------------------------------------------------------------------------
def update_experiments_config(
    eps_by_group: dict[str, dict[str, Optional[float]]],
    config_path: Path = _EXPERIMENTS_PATH,
) -> dict[str, list]:
    """Write calibrated ``eps_archive`` values into ``config_path``.

    Only non-``None`` entries are written, as a minimal
    ``{"eps_archive": <value>}`` block merged into any existing per-difficulty
    override (preserving its other fields, e.g. ``budget``/``batch_size``/
    ``n_init``). ``None`` entries (insufficient pooled data) are left
    untouched entirely, so the existing resolution order
    (``problems[problem][difficulty] -> problems[problem].defaults ->
    defaults``) transparently falls back to whatever hand-set value is
    already on disk for that slot.

    The top-level structure (``_comment``, ``defaults``, per-problem
    ``_comment``/``defaults``) is preserved as-is; only the per-difficulty
    ``eps_archive`` numeric field is touched.
    """
    data = json.loads(config_path.read_text())
    problems = data.setdefault("problems", {})

    updated: list[tuple[str, str, float]] = []
    skipped: list[tuple[str, str]] = []
    missing_problem: list[str] = []

    for problem, diffs in eps_by_group.items():
        prob_cfg = problems.get(problem)
        if prob_cfg is None:
            missing_problem.append(problem)
            continue
        for key, eps in diffs.items():
            if eps is None:
                skipped.append((problem, key))
                continue
            block = prob_cfg.get(key)
            if not isinstance(block, dict):
                block = {}
            # 6 significant figures keeps the file readable across the wide
            # dynamic range of eps_archive values (~1e-3 .. ~1e3) without
            # spurious float64 noise in the trailing digits.
            block["eps_archive"] = float(f"{eps:.6g}")
            prob_cfg[key] = block
            updated.append((problem, key, eps))

    if missing_problem:
        warnings.warn(
            "tune_eps_archive: the following problems were found under "
            f"results/sweep but are not listed in {config_path}: "
            f"{missing_problem}; skipping.",
            RuntimeWarning,
        )

    config_path.write_text(json.dumps(data, indent=2) + "\n")
    return {"updated": updated, "skipped": skipped, "missing_problem": missing_problem}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep-root", type=Path, default=_SWEEP_ROOT, help="Root of results/sweep")
    parser.add_argument("--config", type=Path, default=_EXPERIMENTS_PATH, help="configs/experiments.json to patch")
    parser.add_argument("--percentile", type=float, default=DEFAULT_PERCENTILE, help="Percentile of pairwise distances (default 5)")
    parser.add_argument("--min-points", type=int, default=MIN_POOLED_FEASIBLE, help="Minimum pooled feasible points to calibrate (default 10)")
    parser.add_argument("--max-pool-size", type=int, default=DEFAULT_MAX_POOL_SIZE, help="Subsample pooled feasible points beyond this size before pdist (default 5000; caps O(n^2) memory/time)")
    parser.add_argument("--problem", type=str, default=None, help="Only calibrate this problem (for testing)")
    parser.add_argument("--dry-run", action="store_true", help="Compute and print, but do not write the config file")
    args = parser.parse_args(argv)

    print(f"Scanning {args.sweep_root} ...")
    eps_by_group = calibrate_all(
        args.sweep_root,
        percentile=args.percentile,
        min_points=args.min_points,
        max_pool_size=args.max_pool_size,
        problem_filter=args.problem,
        verbose=True,
    )

    if args.dry_run:
        print("(dry run: configs/experiments.json not modified)")
        return

    report = update_experiments_config(eps_by_group, args.config)
    print(
        f"\nWrote {args.config}: updated {len(report['updated'])} (problem, difficulty) "
        f"entries; skipped {len(report['skipped'])} (insufficient pooled feasible data, "
        "kept existing config value)."
    )
    if report["skipped"]:
        print("Skipped:")
        for problem, key in report["skipped"]:
            print(f"  {problem} [{key}]")
    if report["missing_problem"]:
        print(f"WARNING: problems not found in config: {report['missing_problem']}")


if __name__ == "__main__":
    main()
