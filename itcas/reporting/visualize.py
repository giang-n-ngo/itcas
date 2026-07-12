"""Result visualization: per-metric comparison plots across methods.

For a given problem (benchmark) and an output directory containing one or more
runs (per-method, per-seed), this module produces *two PDFs per metric*:

    <problem>_<metric>_vs_evaluations.pdf
    <problem>_<metric>_vs_steps.pdf

Each PDF compares all methods on that problem: the curve is the seed-mean
across runs of the same ``(problem, method)`` pair, with a shaded +/- 1 std
band when more than one seed is present.

Metrics produced (see :mod:`itcas.reporting.metrics`):
    - cumulative_positives (Number of Positives)
    - context_fill_distance, feasible_context_fill_distance (contextual problems)
        - feasible_convex_hull_volume, epsilon_archive_size (objective diversity metrics;
            gracefully skipped if objective normalisation bounds are unavailable)

AUP is a single number (sum of the positives curve) and is reported in each
run's summary rather than plotted.

The module is also called automatically at the end of a CLI seed sweep so the
final state of a Slurm job emits up-to-date comparison plots.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Iterable, Optional

import torch

from .metrics import REGISTRY as METRIC_REGISTRY, MetricSpec, RunSeries, compute_metric, build_reference


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _iter_jsonl(path: Path) -> Iterable[dict]:
    """Yield JSON records, skipping lines that fail to parse.

    Per-iteration logs are written line-by-line during long Slurm runs, so a
    job that is killed mid-write — or, more rarely, two processes that
    accidentally share an output path — can leave a few malformed lines
    behind. Rather than aborting the whole summary/visualization, we warn
    once per file and skip just the bad lines so partial sweeps remain
    plottable.
    """
    bad = 0
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                if bad == 1:
                    import warnings
                    warnings.warn(
                        f"{path}: skipping malformed JSONL line {lineno} "
                        "(further bad lines in this file are skipped silently).",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                continue


def _to_tensor(rows) -> torch.Tensor:
    if rows is None or len(rows) == 0:
        return torch.empty(0, 0, dtype=torch.double)
    return torch.tensor(rows, dtype=torch.double)


def _load_run(jsonl_path: Path) -> Optional[RunSeries]:
    summary_path = jsonl_path.with_suffix(".summary.json")
    if not summary_path.exists():
        return None
    summary = _load_json(summary_path)
    cfg = summary.get("config", {})
    problem = str(summary.get("problem", cfg.get("problem", "unknown")))
    method = str(cfg.get("method", "unknown"))
    # When a method has named quality sub-variants (e.g. itcas/efig vs
    # itcas/roi_mi), include the variant in the method label so each
    # sub-variant gets its own curve instead of being averaged together.
    # Covers both `itcas` (batch) and its forced-sequential sibling
    # `itcas_seq`, e.g. -> "itcas_ndig" / "itcas_seq_ndig".
    quality = cfg.get("quality")
    if quality and method in ("itcas", "itcas_seq"):
        method = f"{method}_{quality}"
    seed = int(cfg.get("seed", 0))
    n_init = int(summary.get("n_init", cfg.get("n_init", 0)))
    thresholds = torch.tensor(summary.get("thresholds", []), dtype=torch.double)
    context_dims = tuple(summary.get("context_dims", ()))

    init_X = _to_tensor(summary.get("init_X"))
    init_Y = _to_tensor(summary.get("init_Y"))
    init_feas = list(summary.get("init_feasible", []))

    x_evals = [n_init]
    x_steps = [0]
    feasible_per_step: list[list[bool]] = [init_feas]
    X_per_step: list[torch.Tensor] = [init_X]
    Y_per_step: list[torch.Tensor] = [init_Y]

    # First pass: parse every record and bucket by `step`. Concurrent or
    # restarted writers (which truncate-then-append to the same path) can
    # leave a JSONL with duplicate and out-of-order iterations; without
    # deduping + sorting here the rebuilt cumulative state becomes
    # non-monotone, which manifests as a sawtooth in the plots. We keep the
    # last record seen for each step (writes within a step are atomic via
    # flock, so any one record is self-consistent).
    raw_by_step: dict[int, dict] = {}
    for rec in _iter_jsonl(jsonl_path):
        try:
            step = int(rec.get("step", rec.get("iter", 0) + 1))
        except (TypeError, ValueError):
            continue
        if step <= 0:
            continue
        raw_by_step[step] = rec

    # Only keep a contiguous prefix starting at step 1 — a gap means the run
    # was interrupted and we cannot reconstruct a consistent cumulative state
    # past the gap.
    ordered_steps = sorted(raw_by_step)
    contiguous: list[int] = []
    expected = 1
    for s in ordered_steps:
        if s != expected:
            break
        contiguous.append(s)
        expected += 1

    X_run = init_X
    Y_run = init_Y
    bad_records = 0
    for step in contiguous:
        rec = raw_by_step[step]
        try:
            n_eval_total_raw = rec.get("n_eval_total")
            x_new = _to_tensor(rec.get("x", []))
            y_new = _to_tensor(rec.get("y", []))
            feas = [bool(f) for f in rec.get("feasible", [])]
            if X_run.numel() > 0 and x_new.numel() > 0 and x_new.shape[-1] != X_run.shape[-1]:
                raise ValueError(f"x dim mismatch {tuple(x_new.shape)} vs {tuple(X_run.shape)}")
            if Y_run.numel() > 0 and y_new.numel() > 0 and y_new.shape[-1] != Y_run.shape[-1]:
                raise ValueError(f"y dim mismatch {tuple(y_new.shape)} vs {tuple(Y_run.shape)}")
            if X_run.numel() == 0:
                X_run = x_new
            elif x_new.numel() > 0:
                X_run = torch.cat([X_run, x_new], dim=0)
            if Y_run.numel() == 0:
                Y_run = y_new
            elif y_new.numel() > 0:
                Y_run = torch.cat([Y_run, y_new], dim=0)

            # Prefer the rebuilt running total over the record's own
            # `n_eval_total`, which may be inconsistent across the duplicate
            # writers that produced this file. Fall back to the recorded
            # value (or per-iter delta) only when no x payload is available.
            if X_run.numel() > 0:
                n_eval_total = int(X_run.shape[0])
            elif n_eval_total_raw is not None:
                n_eval_total = int(n_eval_total_raw)
            else:
                n_eval_total = x_evals[-1] + int(rec.get("n_eval_this_iter", 0))

            x_evals.append(n_eval_total)
            x_steps.append(step)
            feasible_per_step.append(feas)
            X_per_step.append(X_run)
            Y_per_step.append(Y_run)
        except (ValueError, TypeError, RuntimeError) as exc:
            bad_records += 1
            if bad_records == 1:
                import warnings
                warnings.warn(
                    f"{jsonl_path}: skipping malformed record ({exc}); "
                    "further bad records in this file are skipped silently.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            continue

    if len(raw_by_step) != len(contiguous):
        import warnings
        warnings.warn(
            f"{jsonl_path}: found {len(raw_by_step)} unique steps but only "
            f"{len(contiguous)} form a contiguous prefix from step 1 "
            "(likely concurrent/restarted writers). Truncating at the gap.",
            RuntimeWarning,
            stacklevel=2,
        )

    return RunSeries(
        problem=problem,
        method=method,
        run_name=jsonl_path.stem,
        seed=seed,
        thresholds=thresholds,
        context_dims=context_dims,
        config=cfg,
        x_evals=x_evals,
        x_steps=x_steps,
        feasible_per_step=feasible_per_step,
        X_per_step=X_per_step,
        Y_per_step=Y_per_step,
    )


def _discover_runs(input_dir: Path, benchmark: Optional[str] = None) -> list[RunSeries]:
    runs: list[RunSeries] = []
    for jsonl_path in sorted(input_dir.rglob("*.jsonl")):
        run = _load_run(jsonl_path)
        if run is None:
            continue
        if benchmark is not None and run.problem != benchmark:
            continue
        runs.append(run)
    return runs


def _methods_in(input_dir: Path, problem: Optional[str] = None) -> set[str]:
    """Collect distinct ``method`` values across summaries under ``input_dir``."""
    methods: set[str] = set()
    for summary_path in input_dir.rglob("*.summary.json"):
        try:
            summary = _load_json(summary_path)
        except (json.JSONDecodeError, OSError):
            continue
        if problem is not None and summary.get("problem") != problem:
            continue
        method = summary.get("config", {}).get("method")
        if method:
            methods.add(str(method))
    return methods


def resolve_comparison_root(
    start: str | Path, problem: Optional[str] = None, *, max_levels_up: int = 4
) -> Path:
    """Comparison root for cross-method plotting.

    Heuristic that handles both run layouts:

    * **Cluster** (``<root>/<problem>/<difficulty>/<method>/<run>``): ``start``
      is the per-method leaf and contains runs for exactly one method. The
      natural comparison root is the immediate parent (the difficulty dir),
      where every sibling method's runs live and where new methods will land.
    * **Flat** (``<root>/<run>``): ``start`` already contains runs of one or
      more methods directly; it is itself the comparison root.

    When ``problem`` appears in ``start``'s path components, we deterministically
    return ``<...>/<problem>/<difficulty>`` (the first segment under
    ``<problem>``). This avoids accidentally climbing to ``<problem>`` itself
    while early tasks are still running and only one method has completed under
    the target difficulty.

    Falls back to a method-count heuristic when ``problem`` is absent from the
    path (e.g. unconventional layouts). Walks at most ``max_levels_up`` levels
    in that fallback mode.
    """
    start_path = Path(start).resolve()

    # Prefer a deterministic, path-structure-based root in the standard cluster
    # layout: .../<problem>/<difficulty>/<method>[/quality]/...
    if problem:
        cur = start_path
        while True:
            if cur.name == problem:
                rel = start_path.relative_to(cur)
                if rel.parts:
                    return cur / rel.parts[0]
                break
            parent = cur.parent
            if parent == cur:
                break
            cur = parent

    start_n = len(_methods_in(start_path, problem=problem))

    # If start already has >=2 methods it is the flat layout; don't walk up.
    if start_n >= 2:
        return start_path

    # Walk upward and return the FIRST ancestor whose method count strictly
    # exceeds the starting directory's count.  Stopping at the first
    # improvement (rather than searching for the global maximum) prevents
    # ascending past the difficulty-level directory into the problem-level
    # directory, which would incorrectly mix runs from different difficulty
    # settings in the same plot.
    cur = start_path
    for _ in range(max_levels_up):
        parent = cur.parent
        if parent == cur:
            break
        parent_methods = _methods_in(parent, problem=problem)
        if not parent_methods:
            break
        if len(parent_methods) > start_n:
            return parent
        cur = parent

    return start_path


def _group_by_method(runs: list[RunSeries]) -> dict[str, list[RunSeries]]:
    grouped: dict[str, list[RunSeries]] = {}
    for run in runs:
        grouped.setdefault(run.method, []).append(run)
    return grouped


# ---------------------------------------------------------------------------
# Aggregation across seeds
# ---------------------------------------------------------------------------
def _trim_to_min(curves: list[list[float]]) -> list[list[float]]:
    if not curves:
        return []
    n = min(len(c) for c in curves)
    return [c[:n] for c in curves]


def _mean_std(curves: list[list[float]]) -> tuple[list[float], Optional[list[float]]]:
    trimmed = _trim_to_min(curves)
    if not trimmed:
        return [], None
    # Drop NaNs per index by averaging the finite values.
    mean: list[float] = []
    std: list[float] = []
    for col in zip(*trimmed):
        vals = [v for v in col if v == v]  # NaN check
        if not vals:
            mean.append(float("nan"))
            std.append(float("nan"))
            continue
        mean.append(fmean(vals))
        std.append(pstdev(vals) if len(vals) > 1 else 0.0)
    return mean, (None if len(trimmed) == 1 else std)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_metric(
    grouped: dict[str, list[RunSeries]],
    spec: MetricSpec,
    axis: str,
    out_path: Path,
    title: str,
    *,
    ref,
) -> Optional[Path]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("tab10").colors
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.grid(True, alpha=0.25)
    ax.set_ylabel(spec.label)
    ax.set_xlabel("Total individual evaluations" if axis == "evals" else "Algorithmic step")
    ax.set_title(title)

    plotted = False
    for idx, method in enumerate(sorted(grouped)):
        runs = grouped[method]
        curves: list[list[float]] = []
        x_axis: Optional[list[int]] = None
        for run in runs:
            y = compute_metric(run, spec, ref=ref)
            if y is None:
                continue
            curves.append(y)
            x_attr = run.x_evals if axis == "evals" else run.x_steps
            x_axis = x_attr if x_axis is None else x_axis
        if not curves or x_axis is None:
            continue
        mean, std = _mean_std(curves)
        n = min(len(mean), len(x_axis))
        x_axis = x_axis[:n]
        mean = mean[:n]
        color = colors[idx % len(colors)]
        ax.plot(x_axis, mean, color=color, linewidth=2, marker="o", markersize=3, label=method)
        if std is not None:
            std = std[:n]
            lower = [m - s for m, s in zip(mean, std)]
            upper = [m + s for m, s in zip(mean, std)]
            ax.fill_between(x_axis, lower, upper, color=color, alpha=0.15)
        plotted = True

    if not plotted:
        plt.close(fig)
        return None
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def visualize_benchmark(
    input_dir: str | Path,
    benchmark: Optional[str] = None,
    output_dir: str | Path | None = None,
    *,
    n_grid: int = 2000,
) -> list[str]:
    """Generate per-metric comparison PDFs for one benchmark.

    For each metric in :data:`metrics.REGISTRY`, two PDFs are written into
    ``output_dir`` (defaults to ``input_dir``): one with x-axis = total
    evaluations, one with x-axis = algorithmic step. Returns the list of
    generated file paths.
    """
    input_path = Path(input_dir)
    out_dir = Path(output_dir) if output_dir is not None else input_path

    runs = _discover_runs(input_path, benchmark=benchmark)
    if not runs:
        raise ValueError(f"No run logs found in {input_path}")

    problems = sorted({run.problem for run in runs})
    if benchmark is None and len(problems) != 1:
        raise ValueError(
            f"Multiple benchmarks found ({problems}); pass benchmark=..."
        )
    problem = benchmark or problems[0]
    selected = [r for r in runs if r.problem == problem]
    grouped = _group_by_method(selected)
    if not grouped:
        raise ValueError(f"No runs found for benchmark '{problem}' in {input_path}")

    # Build the problem-specific reference sets once (shared across the
    # fill-distance metrics). Uses any run with thresholds populated; runs
    # without thresholds (or unknown problems) skip the fill-distance metrics.
    ref = None
    for r in selected:
        if r.thresholds.numel() > 0:
            ref = build_reference(r, n_context=n_grid)
            break

    out: list[str] = []
    for spec in METRIC_REGISTRY.values():
        for axis, suffix in (("evals", "vs_evaluations"), ("steps", "vs_steps")):
            path = out_dir / f"{problem}_{spec.key}_{suffix}.pdf"
            label_axis = "total individual evaluations" if axis == "evals" else "algorithmic step"
            ok = _plot_metric(
                grouped, spec, axis, path,
                title=f"{problem}: {spec.label} vs {label_axis}",
                ref=ref,
            )
            if ok is not None:
                out.append(str(ok))
    return out


def visualize_all_benchmarks(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
) -> list[str]:
    """Generate plots for every benchmark present in ``input_dir``."""
    input_path = Path(input_dir)
    runs = _discover_runs(input_path)
    if not runs:
        return []
    paths: list[str] = []
    for problem in sorted({r.problem for r in runs}):
        try:
            paths.extend(visualize_benchmark(input_path, benchmark=problem, output_dir=output_dir))
        except ValueError:
            continue
    return paths


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser("itcas.visualize")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--benchmark", type=str, default=None,
                        help="Restrict to one problem; default plots all problems found.")
    args = parser.parse_args(argv)

    if args.benchmark is None:
        paths = visualize_all_benchmarks(args.input_dir, output_dir=args.output_dir)
    else:
        paths = visualize_benchmark(args.input_dir, benchmark=args.benchmark,
                                    output_dir=args.output_dir)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
