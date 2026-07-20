"""Per-problem summary plots aggregating metric *areas* across seeds.

Unlike :mod:`itcas.reporting.visualize` which produces per-iteration line
charts, this module collapses each ``(method, problem, difficulty, seed)``
metric curve to a single scalar — the area under the curve along the chosen
x-axis — and renders horizontal boxplots over seeds.

Output structure (one PDF per ``problem`` per axis):

    <problem>_summary_vs_evaluations.pdf
    <problem>_summary_vs_steps.pdf

Each PDF contains one subplot row per difficulty (e.g. ``p0_05``,
``p0_10``...) and one column per metric:

    [ #pos | FCFD | FCHV | ε-Archive | hypervolume ]

* x-axis: metric value (area). y-axis: method names.
* Each subplot stacks horizontal boxplots, one per method, summarising the
  spread of the per-seed area for that ``(method, difficulty, metric)``.
* "Hypervolume" is the per-seed combined ranking score (higher = better).
  Higher-is-better metrics (#positives, FCHV, ε-Archive Size) contribute
  directly; FCFD contributes as its reciprocal (1/area), so that a smaller
  fill-distance (better) always increases the product.
"""
from __future__ import annotations

import argparse
import bisect
import json
import re
from pathlib import Path
from statistics import median
from typing import Iterable, Optional

import torch

from .metrics import REGISTRY as METRIC_REGISTRY, MetricSpec, RunSeries, build_reference, compute_metric
from .visualize import _discover_runs
from .stats import (
    StatsReport,
    _fmt,
    _sig_marker,
    report_to_json,
    run_stats,
    run_stats_per_variant,
    write_dominance_table,
    write_stats_report,
)


# ---------------------------------------------------------------------------
# Difficulty / budget helpers
# ---------------------------------------------------------------------------
_PROBLEM_DIFF_RE = re.compile(r"/(?P<problem>[^/]+)/(?P<difficulty>p\d+_\d+)/")


def _difficulty_of(run: RunSeries) -> str:
    """Return a stable, sortable difficulty label.

    Prefers ``cfg.extra.threshold_pct`` (formatted like ``p0_05``) and falls
    back to scraping the run's ``out_dir`` for a ``p<int>_<int>`` segment.
    Returns ``"default"`` when neither is available.
    """
    extra = run.config.get("extra") or {}
    pct = extra.get("threshold_pct")
    if pct is not None:
        try:
            v = float(pct)
            return f"p{int(v):d}_{int(round((v - int(v)) * 100)):02d}"
        except (TypeError, ValueError):
            pass
    out_dir = str(run.config.get("out_dir", ""))
    m = re.search(r"p\d+_\d+", out_dir)
    if m:
        return m.group(0)
    return "default"


# ---------------------------------------------------------------------------
# Area integration
# ---------------------------------------------------------------------------
def _trapz(x: list[float], y: list[float]) -> float:
    """Geometric area between the curve and the x-axis (always >= 0).

    Integrates ``|y|`` rather than the signed value so any curve that dips
    below zero still contributes positive area. Segments that cross the x-axis
    are split at the zero-crossing so each side is measured against the axis.
    """
    if len(x) < 2:
        return 0.0
    s = 0.0
    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]
        y0, y1 = y[i - 1], y[i]
        if y0 == 0.0 or y1 == 0.0 or (y0 > 0.0) == (y1 > 0.0):
            # Same sign (or a zero endpoint): plain trapezoid of |y|.
            s += 0.5 * dx * (abs(y0) + abs(y1))
        else:
            # Sign change: split at the zero-crossing into two triangles.
            t = y0 / (y0 - y1)  # fraction of dx where y == 0
            s += 0.5 * dx * (t * abs(y0) + (1.0 - t) * abs(y1))
    return s


def _curve_area(
    x_axis: list[float],
    y: list[float],
    *,
    above: bool = False,
    ceiling: Optional[float] = None,
) -> Optional[float]:
    """Trapezoidal integral; ``above=True`` integrates ``ceiling - y``."""
    if not y or len(x_axis) < 2:
        return None
    n = min(len(x_axis), len(y))
    xs = [float(v) for v in x_axis[:n]]
    ys = [float(v) for v in y[:n]]
    # Drop any leading/trailing NaNs.
    finite = [(xv, yv) for xv, yv in zip(xs, ys) if yv == yv]
    if len(finite) < 2:
        return None
    xs = [p[0] for p in finite]
    ys = [p[1] for p in finite]
    if above:
        if ceiling is None:
            return None
        ys = [ceiling - v for v in ys]
    return _trapz(xs, ys)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _ordered_metrics() -> list[MetricSpec]:
    return list(METRIC_REGISTRY.values())


def _short_metric_label(spec: MetricSpec) -> str:
    short = {
        "cumulative_positives": "#Positives (area)",
        "feasible_context_fill_distance": "FCFD (area)",
        "feasible_convex_hull_volume": "FCHV (area)",
        "epsilon_archive_size": "ε-Archive Size (area)",
    }
    return short.get(spec.key, spec.label)


# CurveCache maps run_name -> metric_key -> curve (list[float] | None)
CurveCache = dict[str, dict[str, Optional[list[float]]]]


def _seed_area(
    run: RunSeries,
    spec: MetricSpec,
    axis: str,
    y: Optional[list[float]],
) -> Optional[float]:
    """Compute the per-seed area for one run/metric on the chosen x-axis."""
    if y is None:
        return None
    x_axis = run.x_evals if axis == "evals" else run.x_steps
    return _curve_area(x_axis, y, above=False)


_EPS_FLOOR = 1e-3  # minimum value for FCFD in product to avoid division by zero


def _compute_seed_product_curve(
    run: RunSeries,
    axis: str,
    run_curves: dict[str, Optional[list[float]]],
    metrics: list[MetricSpec],
) -> Optional[list[float]]:
    """Point-wise product of all metric values at each iteration for one seed.

    Higher-is-better metrics contribute directly; lower-is-better (FCFD)
    contributes as 1/value so the product grows when all metrics improve.
    Context-only metrics absent from this run (FCFD on non-context problems)
    are skipped. Returns None when no metric has data.
    """
    x_axis = run.x_evals if axis == "evals" else run.x_steps
    valid: list[tuple[bool, list[float]]] = []
    for spec in metrics:
        y = run_curves.get(spec.key)
        if y is None:
            if spec.needs_context:
                continue
            return None
        n = min(len(x_axis), len(y))
        valid.append((spec.higher_is_better, [float(v) for v in y[:n]]))
    if not valid:
        return None
    n = min(len(ys) for _, ys in valid)
    out: list[float] = []
    for i in range(n):
        p = 1.0
        for higher_is_better, ys in valid:
            v = ys[i]
            if v != v:  # NaN propagates
                p = float("nan")
                break
            p *= v if higher_is_better else 1.0 / max(abs(v), _EPS_FLOOR)
        out.append(p)
    return out


def _min_med_max(
    curves: list[list[float]],
) -> tuple[list[float], list[float], list[float]]:
    """Elementwise min/median/max across curves from the same method's seeds.

    Truncating to the shortest curve here is safe because all ``curves``
    passed in share one method's own ``x_evals``/``x_steps`` (its seeds all
    log on the same iteration schedule) -- unlike the cross-method product/
    rank columns, which must never share this kind of positional truncation
    across *different* methods (see ``_per_method_product_curves`` and
    ``_rank_curves_on_union_grid`` below).
    """
    n = min(len(c) for c in curves)
    trimmed = [c[:n] for c in curves]
    lo, med, hi = [], [], []
    for col in zip(*trimmed):
        finite = [v for v in col if v == v]
        if not finite:
            lo.append(float("nan"))
            med.append(float("nan"))
            hi.append(float("nan"))
        else:
            lo.append(min(finite))
            med.append(median(finite))
            hi.append(max(finite))
    return lo, med, hi


def _per_method_product_curves(
    method_runs: dict[str, list["RunSeries"]],
    methods: Iterable[str],
    axis: str,
    cache: "CurveCache",
    metrics_present: list[MetricSpec],
) -> tuple[
    dict[str, list[float]], dict[str, list[float]], dict[str, list[float]], dict[str, list[float]],
]:
    """Per-method product curve, each against its OWN x-values.

    Mirrors how the metric columns capture a fresh ``x_ref`` per method (see
    the per-method loop in the metric-column code) instead of sharing one
    ``x_ref``/one truncation length across every method. This matters because
    a batch method (q>1) logs one record per algorithmic *step* while a
    sequential method logs one per individual *evaluation*, so their curve
    arrays can have very different lengths even when both span the same
    total-evaluations range -- positionally truncating to the shortest array
    (as the old shared-``x_ref_prod`` code did) silently mislabels a sparse
    method's full-range curve onto a dense method's short early-range x-axis.

    Returns ``(method_x, method_med, method_lo, method_hi)``, each keyed by
    method name and holding only that method's own x-values and min/median/
    max product curve (combined across its own seeds via ``_min_med_max``,
    which is safe since seeds of the same method do share an iteration
    schedule).
    """
    method_x: dict[str, list[float]] = {}
    method_med: dict[str, list[float]] = {}
    method_lo: dict[str, list[float]] = {}
    method_hi: dict[str, list[float]] = {}
    for method in methods:
        seeded_runs = method_runs.get(method, [])
        prod_curves: list[list[float]] = []
        x_ref: Optional[list] = None
        for run in seeded_runs:
            run_curves = cache.get(run.run_name) or {}
            prod = _compute_seed_product_curve(run, axis, run_curves, metrics_present)
            if prod is None:
                continue
            xs = run.x_evals if axis == "evals" else run.x_steps
            n = min(len(xs), len(prod))
            prod_curves.append(prod[:n])
            if x_ref is None:
                x_ref = list(xs[:n])
        if not prod_curves or x_ref is None:
            continue
        lo, med, hi = _min_med_max(prod_curves)
        n = min(len(med), len(x_ref))
        method_x[method] = [float(v) for v in x_ref[:n]]
        method_med[method] = med[:n]
        method_lo[method] = lo[:n]
        method_hi[method] = hi[:n]
    return method_x, method_med, method_lo, method_hi


def _forward_fill_at(xs: list[float], ys: list[float], t: float) -> float:
    """Step-function (last-known-value) lookup of ``ys`` at ``t``.

    Returns the value ``ys[i]`` for the largest ``xs[i] <= t``, i.e. carries
    forward a method's most recently logged value, since a method's curve is
    only known (piecewise-constant) at its own logged checkpoints. Returns
    NaN if ``xs`` is empty or ``t`` is before ``xs[0]`` (no data logged yet).
    """
    if not xs:
        return float("nan")
    idx = bisect.bisect_right(xs, t) - 1
    if idx < 0:
        return float("nan")
    return ys[idx]


def _rank_curves_on_union_grid(
    method_x: dict[str, list[float]],
    method_med: dict[str, list[float]],
) -> tuple[list[float], dict[str, list[float]]]:
    """Cross-method rank of the median product curve, aligned via forward-fill.

    Methods with different numbers of logged points (e.g. a batch method with
    ``q=5`` logs one record per algorithmic *step* while a sequential method
    logs one per individual *evaluation*) cannot be compared by positional
    index, nor by truncating every method to the shortest array length (that
    silently mislabels a sparse-but-full-range method's curve onto a dense
    method's short early-range x-axis). Instead:

    1. The shared x-grid is the sorted union of every method's own x-values
       actually present for this row -- no evaluation counts are invented.
    2. Each method's product curve is treated as piecewise-constant between
       its own logged checkpoints, so its value at any grid point is
       forward-filled (last-known-value, see ``_forward_fill_at``) from its
       most recent checkpoint at or before that point; NaN before a method's
       first checkpoint (no data yet).
    3. Rank at each grid point is computed only over the methods that have a
       (non-NaN) value there, so a method with a smaller max budget simply
       stops contributing to the rank once its own run ends instead of
       distorting the comparison for the methods that keep going.

    Returns ``(grid, rank_curves)`` where ``rank_curves[method]`` holds one
    rank (1 = best, i.e. largest product value) per grid point, or NaN at
    grid points where that method has no forward-filled value.
    """
    grid = sorted({x for xs in method_x.values() for x in xs})
    rank_curves: dict[str, list[float]] = {m: [] for m in method_x}
    for t in grid:
        values: dict[str, float] = {}
        for method, xs in method_x.items():
            v = _forward_fill_at(xs, method_med[method], t)
            if v == v:  # not NaN
                values[method] = v
        ranked = sorted(values, key=lambda m: -values[m])
        rank_lookup = {m: i + 1 for i, m in enumerate(ranked)}
        for method in method_x:
            rank_curves[method].append(rank_lookup.get(method, float("nan")))
    return grid, rank_curves


def _precompute(runs: list[RunSeries]) -> CurveCache:
    """Build the metric reference once and compute every metric curve once per run.

    This is the only place where ``build_reference`` and ``compute_metric`` are
    called. ``_collect`` then becomes a cheap trapezoidal-integration pass over
    the returned cache, so callers can safely call ``_collect`` for multiple
    x-axes and multiple figures without repeating any expensive work.
    """
    metrics = _ordered_metrics()
    ref = None
    for r in runs:
        if r.thresholds.numel() > 0:
            ref = build_reference(r)
            break
    cache: CurveCache = {}
    for run in runs:
        cache[run.run_name] = {
            spec.key: compute_metric(run, spec, ref=ref) for spec in metrics
        }
    return cache


def _collect(
    runs: list[RunSeries],
    axis: str,
    *,
    curve_cache: Optional[CurveCache] = None,
) -> tuple[
    dict[str, dict[str, dict[str, list[float]]]],  # by_diff_metric_method[diff][metric_key][method] -> [areas]
    dict[str, dict[str, list[float]]],             # hyper[diff][method] -> [hypervolumes per seed]
    list[MetricSpec],                              # the metric specs actually encountered (any data)
]:
    """Return per-difficulty/metric/method seed-area lists and hypervolumes.

    Hypervolume per seed = product of that seed's areas across every metric
    for which the run produced a value. Seeds missing all metrics are skipped.

    Pass a ``curve_cache`` (from :func:`_precompute`) to skip redundant metric
    computation when calling this function multiple times for the same runs.
    If omitted a fresh cache is built (equivalent to the old behaviour).
    """
    metrics = _ordered_metrics()
    if curve_cache is None:
        curve_cache = _precompute(runs)

    by: dict[str, dict[str, dict[str, list[float]]]] = {}
    hyper: dict[str, dict[str, list[float]]] = {}
    seen_keys: set[str] = set()

    # Group by (difficulty, method, run) so we can compute per-seed
    # cross-metric products consistently.
    for run in runs:
        diff = _difficulty_of(run)
        method = run.method
        run_curves = curve_cache.get(run.run_name, {})
        per_metric_area: dict[str, Optional[float]] = {}
        for spec in metrics:
            y = run_curves.get(spec.key)
            a = _seed_area(run, spec, axis, y)
            per_metric_area[spec.key] = a
            if a is not None:
                seen_keys.add(spec.key)
                by.setdefault(diff, {}).setdefault(spec.key, {}).setdefault(method, []).append(a)

        # Build the combined ranking score (higher = better).
        # Higher-is-better metrics (positives, FCHV, ε-Archive Size) contribute
        # directly. FCFD (lower-is-better) contributes as 1/area so that a
        # smaller fill-distance increases the product.
        EPS_FLOOR = 1e-3  # minimum plausible area for FCFD to avoid division by zero
        spec_map = {s.key: s for s in metrics}
        prod = 1.0
        n_valid = 0
        for key, v in per_metric_area.items():
            if v is None:
                spec_m_check = spec_map.get(key)
                if spec_m_check is not None and spec_m_check.needs_context:
                    continue  # FCFD on a problem without context dims — genuinely N/A
                import warnings
                warnings.warn(
                    f"metric '{key}' returned None for run '{run.run_name}'; "
                    "expected a numeric value for a non-context metric",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            spec_m = spec_map.get(key)
            if spec_m is not None and not spec_m.higher_is_better:
                prod *= 1.0 / max(v, EPS_FLOOR)  # FCFD: smaller area → larger contribution
            else:
                prod *= v  # higher-is-better: larger area → larger product
            n_valid += 1
        if n_valid:
            hyper.setdefault(diff, {}).setdefault(method, []).append(prod)

    metrics_present = [s for s in metrics if s.key in seen_keys]
    return by, hyper, metrics_present


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_problem(
    runs: list[RunSeries],
    problem: str,
    axis: str,
    out_path: Path,
    *,
    curve_cache: Optional[CurveCache] = None,
) -> Optional[Path]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    by, hyper, metrics_present = _collect(runs, axis, curve_cache=curve_cache)
    if not by:
        return None

    difficulties = sorted(by.keys())
    n_rows = len(difficulties)
    n_cols = len(metrics_present) + 1  # +1 for hypervolume column

    methods = sorted({m for d in by.values() for mm in d.values() for m in mm.keys()})
    if not methods:
        return None
    method_to_y = {m: i + 1 for i, m in enumerate(methods)}

    fig_w = max(4.0 * n_cols, 12.0)
    fig_h = max(0.55 * len(methods) * n_rows + 1.2 * n_rows, 3.5)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    axis_label = "Total individual evaluations" if axis == "evals" else "Algorithmic step"

    for r_idx, diff in enumerate(difficulties):
        for c_idx, spec in enumerate(metrics_present):
            ax = axes[r_idx][c_idx]
            data = []
            positions = []
            labels = []
            for m in methods:
                vals = by.get(diff, {}).get(spec.key, {}).get(m, [])
                if vals:
                    data.append(vals)
                    positions.append(method_to_y[m])
                    labels.append(m)
            ax.tick_params(axis="x", labelsize=7)
            ax.grid(True, axis="x", alpha=0.25)
            if data:
                ax.boxplot(
                    data,
                    positions=positions,
                    vert=False,
                    widths=0.55,
                    showfliers=True,
                    patch_artist=True,
                    boxprops=dict(facecolor="#cfe2ff", edgecolor="#1f3a93"),
                    medianprops=dict(color="#b22222", linewidth=1.4),
                    whiskerprops=dict(color="#1f3a93"),
                    capprops=dict(color="#1f3a93"),
                    flierprops=dict(marker=".", markersize=3, alpha=0.6),
                )
                ax.set_xscale("log")
            else:
                ax.text(
                    0.5, 0.5, "(no data)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="grey",
                )
            # Re-apply ticks/labels AFTER boxplot (which overrides them).
            ax.set_yticks(list(method_to_y.values()))
            ax.set_yticklabels(methods, fontsize=7)
            ax.set_ylim(0.5, len(methods) + 0.5)
            ax.invert_yaxis()
            if r_idx == 0:
                ax.set_title(_short_metric_label(spec), fontsize=8)
            if r_idx == n_rows - 1:
                ax.set_xlabel("area", fontsize=8)
            if c_idx == 0:
                ax.set_ylabel(f"{diff}\nmethod", fontsize=8)
            else:
                # Free up horizontal space — y ticks repeat the method names.
                ax.tick_params(labelleft=False)

        # Hypervolume column
        ax = axes[r_idx][-1]
        data, positions, labels = [], [], []
        for m in methods:
            vals = hyper.get(diff, {}).get(m, [])
            if vals:
                data.append(vals)
                positions.append(method_to_y[m])
                labels.append(m)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(True, axis="x", alpha=0.25)
        if data:
            ax.boxplot(
                data,
                positions=positions,
                vert=False,
                widths=0.55,
                showfliers=True,
                patch_artist=True,
                boxprops=dict(facecolor="#ffe2b8", edgecolor="#8a4b00"),
                medianprops=dict(color="#a02828", linewidth=1.4),
                whiskerprops=dict(color="#8a4b00"),
                capprops=dict(color="#8a4b00"),
                flierprops=dict(marker=".", markersize=3, alpha=0.6),
            )
        else:
            ax.text(
                0.5, 0.5, "(no data)", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="grey",
            )
        ax.set_yticks(list(method_to_y.values()))
        ax.set_yticklabels(methods, fontsize=7)
        ax.set_ylim(0.5, len(methods) + 0.5)
        ax.invert_yaxis()
        if r_idx == 0:
            ax.set_title("Hypervolume\n(product of metric areas, higher is better)", fontsize=8)
        if r_idx == n_rows - 1:
            ax.set_xlabel("product of areas", fontsize=8)
        ax.tick_params(labelleft=False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Curve grid: actual metric values per iteration (not areas)
# ---------------------------------------------------------------------------
_SHORT_CURVE_LABELS: dict[str, str] = {
    "cumulative_positives": "#Positives",
    "feasible_context_fill_distance": "FCFD",
    "feasible_convex_hull_volume": "FCHV",
    "epsilon_archive_size": "ε-Archive Size",
}


def _plot_problem_curves(
    runs: list[RunSeries],
    problem: str,
    axis: str,
    out_path: Path,
    *,
    curve_cache: Optional[CurveCache] = None,
) -> Optional[Path]:
    """Grid of metric curves: rows=difficulty, cols=metrics + raw product + product rank.

    Each cell shows one line per method (seed-mean ± std shading). The second-to-last
    column is the raw point-wise product across all available metrics at each iteration
    (1/FCFD for lower-is-better), so it grows whenever any metric improves — higher is
    always better. The last column ranks methods by that product at each step (1 = best).
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if curve_cache is None:
        curve_cache = _precompute(runs)

    metrics = _ordered_metrics()

    by_diff_method: dict[str, dict[str, list[RunSeries]]] = {}
    for run in runs:
        diff = _difficulty_of(run)
        by_diff_method.setdefault(diff, {}).setdefault(run.method, []).append(run)

    if not by_diff_method:
        return None

    difficulties = sorted(by_diff_method.keys())
    methods = sorted({m for d in by_diff_method.values() for m in d})
    if not methods:
        return None

    seen_keys: set[str] = set()
    for run in runs:
        for key, y in (curve_cache.get(run.run_name) or {}).items():
            if y is not None:
                seen_keys.add(key)
    metrics_present = [s for s in metrics if s.key in seen_keys]
    if not metrics_present:
        return None

    n_rows = len(difficulties)
    n_cols = len(metrics_present) + 2  # +1 raw product, +1 product rank
    colors = plt.get_cmap("tab10").colors
    method_colors = {m: colors[i % len(colors)] for i, m in enumerate(methods)}

    fig_w = max(4.0 * n_cols, 12.0)
    fig_h = max(2.5 * n_rows + 1.0, 5.0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    axis_label = "Total individual evaluations" if axis == "evals" else "Algorithmic step"

    for r_idx, diff in enumerate(difficulties):
        method_runs = by_diff_method.get(diff, {})

        # Metric columns
        for c_idx, spec in enumerate(metrics_present):
            ax = axes[r_idx][c_idx]
            ax.grid(True, alpha=0.25)
            ax.tick_params(axis="both", labelsize=7)
            plotted = False

            for method in methods:
                seeded_runs = method_runs.get(method, [])
                curves: list[list[float]] = []
                x_ref: Optional[list] = None
                for run in seeded_runs:
                    y = (curve_cache.get(run.run_name) or {}).get(spec.key)
                    if y is None:
                        continue
                    xs = run.x_evals if axis == "evals" else run.x_steps
                    n = min(len(xs), len(y))
                    curves.append([float(v) for v in y[:n]])
                    if x_ref is None:
                        x_ref = list(xs[:n])
                if not curves or x_ref is None:
                    continue
                lo, med, hi = _min_med_max(curves)
                n = min(len(med), len(x_ref))
                x_plot = x_ref[:n]
                color = method_colors[method]
                ax.plot(x_plot, med[:n], color=color, linewidth=1.5, label=method)
                if len(curves) > 1:
                    ax.fill_between(x_plot, lo[:n], hi[:n], color=color, alpha=0.15)
                plotted = True

            if not plotted:
                ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="grey")
            if r_idx == 0:
                ax.set_title(_SHORT_CURVE_LABELS.get(spec.key, spec.label), fontsize=8)
            if r_idx == n_rows - 1:
                ax.set_xlabel(axis_label, fontsize=8)
            if c_idx == 0:
                ax.set_ylabel(f"{diff}", fontsize=8)

        # Pre-compute each method's own product curve (own x-values -- see
        # _per_method_product_curves; no cross-method truncation/alignment).
        method_x, method_med, method_lo, method_hi = _per_method_product_curves(
            method_runs, methods, axis, curve_cache, metrics_present,
        )

        # Raw product column (col N+1) — each method plotted against its OWN
        # x-values, exactly like the metric columns above; this column needs
        # no cross-method alignment since it's N independent lines.
        ax = axes[r_idx][-2]
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=7)
        plotted = False
        for method in sorted(method_med.keys()):
            color = method_colors[method]
            x_plot = method_x[method]
            ax.plot(x_plot, method_med[method], color=color,
                    linewidth=1.5, label=method)
            if len(method_runs.get(method, [])) > 1:
                ax.fill_between(x_plot, method_lo[method], method_hi[method],
                                color=color, alpha=0.15)
            plotted = True
        if not plotted:
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="grey")
        if r_idx == 0:
            ax.set_title("Product\n(raw, higher is better)", fontsize=8)
        if r_idx == n_rows - 1:
            ax.set_xlabel(axis_label, fontsize=8)

        # Rank column (col N+2) — rank of the median product at each step
        # (1 = best), aligned across methods on the union-of-x-values grid
        # via forward-fill (see _rank_curves_on_union_grid).
        ax = axes[r_idx][-1]
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="both", labelsize=7)
        plotted = False
        if method_med:
            grid, rank_curves = _rank_curves_on_union_grid(method_x, method_med)
            for method in sorted(method_med.keys()):
                ax.plot(grid, rank_curves[method], color=method_colors[method],
                        linewidth=1.5, label=method)
                plotted = True
            n_ranked = len(method_med)
            ax.set_ylim(n_ranked + 0.5, 0.5)  # rank 1 at top
            ax.set_yticks(list(range(1, n_ranked + 1)))
        if not plotted:
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="grey")
        if r_idx == 0:
            ax.set_title("Product rank\n(1 = best)", fontsize=8)
        if r_idx == n_rows - 1:
            ax.set_xlabel(axis_label, fontsize=8)

    # Shared legend drawn once below all subplots
    handles_seen: dict[str, object] = {}
    for row in axes:
        for a in row:
            for h, lbl in zip(*a.get_legend_handles_labels()):
                handles_seen.setdefault(lbl, h)
    if handles_seen:
        fig.legend(
            list(handles_seen.values()), list(handles_seen.keys()),
            loc="lower center", ncol=min(len(methods), 6),
            fontsize=8, bbox_to_anchor=(0.5, 0.0),
        )
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    else:
        fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Combined hypervolume figure (all problems)
# ---------------------------------------------------------------------------
def _plot_hypervolume_from_data(
    hv_by_problem: dict[str, dict[str, dict[str, list[float]]]],
    axis: str,
    out_path: Path,
) -> Optional[Path]:
    """One figure with rows=difficulties, cols=problems, cells=HV boxplots.

    Each cell mirrors the hypervolume column produced by :func:`_plot_problem`
    but for every problem at once, making cross-problem comparisons easy.

    ``hv_by_problem`` maps ``{problem: {diff: {method: [hv_per_seed]}}}``.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    all_diffs: list[str] = sorted({d for hv in hv_by_problem.values() for d in hv})
    all_methods: list[str] = sorted(
        {m for hv in hv_by_problem.values() for d in hv.values() for m in d}
    )
    problems = sorted(hv_by_problem.keys())

    if not all_diffs or not all_methods or not problems:
        return None

    method_to_y = {m: i + 1 for i, m in enumerate(all_methods)}
    n_rows = len(all_diffs)
    n_cols = len(problems)

    fig_w = max(3.5 * n_cols, 8.0)
    fig_h = max(0.55 * len(all_methods) * n_rows + 1.2 * n_rows, 4.0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    axis_label = "Total individual evaluations" if axis == "evals" else "Algorithmic step"

    for c_idx, problem in enumerate(problems):
        hyper = hv_by_problem.get(problem, {})
        for r_idx, diff in enumerate(all_diffs):
            ax = axes[r_idx][c_idx]
            data, positions = [], []
            for m in all_methods:
                vals = hyper.get(diff, {}).get(m, [])
                if vals:
                    data.append(vals)
                    positions.append(method_to_y[m])
            ax.tick_params(axis="x", labelsize=7)
            ax.grid(True, axis="x", alpha=0.25)
            if data:
                ax.boxplot(
                    data,
                    positions=positions,
                    vert=False,
                    widths=0.55,
                    showfliers=True,
                    patch_artist=True,
                    boxprops=dict(facecolor="#ffe2b8", edgecolor="#8a4b00"),
                    medianprops=dict(color="#a02828", linewidth=1.4),
                    whiskerprops=dict(color="#8a4b00"),
                    capprops=dict(color="#8a4b00"),
                    flierprops=dict(marker=".", markersize=3, alpha=0.6),
                )
            else:
                ax.text(
                    0.5, 0.5, "(no data)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="grey",
                )
            ax.set_yticks(list(method_to_y.values()))
            ax.set_yticklabels(all_methods, fontsize=7)
            ax.set_ylim(0.5, len(all_methods) + 0.5)
            ax.invert_yaxis()
            if r_idx == 0:
                ax.set_title(problem, fontsize=9)
            if r_idx == n_rows - 1:
                ax.set_xlabel("product of areas", fontsize=8)
            if c_idx == 0:
                ax.set_ylabel(f"{diff}\nmethod", fontsize=8)
            else:
                ax.tick_params(labelleft=False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_hypervolume_combined(
    runs_by_problem: dict[str, list[RunSeries]],
    axis: str,
    out_path: Path,
    *,
    caches_by_problem: Optional[dict[str, CurveCache]] = None,
) -> Optional[Path]:
    hv_data: dict[str, dict[str, dict[str, list[float]]]] = {}
    for problem, runs in runs_by_problem.items():
        cache = (caches_by_problem or {}).get(problem)
        _, hyper, _ = _collect(runs, axis, curve_cache=cache)
        hv_data[problem] = hyper
    return _plot_hypervolume_from_data(hv_data, axis, out_path)


# ---------------------------------------------------------------------------
# Intermediate metrics persistence (used by per-problem Slurm jobs)
# ---------------------------------------------------------------------------
def save_problem_metrics(
    metrics_dir: Path,
    problem: str,
    hyper_by_axis: dict[str, dict[str, dict[str, list[float]]]],
) -> Path:
    """Write per-problem hypervolume data to a JSON file for later aggregation.

    ``hyper_by_axis`` maps ``{axis: {diff: {method: [hv_per_seed]}}}``.
    """
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / f"{problem}_metrics.json"
    with out_path.open("w") as f:
        json.dump({"problem": problem, **hyper_by_axis}, f)
    return out_path


def load_problem_metrics(
    path: Path,
) -> tuple[str, dict[str, dict[str, dict[str, list[float]]]]]:
    """Load per-problem metrics written by :func:`save_problem_metrics`.

    Returns ``(problem_name, {axis: {diff: {method: [hv_per_seed]}}})``.
    """
    with path.open() as f:
        data = json.load(f)
    problem = data.pop("problem")
    return problem, data


# ---------------------------------------------------------------------------
# Stats helper: reshape hypervolume data for the stats pipeline
# ---------------------------------------------------------------------------
def _collect_hv_for_stats(
    runs_by_problem: dict[str, list[RunSeries]],
    caches_by_problem: dict[str, "CurveCache"],
) -> dict[str, dict[str, dict[str, dict[str, list[float]]]]]:
    """Return ``{problem: {axis: {difficulty: {method: [hv_seed_0, …]}}}}`.

    Hypervolumes are computed by :func:`_collect` and gathered here into the
    nested structure expected by :func:`itcas.reporting.stats.run_stats`.
    Per-seed ordering is position-based (index = seed rank); callers must
    ensure that runs within a problem are sorted consistently (they are, since
    :func:`_discover_runs` sorts by seed).
    """
    out: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}
    for problem, runs in runs_by_problem.items():
        cache = caches_by_problem.get(problem)
        out[problem] = {}
        for axis in ("evals", "steps"):
            _, hyper, _ = _collect(runs, axis, curve_cache=cache)
            # hyper: {diff: {method: [per-seed hv]}}
            out[problem][axis] = {d: dict(m) for d, m in hyper.items()}
    return out


# ---------------------------------------------------------------------------
# Synthetic-function comparison: ITCAS (batch) vs 5 baselines, one folder per
# difficulty level.
#
# This is the "headline" comparison for the paper's synthetic suite: the
# proposed method (``itcas_ndig`` -- full ITCAS, QD-DPP greedy batch
# selection, quality=``ndig``) against Random, the two Family-C
# "LSE-then-sample" baselines at a 10% Stage-1 split
# (``straddle_then_sample_lse10``, ``bes_then_sample_lse10``, both forced
# sequential), and the two CAS-family sequential acquisitions (``cas_eci``,
# ``moc_cas_hard``). All five baselines are forced-sequential (one record per
# individual evaluation); only ``itcas_ndig`` is batch, so -- exactly as in
# ``ndig_comparison``/``ff_comparison`` -- **total individual evaluations**
# (``RunSeries.x_evals``) is the only fair shared x-axis; there is no steps
# variant here.
#
# "Synthetic" means every problem in ``configs/final_problems.json`` except
# the real-world problems (``spacecraft_formation_flying_a1``, backed by a
# real Basilisk simulation; ``casd_llm``, backed by a live LLM + judge-model
# server -- see ``_REAL_WORLD_PROBLEMS`` below and each's dedicated report,
# ``ff_comparison.py`` / ``casd_comparison.py``) rather than a closed-form
# objective. Every difficulty
# level present on disk gets its own output subfolder (rows = problems,
# columns = metric curves + raw product + product-rank, plus a bottom
# average-rank row and a Friedman/Wilcoxon stats report) so a reader can open
# exactly one difficulty's results without wading through the rest.
#
# The grid rendering (``plot_group_grid``) and per-row helpers
# (``_collect_family_runs``, ``_difficulties_present``, ``_rows_by_problem``,
# ``_collect_product_auc_for_stats``) already live in ``batch_vs_sequential``;
# ``average_ranks_over_rows`` lives in ``ranking``. Both of those modules
# import from *this* one at their own top level, so importing them back here
# at module scope would be circular -- the imports below are deferred inside
# the function body instead, which is safe because by the time it runs, every
# module involved has already finished loading.
# ---------------------------------------------------------------------------
_DEFAULT_PROBLEMS_CONFIG = "configs/final_problems.json"
# Real-world problems, excluded from "synthetic" -- keep in sync with
# batch_vs_sequential._REAL_WORLD_PROBLEMS (not imported directly here to
# avoid a circular import; see the module-level section docstring above).
_REAL_WORLD_PROBLEMS: tuple[str, ...] = ("spacecraft_formation_flying_a1", "casd_llm")
_SYNTHETIC_AXIS = "evals"  # evaluations only -- see section docstring above
_SYNTHETIC_OUTPUT_DIR = "results/synthetic_comparison"

SYNTHETIC_PROPOSED_METHOD = "itcas_ndig"
SYNTHETIC_BASELINE_METHODS: tuple[str, ...] = (
    "random",
    "straddle_then_sample_lse10",
    "bes_then_sample_lse10",
    "cas_eci",
    "moc_cas_hard",
)
SYNTHETIC_METHODS: tuple[str, ...] = (SYNTHETIC_PROPOSED_METHOD,) + SYNTHETIC_BASELINE_METHODS

# Stable hue per method (tab10), all solid -- six distinct algorithms, no
# sequential/batch pair sharing one family here. itcas_ndig (proposed) gets
# blue, matching the "proposed = blue" convention used in
# ndig_comparison.py/ff_comparison.py; random (the trivial baseline) gets grey.
_SYNTHETIC_METHOD_STYLES: dict[str, dict] = {
    SYNTHETIC_PROPOSED_METHOD: {"color": "#1f77b4", "linestyle": "-"},
    "random": {"color": "#7f7f7f", "linestyle": "-"},
    "straddle_then_sample_lse10": {"color": "#2ca02c", "linestyle": "-"},
    "bes_then_sample_lse10": {"color": "#ff7f0e", "linestyle": "-"},
    "cas_eci": {"color": "#9467bd", "linestyle": "-"},
    "moc_cas_hard": {"color": "#d62728", "linestyle": "-"},
}


def _synthetic_problems(problems_config: str | Path = _DEFAULT_PROBLEMS_CONFIG) -> list[str]:
    """Every problem in ``problems_config`` except the real-world problems."""
    with Path(problems_config).open() as f:
        problems = json.load(f)["problems"]
    return [p for p in problems if p not in _REAL_WORLD_PROBLEMS]


def _synthetic_report_to_markdown(report: StatsReport, difficulty: str) -> str:
    """Custom renderer: higher product-curve AUC is better, rows = problems.

    Mirrors ``ff_comparison._report_to_markdown``/``ndig_comparison._report_to_markdown``
    (both of which test the opposite -- higher-is-better -- direction from
    ``stats.report_to_markdown``'s hardcoded prose), generalized to this
    report's 5 baseline columns and to one difficulty level's worth of groups
    (one row per problem) rather than one row per difficulty.
    """
    lines: list[str] = []
    lines.append(f"# Synthetic comparison — {SYNTHETIC_PROPOSED_METHOD} vs 5 baselines ({difficulty})")
    lines.append("")
    lines.append(
        f"**Proposed:** `{report.proposed_method}` &nbsp;|&nbsp; "
        f"**Baselines:** {', '.join(f'`{b}`' for b in SYNTHETIC_BASELINE_METHODS)} "
        f"&nbsp;|&nbsp; **α =** {report.alpha}"
    )
    lines.append("")
    lines.append(
        "Per problem: a **Friedman omnibus test** over all six methods gates a "
        f"**one-sided paired Wilcoxon signed-rank test** (`H1: {SYNTHETIC_PROPOSED_METHOD} > "
        "baseline`) against each of the five baselines individually, Holm-Bonferroni corrected "
        "over those five baselines. The per-seed scalar is the **area under the point-wise "
        "product curve** (`summary._compute_seed_product_curve` integrated via "
        "`summary._curve_area`) on the total-individual-evaluations axis -- higher is better "
        "(higher-is-better metrics multiply directly into the product; FCFD, the one "
        "lower-is-better metric, contributes as a reciprocal; see `contexts/metrics.md`)."
    )
    lines.append("")
    lines.append(
        "Significance markers: `***` p_adj < 0.001, `**` p_adj < 0.01, `*` p_adj < 0.05, `ns` not "
        "significant. If the Friedman omnibus test does not reach significance, pairwise tests "
        "are skipped for that problem (noted below)."
    )
    lines.append("")

    baseline_headers = " | ".join(f"vs `{b}`" for b in SYNTHETIC_BASELINE_METHODS)
    baseline_sep = "".join(":--------------------:|" for _ in SYNTHETIC_BASELINE_METHODS)
    lines.append(f"| Problem | Seeds | Friedman p | Friedman sig | {baseline_headers} |")
    lines.append(f"|:--------|------:|-----------:|:------------:|{baseline_sep}")

    for g in sorted(report.groups, key=lambda x: x.problem):
        friedman_sig = "Yes" if g.friedman_significant else "No"
        if g.note or not g.pairwise:
            note = g.note or "no pairwise result"
            blanks = " | ".join(f"_{note}_" for _ in SYNTHETIC_BASELINE_METHODS)
            lines.append(
                f"| `{g.problem}` | {g.n_seeds} | {_fmt(g.friedman_p)} "
                f"| {friedman_sig} | {blanks} |"
            )
            continue
        by_baseline = {pw.baseline: pw for pw in g.pairwise}
        cells = []
        for baseline in SYNTHETIC_BASELINE_METHODS:
            pw = by_baseline.get(baseline)
            if pw is None:
                cells.append("—")
                continue
            sig_str = _sig_marker(pw.significant, pw.p_adj)
            cells.append(f"p_adj={_fmt(pw.p_adj)} {sig_str} (Δ={_fmt(pw.effect_median_diff)})")
        cells_str = " | ".join(cells)
        lines.append(
            f"| `{g.problem}` | {g.n_seeds} | {_fmt(g.friedman_p)} "
            f"| {friedman_sig} | {cells_str} |"
        )
    lines.append("")

    n_tested = sum(1 for g in report.groups if g.pairwise)
    n_total = len(report.groups)

    def _count_sig(baseline: str) -> int:
        return sum(
            1
            for g in report.groups
            for pw in g.pairwise
            if pw.baseline == baseline and pw.significant
        )

    per_baseline_summary = ", ".join(
        f"`{b}` in **{_count_sig(b)} / {n_tested}**" for b in SYNTHETIC_BASELINE_METHODS
    )
    lines.append(
        f"**Summary:** {n_tested} / {n_total} synthetic problems had a significant Friedman "
        f"omnibus test (α={report.alpha}). Among those, `{SYNTHETIC_PROPOSED_METHOD}` "
        f"significantly outperforms {per_baseline_summary} problems."
    )
    lines.append("")
    return "\n".join(lines)


def _metrics_present_in_rows(rows: list) -> list[MetricSpec]:
    """Every :class:`MetricSpec` with at least one non-``None`` curve across ``rows``.

    ``rows`` is the ``(row_label, runs, cache)`` triple used throughout this
    section (and in ``batch_vs_sequential``/``ranking``) -- kept in
    :func:`_ordered_metrics`' registry order, matching every other grid in
    this package.
    """
    seen_keys: set[str] = set()
    for _, _, cache in rows:
        for run_curves in cache.values():
            for key, y in run_curves.items():
                if y is not None:
                    seen_keys.add(key)
    return [s for s in _ordered_metrics() if s.key in seen_keys]


def _draw_metric_bar_panel(
    ax,
    data: Optional[dict[str, float]],
    methods: list[str],
    method_styles: dict[str, dict],
    method_labels: Optional[dict[str, str]],
    *,
    ascending_is_better: bool,
    reference_line: Optional[float] = None,
) -> None:
    """Horizontal bar chart of one column's per-method scalar (rank or ratio).

    ``ascending_is_better=True`` sorts smallest-first (e.g. rank, 1 = best);
    ``False`` sorts largest-first (e.g. a higher-is-better relative-AUC
    ratio). ``reference_line``, if given, draws a dashed vertical guide (e.g.
    the "1.0 = best" mark for the relative-AUC figure).
    """
    ax.tick_params(axis="both", labelsize=7)
    if not data:
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="grey")
        return
    present = [m for m in methods if m in data]
    present_sorted = sorted(present, key=lambda m: data[m], reverse=not ascending_is_better)
    y_pos = list(range(len(present_sorted)))
    colors = [method_styles[m]["color"] for m in present_sorted]
    values = [data[m] for m in present_sorted]
    labels = [
        (method_labels.get(m, m) if method_labels else m) for m in present_sorted
    ]
    ax.barh(y_pos, values, color=colors)
    if reference_line is not None:
        ax.axvline(reference_line, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()


def _plot_avg_rank_figure(
    avg_rank_row: dict[str, dict[str, float]],
    methods: list[str],
    method_styles: dict[str, dict],
    metrics_present: list[MetricSpec],
    n_rows: int,
    out_path: str | Path,
    method_labels: Optional[dict[str, str]] = None,
) -> Optional[Path]:
    """Standalone bar-chart figure: average rank per metric (+ product), 1 = best.

    Renders what used to be the bottom row of the combined grid (see
    ``batch_vs_sequential.plot_group_grid``'s ``avg_rank_row`` parameter) as
    its own one-row figure, since problems are now split one-per-PDF (see
    :func:`summarize_synthetic_comparison`) and there is no longer a shared
    grid for this row to sit under.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if not metrics_present:
        return None

    n_cols = len(metrics_present) + 1  # + product
    fig_w = max(3.0 * n_cols, 10.0)
    fig_h = max(0.4 * len(methods) + 1.5, 3.0)
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    axes = axes[0]

    for c_idx, spec in enumerate(metrics_present):
        ax = axes[c_idx]
        _draw_metric_bar_panel(
            ax, avg_rank_row.get(spec.key), methods, method_styles, method_labels,
            ascending_is_better=True,
        )
        ax.set_title(f"{_SHORT_CURVE_LABELS.get(spec.key, spec.label)}", fontsize=8)
        if c_idx == 0:
            ax.set_ylabel(f"Avg rank across {n_rows} problems\n(best to worst)", fontsize=8)

    _draw_metric_bar_panel(
        axes[-1], avg_rank_row.get("product"), methods, method_styles, method_labels,
        ascending_is_better=True,
    )
    axes[-1].set_title("Product\n(raw)", fontsize=8)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_relative_auc_figure(
    relative_auc_row: dict[str, dict[str, float]],
    methods: list[str],
    method_styles: dict[str, dict],
    metrics_present: list[MetricSpec],
    n_rows: int,
    out_path: str | Path,
    method_labels: Optional[dict[str, str]] = None,
) -> Optional[Path]:
    """Standalone bar-chart figure: average relative-AUC ratio per metric (+ product).

    See :func:`itcas.reporting.ranking.relative_auc_ratios_over_rows` for the
    exact definition: per problem, every (method, seed) AUC is divided by
    that problem's best AUC (over every method/seed, regardless of which),
    averaged over seeds then over problems. A ratio of 1.0 (dashed guide
    line) means "matched the best seed-level AUC seen anywhere for that
    problem"; a metric's *worse* direction is below 1.0 for a higher-is-better
    metric (larger AUC is better) and above 1.0 for a lower-is-better one
    (FCFD, smaller AUC is better) -- ``ascending_is_better`` is set per column
    from ``spec.higher_is_better`` so each panel always sorts "closest to the
    1.0 guide line" at the top, regardless of that column's direction.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if not metrics_present:
        return None

    n_cols = len(metrics_present) + 1  # + product
    fig_w = max(3.0 * n_cols, 10.0)
    fig_h = max(0.4 * len(methods) + 1.5, 3.0)
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    axes = axes[0]

    for c_idx, spec in enumerate(metrics_present):
        ax = axes[c_idx]
        _draw_metric_bar_panel(
            ax, relative_auc_row.get(spec.key), methods, method_styles, method_labels,
            ascending_is_better=not spec.higher_is_better, reference_line=1.0,
        )
        arrow = "↑" if spec.higher_is_better else "↓"
        ax.set_title(f"{_SHORT_CURVE_LABELS.get(spec.key, spec.label)} {arrow}", fontsize=8)
        if c_idx == 0:
            ax.set_ylabel(
                f"Avg relative AUC across {n_rows} problems\n(1.0 = best)", fontsize=8
            )

    _draw_metric_bar_panel(
        axes[-1], relative_auc_row.get("product"), methods, method_styles, method_labels,
        ascending_is_better=False, reference_line=1.0,
    )
    axes[-1].set_title("Product ↑\n(raw)", fontsize=8)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def _draw_metric_line_panel(
    ax,
    relative_auc_by_level: list[tuple[str, dict[str, dict[str, float]]]],
    column_key: str,
    methods: list[str],
    method_styles: dict[str, dict],
    method_labels: Optional[dict[str, str]],
    *,
    higher_is_better: bool,
) -> None:
    """One panel of :func:`_plot_relative_auc_by_difficulty_figure`: one line per method.

    ``relative_auc_by_level`` is ``[(level_label, relative_auc_row), ...]``
    already in x-axis order (see that function's docstring). Plots evenly
    spaced categorical x positions (``range(n)``, never the real difficulty
    value) against each method's relative-AUC ratio for ``column_key`` at
    that level, using ``float("nan")`` for a (method, level) combination with
    no data so the line breaks there instead of raising or silently skipping
    the method. Draws a dashed ``y=1.0`` reference line (mirrors
    ``_draw_metric_bar_panel``'s ``reference_line=1.0``); ``higher_is_better``
    is accepted for signature symmetry with the bar-panel helper but doesn't
    otherwise affect this panel's rendering (a line plot has no "sort
    direction" the way a bar chart does).
    """
    ax.tick_params(axis="both", labelsize=7)
    n = len(relative_auc_by_level)
    x = list(range(n))
    any_data = False
    for method in methods:
        ys = []
        for _level_label, row in relative_auc_by_level:
            val = (row.get(column_key) or {}).get(method)
            ys.append(float(val) if val is not None else float("nan"))
        if all(v != v for v in ys):  # all NaN -- no data anywhere for this method
            continue
        any_data = True
        style = method_styles.get(method, {})
        label = method_labels.get(method, method) if method_labels else method
        ax.plot(
            x, ys,
            color=style.get("color"), linestyle=style.get("linestyle", "-"),
            marker="o", markersize=3, linewidth=1.2, label=label,
        )
    if not any_data:
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="grey")
        return
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_xticks(x)
    ax.set_xticklabels([lvl for lvl, _ in relative_auc_by_level], fontsize=7)


def _plot_relative_auc_by_difficulty_figure(
    relative_auc_by_level: list[tuple[str, dict[str, dict[str, float]]]],
    methods: list[str],
    method_styles: dict[str, dict],
    metrics_present: list[MetricSpec],
    out_path: str | Path,
    method_labels: Optional[dict[str, str]] = None,
) -> Optional[Path]:
    """Standalone line-plot figure: relative-AUC ratio per metric (+ product), across difficulty.

    A sibling of :func:`_plot_relative_auc_figure` that shows the same
    per-(row, method, column) relative-AUC ratios (see
    :func:`itcas.reporting.ranking.relative_auc_ratios_over_rows`) *without*
    averaging away the per-difficulty breakdown: :func:`_plot_relative_auc_figure`
    collapses every row (difficulty level) into one number per method per
    column, whereas this figure draws one line per method per column, plotted
    across difficulty levels on the x-axis, so a method's trend as the
    problem gets harder/easier stays visible.

    ``relative_auc_by_level`` is ``[(level_label, relative_auc_row), ...]``,
    already in the desired x-axis order, where each ``relative_auc_row`` is
    one call to ``ranking.relative_auc_ratios_over_rows`` restricted to just
    that level's own row(s) (i.e. *not* averaged across levels -- callers are
    responsible for computing each level's ratios independently; see
    ``ff_comparison``/``casd_comparison``'s single-row calls, or the
    synthetic pipeline's already-per-difficulty ``_combine_synthetic_summaries``
    output). ``level_label`` is used verbatim as that level's x-tick label
    and should already be short (unlike the verbose multi-line row labels
    used elsewhere in this package).

    Same panel layout as :func:`_plot_relative_auc_figure` (one column per
    metric in ``metrics_present`` plus a trailing Product column, each titled
    with a ``↑``/``↓`` direction arrow) and the same per-method
    color/linestyle from ``method_styles``, but a single legend shared across
    every panel (built from the first panel's line handles) instead of
    per-panel y-tick method labels, since every panel here shares the same
    x-axis and set of methods.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if not metrics_present:
        return None

    n_cols = len(metrics_present) + 1  # + product
    fig_w = max(3.0 * n_cols, 10.0)
    fig_h = 3.5
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    axes = axes[0]

    for c_idx, spec in enumerate(metrics_present):
        ax = axes[c_idx]
        _draw_metric_line_panel(
            ax, relative_auc_by_level, spec.key, methods, method_styles, method_labels,
            higher_is_better=spec.higher_is_better,
        )
        arrow = "↑" if spec.higher_is_better else "↓"
        ax.set_title(f"{_SHORT_CURVE_LABELS.get(spec.key, spec.label)} {arrow}", fontsize=8)
        if c_idx == 0:
            ax.set_ylabel("Relative AUC by difficulty\n(1.0 = best)", fontsize=8)

    _draw_metric_line_panel(
        axes[-1], relative_auc_by_level, "product", methods, method_styles, method_labels,
        higher_is_better=True,
    )
    axes[-1].set_title("Product ↑\n(raw)", fontsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="lower center", ncol=min(len(labels), 6),
            fontsize=7, bbox_to_anchor=(0.5, -0.05),
        )

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Synthetic comparison: per-problem summary + cross-problem combination.
#
# The two halves below (:func:`_synthetic_problem_report` and
# :func:`_combine_synthetic_summaries`) are the shared core of a two-stage,
# memory-bounded pipeline (mirroring ``summarize_benchmark`` +
# ``summarize_from_metrics``, see ``itcas/reporting/tune_eps_archive.py``'s
# module docstring for the anti-pattern both avoid): a per-problem stage that
# only ever needs *one* problem's ``RunSeries``/``CurveCache`` in memory at a
# time, and an aggregate stage that only needs the small serializable
# summaries the per-problem stage produced -- never the raw runs of more than
# one problem at once.
#
# This decomposition is exact, not approximate, because both
# ``ranking.average_ranks_over_rows`` and
# ``ranking.relative_auc_ratios_over_rows`` are themselves already a
# per-row computation followed by a trivial ``sum(rs) / len(rs)`` reduction
# across rows (see their own docstrings) -- and here each "row" is exactly
# one problem's slice at one difficulty (:func:`batch_vs_sequential._rows_by_problem`
# with a single-element ``problems`` list always returns at most one row).
# So computing a rank/ratio for a single-problem row and only *combining* the
# results later (:func:`_combine_synthetic_summaries`) reproduces bit-for-bit
# what calling those functions once over every problem's row together would
# produce. The monolithic :func:`summarize_synthetic_comparison` below now
# uses this exact same pair of helpers (just without ever serializing to
# disk between the two stages), so the two pipelines cannot silently drift
# apart.
# ---------------------------------------------------------------------------
def _synthetic_problem_report(
    problem: str,
    runs: list[RunSeries],
    cache: CurveCache,
    *,
    out_dir: str | Path | None = None,
) -> tuple[dict[str, dict], list[str]]:
    """One problem's contribution to the synthetic comparison, per difficulty.

    Uses only ``runs``/``cache`` for this single ``problem`` -- never touches
    any other problem's data -- so this is safe to call from a per-problem
    job holding just one problem's ``RunSeries`` in memory.

    When ``out_dir`` is given, also renders this problem's own per-difficulty
    PDF (``<out_dir>/<difficulty>/synthetic_comparison_<problem>_vs_evaluations.pdf``),
    byte-for-byte the same file :func:`summarize_synthetic_comparison` used to
    render inline for this problem's row.

    Returns ``(summary_by_diff, paths_written)`` where ``summary_by_diff`` is
    ``{difficulty: {"metrics_present": [metric_key, ...], "avg_rank":
    {column_key: {method: rank}}, "relative_auc": {column_key: {method:
    ratio}}, "auc": {method: [seed_aucs]}}}`` -- fully JSON-serializable (see
    :func:`save_synthetic_problem_metrics`) and exactly what
    :func:`_combine_synthetic_summaries` expects as one entry of its
    ``summaries_by_problem`` argument.
    """
    from .batch_vs_sequential import _collect_product_auc_for_stats, _rows_by_problem, plot_group_grid
    from .method_labels import METHOD_ABBREVIATIONS
    from .ranking import average_ranks_over_rows, relative_auc_ratios_over_rows

    runs_by_problem = {problem: runs}
    caches_by_problem = {problem: cache}
    auc_by_diff = _collect_product_auc_for_stats(runs_by_problem, caches_by_problem).get(problem, {})

    diffs = sorted({_difficulty_of(r) for r in runs})
    summary: dict[str, dict] = {}
    paths: list[str] = []
    for diff in diffs:
        rows = _rows_by_problem([problem], runs_by_problem, caches_by_problem, diff)
        if not rows:
            continue
        problem_label, row_runs, row_cache = rows[0]

        if out_dir is not None:
            problem_path = (
                Path(out_dir) / diff / f"synthetic_comparison_{problem_label}_vs_evaluations.pdf"
            )
            ok = plot_group_grid(
                list(SYNTHETIC_METHODS), _SYNTHETIC_METHOD_STYLES,
                [(problem_label, row_runs, row_cache)], None, problem_path,
                include_product_rank_column=False, method_labels=METHOD_ABBREVIATIONS,
            )
            if ok is not None:
                paths.append(str(ok))

        metrics_present = _metrics_present_in_rows(rows)
        avg_rank = average_ranks_over_rows(rows, list(SYNTHETIC_METHODS), _SYNTHETIC_AXIS)
        relative_auc = relative_auc_ratios_over_rows(rows, list(SYNTHETIC_METHODS), _SYNTHETIC_AXIS)
        summary[diff] = {
            "metrics_present": [s.key for s in metrics_present],
            "avg_rank": avg_rank,
            "relative_auc": relative_auc,
            "auc": auc_by_diff.get(diff, {}),
        }

    return summary, paths


def save_synthetic_problem_metrics(
    metrics_dir: str | Path,
    problem: str,
    summary_by_diff: dict[str, dict],
) -> Path:
    """Write one problem's :func:`_synthetic_problem_report` summary to JSON.

    Mirrors :func:`save_problem_metrics`/:func:`load_problem_metrics`'s
    intermediate-JSON idiom for the hypervolume pipeline, applied to the
    synthetic-comparison summary shape instead.
    """
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / f"{problem}_synthetic_metrics.json"
    with out_path.open("w") as f:
        json.dump({"problem": problem, "difficulties": summary_by_diff}, f)
    return out_path


def load_synthetic_problem_metrics(path: str | Path) -> tuple[str, dict[str, dict]]:
    """Load one problem's summary written by :func:`save_synthetic_problem_metrics`.

    Returns ``(problem_name, summary_by_diff)`` -- see
    :func:`_synthetic_problem_report` for the shape of ``summary_by_diff``.
    """
    with Path(path).open() as f:
        data = json.load(f)
    return data["problem"], data.get("difficulties", {})


def _combine_synthetic_summaries(
    summaries_by_problem: dict[str, dict[str, dict]],
) -> dict[str, dict]:
    """Combine every problem's :func:`_synthetic_problem_report` summary.

    ``summaries_by_problem`` maps ``{problem: summary_by_diff}`` -- either
    produced in-process (the monolithic path) or reloaded from disk via
    :func:`load_synthetic_problem_metrics` (the split aggregate path); both
    produce identical output here since the shape is the same either way.

    Each problem contributes at most one "row" per difficulty, so combining
    those rows here by plain averaging (``sum(values) / len(values)``)
    reproduces exactly the final reduction step inside
    ``ranking.average_ranks_over_rows``/``relative_auc_ratios_over_rows``
    (see the section docstring above), just performed over pre-computed
    per-row scalars instead of raw runs.

    Returns ``{difficulty: {"metrics_present": [MetricSpec, ...], "avg_rank":
    {column_key: {method: avg_rank}}, "relative_auc": {column_key: {method:
    avg_ratio}}, "n_rows": int, "auc_by_problem": {problem: {method:
    [seed_aucs]}}}}`` -- ``n_rows`` is the number of problems present at that
    difficulty, matching ``len(rows)`` in the old monolithic pipeline exactly
    (a problem is "present" at a difficulty iff its own per-problem job found
    at least one run there, which is precisely when it wrote an entry for
    that difficulty in its summary).
    """
    diffs: set[str] = set()
    for diff_map in summaries_by_problem.values():
        diffs.update(diff_map.keys())

    out: dict[str, dict] = {}
    for diff in sorted(diffs):
        metric_keys: set[str] = set()
        rank_lists: dict[str, dict[str, list[float]]] = {}
        ratio_lists: dict[str, dict[str, list[float]]] = {}
        auc_by_problem: dict[str, dict[str, list[float]]] = {}
        n_rows = 0
        for problem, diff_map in summaries_by_problem.items():
            entry = diff_map.get(diff)
            if entry is None:
                continue
            n_rows += 1
            metric_keys.update(entry.get("metrics_present") or [])
            for column_key, method_vals in (entry.get("avg_rank") or {}).items():
                for method, val in method_vals.items():
                    rank_lists.setdefault(column_key, {}).setdefault(method, []).append(val)
            for column_key, method_vals in (entry.get("relative_auc") or {}).items():
                for method, val in method_vals.items():
                    ratio_lists.setdefault(column_key, {}).setdefault(method, []).append(val)
            if entry.get("auc"):
                auc_by_problem[problem] = entry["auc"]

        avg_rank = {
            ck: {m: sum(vs) / len(vs) for m, vs in md.items() if vs}
            for ck, md in rank_lists.items()
        }
        relative_auc = {
            ck: {m: sum(vs) / len(vs) for m, vs in md.items() if vs}
            for ck, md in ratio_lists.items()
        }
        metrics_present = [s for s in _ordered_metrics() if s.key in metric_keys]

        out[diff] = {
            "metrics_present": metrics_present,
            "avg_rank": avg_rank,
            "relative_auc": relative_auc,
            "n_rows": n_rows,
            "auc_by_problem": auc_by_problem,
        }
    return out


def _render_synthetic_aggregate(
    summaries_by_problem: dict[str, dict[str, dict]],
    output_dir: str | Path,
    alpha: float = 0.05,
) -> list[str]:
    """Render the cross-problem outputs from combined per-problem summaries.

    Shared by :func:`summarize_synthetic_comparison` (monolithic, in-memory)
    and :func:`summarize_synthetic_comparison_aggregate` (split, disk-backed)
    so both write byte-for-byte the same
    ``synthetic_comparison_avg_rank_vs_evaluations.pdf``,
    ``synthetic_comparison_relative_auc_vs_evaluations.pdf``, and
    ``synthetic_comparison_stats_report.{json,md}`` per difficulty, plus one
    top-level ``synthetic_comparison_relative_auc_by_difficulty.pdf`` (see
    :func:`_plot_relative_auc_by_difficulty_figure`) spanning every
    difficulty at once -- unlike the three per-difficulty outputs above, this
    one is written directly under ``output_dir``, not a per-difficulty
    subfolder.
    """
    from .method_labels import METHOD_ABBREVIATIONS

    out_dir = Path(output_dir)
    combined = _combine_synthetic_summaries(summaries_by_problem)

    paths: list[str] = []
    for diff, entry in combined.items():
        diff_dir = out_dir / diff
        metrics_present = entry["metrics_present"]
        n_rows = entry["n_rows"]

        avg_rank_path = diff_dir / "synthetic_comparison_avg_rank_vs_evaluations.pdf"
        ok = _plot_avg_rank_figure(
            entry["avg_rank"], list(SYNTHETIC_METHODS), _SYNTHETIC_METHOD_STYLES,
            metrics_present, n_rows, avg_rank_path,
            method_labels=METHOD_ABBREVIATIONS,
        )
        if ok is not None:
            paths.append(str(ok))

        relative_auc_path = diff_dir / "synthetic_comparison_relative_auc_vs_evaluations.pdf"
        ok = _plot_relative_auc_figure(
            entry["relative_auc"], list(SYNTHETIC_METHODS), _SYNTHETIC_METHOD_STYLES,
            metrics_present, n_rows, relative_auc_path,
            method_labels=METHOD_ABBREVIATIONS,
        )
        if ok is not None:
            paths.append(str(ok))

        data = {
            problem: {_SYNTHETIC_AXIS: {diff: method_aucs}}
            for problem, method_aucs in entry["auc_by_problem"].items()
        }
        report = run_stats(data, proposed_method=SYNTHETIC_PROPOSED_METHOD, alpha=alpha)

        diff_dir.mkdir(parents=True, exist_ok=True)
        json_path = diff_dir / "synthetic_comparison_stats_report.json"
        md_path = diff_dir / "synthetic_comparison_stats_report.md"
        json_path.write_text(report_to_json(report), encoding="utf-8")
        md_path.write_text(_synthetic_report_to_markdown(report, diff), encoding="utf-8")
        paths.extend([str(json_path), str(md_path)])

    # One top-level (not per-difficulty) line-plot figure spanning every
    # difficulty at once -- see _plot_relative_auc_by_difficulty_figure and
    # this function's own docstring. `combined[diff]["relative_auc"]` is
    # already exactly one difficulty's own {column_key: {method: ratio}}
    # (averaged across problems present at that difficulty, never across
    # difficulties), so no further per-level computation is needed here.
    if combined:
        relative_auc_by_level = [(diff, combined[diff]["relative_auc"]) for diff in sorted(combined)]
        metric_keys_all = {s.key for e in combined.values() for s in e["metrics_present"]}
        metrics_present_all = [s for s in _ordered_metrics() if s.key in metric_keys_all]
        by_diff_path = out_dir / "synthetic_comparison_relative_auc_by_difficulty.pdf"
        ok = _plot_relative_auc_by_difficulty_figure(
            relative_auc_by_level, list(SYNTHETIC_METHODS), _SYNTHETIC_METHOD_STYLES,
            metrics_present_all, by_diff_path, method_labels=METHOD_ABBREVIATIONS,
        )
        if ok is not None:
            paths.append(str(ok))

    return paths


def summarize_synthetic_comparison(
    input_dir: str | Path,
    problems_config: str | Path = _DEFAULT_PROBLEMS_CONFIG,
    output_dir: str | Path | None = None,
    alpha: float = 0.05,
) -> list[str]:
    """ITCAS (batch) vs 5 baselines on every synthetic problem, one folder per difficulty.

    Writes, under ``<output_dir>/<difficulty>/``:

    * ``synthetic_comparison_<problem>_vs_evaluations.pdf`` -- one PDF **per
      problem** (not one combined grid): a single row of metric curves + raw
      product, no product-rank column (that per-iteration "who's ahead right
      now" column made sense when comparing across a shared grid; split
      one-problem-per-PDF it added nothing beyond the metric/product curves
      already shown).
    * ``synthetic_comparison_avg_rank_vs_evaluations.pdf`` -- one standalone
      figure: each method's average rank (1 = best) per metric + product,
      averaged across every problem at this difficulty (see
      :func:`_plot_avg_rank_figure` and
      ``ranking.average_ranks_over_rows``). This used to be a bottom row
      glued onto the combined grid; it is now its own PDF since there is no
      longer one combined grid for it to sit under.
    * ``synthetic_comparison_relative_auc_vs_evaluations.pdf`` -- one more
      standalone figure: for each metric (+ product) on each problem, the
      best AUC across every method/seed is found, every (method, seed) AUC is
      divided by that best, averaged over seeds then over problems (see
      :func:`_plot_relative_auc_figure` and
      ``ranking.relative_auc_ratios_over_rows``).
    * ``synthetic_comparison_stats_report.{json,md}`` -- Friedman-gated,
      Holm-Bonferroni corrected one-sided paired Wilcoxon dominance test
      (``H1: itcas_ndig > baseline``) against each of the five baselines,
      one row per problem (see :func:`_synthetic_report_to_markdown`).

    Additionally writes one figure directly under ``<output_dir>`` (not a
    per-difficulty subfolder, since it spans every difficulty at once):

    * ``synthetic_comparison_relative_auc_by_difficulty.pdf`` -- a line-plot
      sibling of the per-difficulty relative-AUC bar chart above: instead of
      averaging each method's relative-AUC ratio across difficulties into one
      bar, one line per method is drawn across difficulty levels on the
      x-axis, so a method's trend as the problem gets harder/easier stays
      visible (see :func:`_plot_relative_auc_by_difficulty_figure`).

    See the module-level section docstring above for why this only ever uses
    the evaluations axis and excludes the real-world problems
    (``spacecraft_formation_flying_a1``, ``casd_llm``).

    This is the monolithic (single-process) form of the pipeline: it holds
    every synthetic problem's ``RunSeries``/``CurveCache`` in memory at once,
    which is fine for small local test runs but not for the full sweep (see
    :func:`summarize_synthetic_comparison_problem` /
    :func:`summarize_synthetic_comparison_aggregate` for the memory-bounded,
    per-problem + aggregate split used operationally on the cluster). Both
    forms share :func:`_synthetic_problem_report` and
    :func:`_combine_synthetic_summaries`, so they cannot silently drift apart.
    """
    from .batch_vs_sequential import _collect_family_runs

    input_path = Path(input_dir)
    problems = _synthetic_problems(problems_config)
    out_dir = Path(output_dir) if output_dir is not None else Path(_SYNTHETIC_OUTPUT_DIR)

    runs_by_problem = _collect_family_runs(input_path, problems, list(SYNTHETIC_METHODS))
    caches_by_problem = {p: _precompute(runs) for p, runs in runs_by_problem.items() if runs}

    summaries_by_problem: dict[str, dict[str, dict]] = {}
    paths: list[str] = []
    for problem in problems:
        runs = runs_by_problem.get(problem, [])
        if not runs:
            continue
        cache = caches_by_problem.get(problem, {})
        summary, problem_paths = _synthetic_problem_report(problem, runs, cache, out_dir=out_dir)
        summaries_by_problem[problem] = summary
        paths.extend(problem_paths)

    paths.extend(_render_synthetic_aggregate(summaries_by_problem, out_dir, alpha=alpha))
    return paths


def summarize_synthetic_comparison_problem(
    input_dir: str | Path,
    problem: str,
    problems_config: str | Path = _DEFAULT_PROBLEMS_CONFIG,
    output_dir: str | Path | None = None,
    save_metrics_dir: str | Path | None = None,
    alpha: float = 0.05,
) -> list[str]:
    """Per-problem half of the split synthetic-comparison pipeline.

    Loads only ``problem``'s own runs (never any other problem's ``.jsonl``
    logs), renders its per-difficulty PDF(s) under ``output_dir`` exactly as
    :func:`summarize_synthetic_comparison` does for that problem's row (same
    file names, same content), and -- when ``save_metrics_dir`` is given --
    writes ``<save_metrics_dir>/<problem>_synthetic_metrics.json`` (see
    :func:`save_synthetic_problem_metrics`) so a later
    :func:`summarize_synthetic_comparison_aggregate` call can combine every
    problem's contribution without re-reading any run logs.

    ``problems_config`` is used only to validate ``problem`` is one of the
    synthetic problems (i.e. not one of the excluded real-world problems).
    ``alpha`` is accepted for CLI/signature symmetry with the other
    synthetic-comparison entry points but is unused here: the Friedman/
    Wilcoxon stats report needs every problem's raw per-seed AUCs together
    and is only produced by :func:`summarize_synthetic_comparison_aggregate`.
    """
    from .batch_vs_sequential import _collect_family_runs

    del alpha  # unused here, see docstring

    input_path = Path(input_dir)
    out_dir = Path(output_dir) if output_dir is not None else Path(_SYNTHETIC_OUTPUT_DIR)

    synthetic_problems = _synthetic_problems(problems_config)
    if problem not in synthetic_problems:
        raise ValueError(
            f"'{problem}' is not one of the synthetic problems in {problems_config} "
            f"(or is one of the excluded real-world problems: {_REAL_WORLD_PROBLEMS})"
        )

    runs = _collect_family_runs(input_path, [problem], list(SYNTHETIC_METHODS)).get(problem, [])
    if not runs:
        return []
    cache = _precompute(runs)

    summary, paths = _synthetic_problem_report(problem, runs, cache, out_dir=out_dir)

    if save_metrics_dir is not None:
        metrics_path = save_synthetic_problem_metrics(save_metrics_dir, problem, summary)
        paths.append(str(metrics_path))

    return paths


def summarize_synthetic_comparison_aggregate(
    metrics_dir: str | Path,
    output_dir: str | Path | None = None,
    alpha: float = 0.05,
) -> list[str]:
    """Aggregate half of the split synthetic-comparison pipeline.

    Reads every ``*_synthetic_metrics.json`` under ``metrics_dir`` (written
    by :func:`summarize_synthetic_comparison_problem`) -- no ``.jsonl`` run
    logs, no ``RunSeries`` reconstruction, for any problem -- and produces,
    per difficulty found across those files, the same three aggregate
    outputs :func:`summarize_synthetic_comparison` writes today:
    ``synthetic_comparison_avg_rank_vs_evaluations.pdf``,
    ``synthetic_comparison_relative_auc_vs_evaluations.pdf``, and
    ``synthetic_comparison_stats_report.{json,md}``, all under
    ``<output_dir>/<difficulty>/`` -- plus the same top-level
    ``synthetic_comparison_relative_auc_by_difficulty.pdf`` line-plot figure
    (see :func:`summarize_synthetic_comparison`'s docstring), written
    directly under ``<output_dir>``.
    """
    metrics_path = Path(metrics_dir)
    out_dir = Path(output_dir) if output_dir is not None else metrics_path.parent

    summaries_by_problem: dict[str, dict[str, dict]] = {}
    for f in sorted(metrics_path.glob("*_synthetic_metrics.json")):
        problem, diffs = load_synthetic_problem_metrics(f)
        summaries_by_problem[problem] = diffs

    if not summaries_by_problem:
        return []

    return _render_synthetic_aggregate(summaries_by_problem, out_dir, alpha=alpha)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def summarize_benchmark(
    input_dir: str | Path,
    benchmark: str,
    output_dir: str | Path | None = None,
    *,
    alpha: float = 0.05,
    save_metrics_dir: str | Path | None = None,
) -> list[str]:
    """Generate the two summary PDFs and one statistical report per itcas variant.

    When ``save_metrics_dir`` is given the per-problem hypervolume data are
    written to ``<save_metrics_dir>/<benchmark>_metrics.json`` so that a
    downstream aggregate job can combine results across problems without
    re-reading the raw run logs.
    """
    input_path = Path(input_dir)
    out_dir = Path(output_dir) if output_dir is not None else input_path

    runs = _discover_runs(input_path, benchmark=benchmark)
    if not runs:
        raise ValueError(f"No run logs found for '{benchmark}' under {input_path}")

    # Compute all metric curves once; reuse for both x-axis variants.
    curve_cache = _precompute(runs)

    out: list[str] = []
    for axis, suffix in (("evals", "vs_evaluations"), ("steps", "vs_steps")):
        path = out_dir / f"{benchmark}_summary_{suffix}.pdf"
        ok = _plot_problem(runs, benchmark, axis, path, curve_cache=curve_cache)
        if ok is not None:
            out.append(str(ok))
        path = out_dir / f"{benchmark}_curves_{suffix}.pdf"
        ok = _plot_problem_curves(runs, benchmark, axis, path, curve_cache=curve_cache)
        if ok is not None:
            out.append(str(ok))

    # Collect hypervolume data (used for both stats and optional metrics save).
    runs_by_problem = {benchmark: runs}
    caches_by_problem = {benchmark: curve_cache}
    hv_data = _collect_hv_for_stats(runs_by_problem, caches_by_problem)

    # Statistical reports — one per detected itcas quality variant.
    reports = run_stats_per_variant(hv_data, alpha=alpha)
    for variant, report in reports.items():
        safe_variant = variant.replace("/", "_")
        json_p, md_p = write_stats_report(
            report, out_dir, stem=f"{benchmark}_{safe_variant}_stats_report"
        )
        out.extend([str(json_p), str(md_p)])

    # Save intermediate metrics for the aggregate job when requested.
    if save_metrics_dir is not None:
        metrics_path = save_problem_metrics(
            Path(save_metrics_dir), benchmark, hv_data[benchmark]
        )
        out.append(str(metrics_path))

    return out


def summarize_all_benchmarks(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    alpha: float = 0.05,
) -> list[str]:
    input_path = Path(input_dir)
    out_dir = Path(output_dir) if output_dir is not None else input_path
    runs = _discover_runs(input_path)
    if not runs:
        return []

    # Group runs by problem for the combined hypervolume figure.
    runs_by_problem: dict[str, list[RunSeries]] = {}
    for r in runs:
        runs_by_problem.setdefault(r.problem, []).append(r)

    # Precompute metric curves once per problem — reused across both x-axis
    # variants and the combined hypervolume figure, giving a 4× speedup.
    caches_by_problem: dict[str, CurveCache] = {
        problem: _precompute(prob_runs)
        for problem, prob_runs in runs_by_problem.items()
    }

    paths: list[str] = []
    for problem in sorted(runs_by_problem):
        prob_runs = runs_by_problem[problem]
        cache = caches_by_problem[problem]
        for axis, suffix in (("evals", "vs_evaluations"), ("steps", "vs_steps")):
            path = out_dir / f"{problem}_summary_{suffix}.pdf"
            ok = _plot_problem(prob_runs, problem, axis, path, curve_cache=cache)
            if ok is not None:
                paths.append(str(ok))
            path = out_dir / f"{problem}_curves_{suffix}.pdf"
            ok = _plot_problem_curves(prob_runs, problem, axis, path, curve_cache=cache)
            if ok is not None:
                paths.append(str(ok))

    # Collect hypervolume data once; reuse for combined figure and stats.
    hv_data = _collect_hv_for_stats(runs_by_problem, caches_by_problem)

    # Combined hypervolume figure across all problems.
    for axis, suffix in (("evals", "vs_evaluations"), ("steps", "vs_steps")):
        hv_for_axis = {p: d[axis] for p, d in hv_data.items() if axis in d}
        path = out_dir / f"all_problems_hypervolume_{suffix}.pdf"
        ok = _plot_hypervolume_from_data(hv_for_axis, axis, path)
        if ok is not None:
            paths.append(str(ok))

    # Statistical reports — one per detected itcas quality variant, using the
    # full cross-problem hypervolume data so each variant is evaluated globally.
    reports = run_stats_per_variant(hv_data, alpha=alpha)
    for variant, report in reports.items():
        safe_variant = variant.replace("/", "_")
        json_p, md_p = write_stats_report(
            report, out_dir, stem=f"all_problems_{safe_variant}_stats_report"
        )
        paths.extend([str(json_p), str(md_p)])

    if reports:
        dom_p = write_dominance_table(reports, out_dir, stem="all_problems_dominance_table")
        paths.append(str(dom_p))

    return paths


def summarize_from_metrics(
    metrics_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    alpha: float = 0.05,
) -> list[str]:
    """Produce combined plots and global stats from per-problem metrics JSON files.

    Loads every ``*_metrics.json`` file written by :func:`summarize_benchmark`
    (with ``save_metrics_dir`` set) and produces:

    * ``all_problems_hypervolume_vs_evaluations.pdf``
    * ``all_problems_hypervolume_vs_steps.pdf``
    * One ``all_problems_<variant>_stats_report.{json,md}`` per itcas variant.
    """
    metrics_path = Path(metrics_dir)
    out_dir = Path(output_dir) if output_dir is not None else metrics_path.parent

    hv_data: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}
    for f in sorted(metrics_path.glob("*_metrics.json")):
        problem, axes = load_problem_metrics(f)
        hv_data[problem] = axes

    if not hv_data:
        return []

    paths: list[str] = []

    for axis, suffix in (("evals", "vs_evaluations"), ("steps", "vs_steps")):
        hv_for_axis = {p: d[axis] for p, d in hv_data.items() if axis in d}
        path = out_dir / f"all_problems_hypervolume_{suffix}.pdf"
        ok = _plot_hypervolume_from_data(hv_for_axis, axis, path)
        if ok is not None:
            paths.append(str(ok))

    reports = run_stats_per_variant(hv_data, alpha=alpha)
    for variant, report in reports.items():
        safe_variant = variant.replace("/", "_")
        json_p, md_p = write_stats_report(
            report, out_dir, stem=f"all_problems_{safe_variant}_stats_report"
        )
        paths.extend([str(json_p), str(md_p)])

    if reports:
        dom_p = write_dominance_table(reports, out_dir, stem="all_problems_dominance_table")
        paths.append(str(dom_p))

    return paths


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser("itcas.summarize")
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--benchmark", type=str, default=None,
        help="Restrict to one problem; default summarises every problem found.",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance level for Friedman gate and Holm-Bonferroni pairwise tests.",
    )
    parser.add_argument(
        "--save-metrics", type=str, default=None, dest="save_metrics",
        help=(
            "Directory to write intermediate per-problem metrics JSON. "
            "Used by per-problem Slurm jobs so the aggregate job can load them "
            "without re-reading all run logs."
        ),
    )
    parser.add_argument(
        "--aggregate-from", type=str, default=None, dest="aggregate_from",
        help=(
            "Load per-problem *_metrics.json files from this directory and "
            "produce combined hypervolume plots and global stats reports. "
            "Does not require --input-dir."
        ),
    )
    parser.add_argument(
        "--synthetic-comparison", action="store_true", dest="synthetic_comparison",
        help=(
            "Produce the itcas_ndig (batch) vs 5-baseline comparison on synthetic "
            "problems (see summarize_synthetic_comparison): one output folder per "
            "difficulty level, evaluations axis only. Ignores --benchmark/"
            "--save-metrics/--aggregate-from."
        ),
    )
    parser.add_argument(
        "--problems-config", type=str, default=_DEFAULT_PROBLEMS_CONFIG,
        dest="problems_config",
        help=(
            "Path to the final-problems config (used with --synthetic-comparison "
            "and --synthetic-comparison-problem)."
        ),
    )
    parser.add_argument(
        "--synthetic-comparison-problem", type=str, default=None,
        dest="synthetic_comparison_problem",
        help=(
            "Per-problem half of the split synthetic-comparison pipeline (see "
            "summarize_synthetic_comparison_problem): loads only this one problem's "
            "runs, renders its own PDF(s), and -- when --save-metrics is given -- "
            "writes <save-metrics>/<problem>_synthetic_metrics.json for a later "
            "--synthetic-comparison-aggregate-from run. Paired with --input-dir/"
            "--output-dir/--save-metrics/--problems-config. Use this (fanned out one "
            "job per problem) instead of --synthetic-comparison on the full sweep to "
            "avoid loading every synthetic problem's run history into memory at once."
        ),
    )
    parser.add_argument(
        "--synthetic-comparison-aggregate-from", type=str, default=None,
        dest="synthetic_comparison_aggregate_from",
        help=(
            "Aggregate half of the split synthetic-comparison pipeline (see "
            "summarize_synthetic_comparison_aggregate): loads every "
            "*_synthetic_metrics.json under this directory (written by "
            "--synthetic-comparison-problem runs) and produces the combined "
            "avg-rank/relative-AUC figures + stats report. Paired with --output-dir/"
            "--alpha. Does not require --input-dir and never reads any run logs."
        ),
    )
    args = parser.parse_args(argv)

    kw = dict(alpha=args.alpha)

    if args.synthetic_comparison_problem is not None:
        if args.input_dir is None:
            parser.error("--input-dir is required for --synthetic-comparison-problem")
        paths = summarize_synthetic_comparison_problem(
            args.input_dir, args.synthetic_comparison_problem,
            problems_config=args.problems_config, output_dir=args.output_dir,
            save_metrics_dir=args.save_metrics, **kw,
        )
        for p in paths:
            print(p)
        return 0

    if args.synthetic_comparison_aggregate_from is not None:
        paths = summarize_synthetic_comparison_aggregate(
            args.synthetic_comparison_aggregate_from, output_dir=args.output_dir, **kw
        )
        for p in paths:
            print(p)
        return 0

    if args.synthetic_comparison:
        if args.input_dir is None:
            parser.error("--input-dir is required for --synthetic-comparison")
        paths = summarize_synthetic_comparison(
            args.input_dir, problems_config=args.problems_config,
            output_dir=args.output_dir, **kw,
        )
        for p in paths:
            print(p)
        return 0

    if args.aggregate_from is not None:
        paths = summarize_from_metrics(
            args.aggregate_from, output_dir=args.output_dir, **kw
        )
        for p in paths:
            print(p)
        return 0

    if args.input_dir is None:
        parser.error("--input-dir is required unless --aggregate-from is used")

    if args.benchmark is None:
        paths = summarize_all_benchmarks(args.input_dir, output_dir=args.output_dir, **kw)
    else:
        paths = summarize_benchmark(
            args.input_dir, args.benchmark, output_dir=args.output_dir,
            save_metrics_dir=args.save_metrics,
            **kw,
        )
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
