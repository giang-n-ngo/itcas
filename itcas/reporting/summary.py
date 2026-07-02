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
import json
import re
from pathlib import Path
from typing import Iterable, Optional

import torch

from .metrics import REGISTRY as METRIC_REGISTRY, MetricSpec, RunSeries, build_reference, compute_metric
from .visualize import _discover_runs
from .stats import (
    StatsReport,
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
    fig.suptitle(
        f"{problem}: per-seed metric areas vs {axis_label.lower()}",
        fontsize=11,
    )

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

    fig.tight_layout(rect=(0, 0, 1, 0.97))
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
    from statistics import median

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
    fig.suptitle(f"{problem}: metric curves vs {axis_label.lower()}", fontsize=11)

    def _min_med_max(
        curves: list[list[float]],
    ) -> tuple[list[float], list[float], list[float]]:
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

        # Pre-compute product curves once; shared by the raw-value and rank columns.
        method_med: dict[str, list[float]] = {}
        method_lo: dict[str, list[float]] = {}
        method_hi: dict[str, list[float]] = {}
        x_ref_prod: Optional[list] = None
        for method in methods:
            seeded_runs = method_runs.get(method, [])
            prod_curves: list[list[float]] = []
            for run in seeded_runs:
                run_curves = curve_cache.get(run.run_name) or {}
                prod = _compute_seed_product_curve(run, axis, run_curves, metrics_present)
                if prod is None:
                    continue
                xs = run.x_evals if axis == "evals" else run.x_steps
                n = min(len(xs), len(prod))
                prod_curves.append(prod[:n])
                if x_ref_prod is None:
                    x_ref_prod = list(xs[:n])
            if not prod_curves:
                continue
            lo, med, hi = _min_med_max(prod_curves)
            method_med[method] = med
            method_lo[method] = lo
            method_hi[method] = hi

        # Raw product column (col N+1) — actual product values, higher is better.
        ax = axes[r_idx][-2]
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=7)
        plotted = False
        if method_med and x_ref_prod is not None:
            n_t = min(min(len(v) for v in method_med.values()), len(x_ref_prod))
            x_plot = x_ref_prod[:n_t]
            for method in sorted(method_med.keys()):
                color = method_colors[method]
                ax.plot(x_plot, method_med[method][:n_t], color=color,
                        linewidth=1.5, label=method)
                if len(method_runs.get(method, [])) > 1:
                    ax.fill_between(x_plot, method_lo[method][:n_t],
                                    method_hi[method][:n_t], color=color, alpha=0.15)
                plotted = True
        if not plotted:
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="grey")
        if r_idx == 0:
            ax.set_title("Product\n(raw, higher is better)", fontsize=8)
        if r_idx == n_rows - 1:
            ax.set_xlabel(axis_label, fontsize=8)

        # Rank column (col N+2) — rank of median product at each step (1 = best).
        ax = axes[r_idx][-1]
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="both", labelsize=7)
        plotted = False
        if method_med and x_ref_prod is not None:
            n_t = min(min(len(v) for v in method_med.values()), len(x_ref_prod))
            x_plot = x_ref_prod[:n_t]
            ranked = sorted(method_med.keys())
            for method in ranked:
                rank_curve = []
                for t in range(n_t):
                    vals = sorted(ranked, key=lambda m: -method_med[m][t])
                    rank_curve.append(vals.index(method) + 1)
                ax.plot(x_plot, rank_curve, color=method_colors[method],
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
        fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.97))

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
    fig.suptitle(
        f"Hypervolume (product of metric areas) — all problems vs {axis_label.lower()}",
        fontsize=11,
    )

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

    fig.tight_layout(rect=(0, 0, 1, 0.97))
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
    args = parser.parse_args(argv)

    kw = dict(alpha=args.alpha)

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
