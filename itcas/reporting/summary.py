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


def _feasible_pct_label(diff_key: str) -> str:
    """Format a shared-standard-problem difficulty key as a percent tick label.

    The shared ``p0_01``/``p0_05``/``p0_10``/``p0_20`` difficulty scale *is*
    the joint-feasible-fraction threshold (``p0_01`` = 1% of the space is
    feasible, etc.) -- this just renders that fraction as ``"N%"`` instead of
    the raw key, in whichever of its two string forms a caller has on hand:
    the bare ``threshold_pct`` value (``"0.01"``) or the directory/config-key
    tag (``"p0_01"``). Only meaningful for that shared scale -- FF/CASD's own
    per-problem difficulty levels are arbitrary numbered tiers, not feasible
    fractions, and must never be passed through this function.
    """
    body = diff_key[1:] if diff_key.startswith("p") else diff_key
    frac = float(body.replace("_", ".", 1))
    return f"{frac * 100:g}%"


# CurveCache maps run_name -> metric_key -> curve (list[float] | None)
CurveCache = dict[str, dict[str, Optional[list[float]]]]

# Sidecar key stashed inside a CurveCache entry alongside the real
# `_ordered_metrics()` keys -- NOT itself a metric. It records the `eps`
# value (`itcas.reporting.metrics.eps_archive_used`) that was actually used
# to compute that run's `epsilon_archive_size` curve, so `_precompute_cached`
# can tell a disk-cached entry apart from one computed under a since-changed
# `configs/experiments.json` calibration (this is the *only* metric with an
# external, orthogonal-to-run-data dependency -- see that function's
# docstring). Every consumer that iterates a CurveCache entry's metrics does
# so via `spec.key` lookups (never by iterating every key), so this extra
# key is inert everywhere except `_is_complete`.
_EPS_ARCHIVE_SIDECAR_KEY = "__eps_archive_used__"


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


def _precompute_cached(
    runs: list[RunSeries],
    cache_dir: str | Path | None,
    problem: str,
) -> CurveCache:
    """Like :func:`_precompute`, but skips ``compute_metric`` for any run
    already fully cached on disk under ``cache_dir``, and persists newly
    computed curves back.

    "Fully cached" means the on-disk entry for that ``run_name`` has every
    key :func:`_ordered_metrics` currently defines *and* -- since
    ``epsilon_archive_size`` is the one metric with an external dependency
    (``configs/experiments.json``'s per-problem/difficulty ``eps_archive``
    calibration, resolved by ``itcas.reporting.metrics.eps_archive_used``) --
    was cached under the ``eps`` value still current for that run today. A
    run cached by an older code version that computed fewer metrics, or
    whose cached ``eps`` no longer matches the config (e.g. after a
    recalibration), is treated as incomplete/stale and recomputed (from
    scratch, for every metric -- there is no partial/per-metric top-up), so
    neither a schema change nor a config recalibration can ever silently
    serve stale/incomplete data. Every other metric has no such external
    dependency (its curve is pure function of the run's own logged data), so
    this check is deliberately scoped to just this one sidecar rather than a
    general cache-versioning mechanism.

    ``cache_dir=None`` reproduces :func:`_precompute` exactly (no caching,
    no attempt to import :mod:`itcas.reporting.auc_cache`) -- every existing
    call site that doesn't pass a ``cache_dir`` keeps working unchanged.
    """
    if cache_dir is None:
        return _precompute(runs)

    from .auc_cache import load_curve_cache_for_problem, save_curve_cache  # lazy, avoid the same import-cycle this module already dodges for auc_cache
    from .metrics import eps_archive_used

    existing = load_curve_cache_for_problem(cache_dir, problem)
    metric_keys = {s.key for s in _ordered_metrics()}

    def _is_complete(run: RunSeries) -> bool:
        entry = existing.get(run.run_name)
        if entry is None or not (metric_keys <= entry.keys()):
            return False
        # `.get(...)` (not `[...]`) so an entry cached before this sidecar
        # existed (`None`) never equals a freshly-resolved eps and is
        # correctly treated as stale -- see docstring.
        return entry.get(_EPS_ARCHIVE_SIDECAR_KEY) == eps_archive_used(run)

    # Computed once per run and reused below (both to build `to_compute` and
    # in the main loop) so `eps_archive_used`'s config read isn't repeated.
    complete = {r.run_name: _is_complete(r) for r in runs}
    to_compute = [r for r in runs if not complete[r.run_name]]

    # Same ref-building rule as _precompute (first run with thresholds),
    # scoped to `to_compute` only -- on a full cache hit `to_compute` is
    # empty, `ref` stays None, and `build_reference` (itself not free -- it
    # constructs a 4000-point context reference set) is skipped entirely.
    ref = None
    for r in to_compute:
        if r.thresholds.numel() > 0:
            ref = build_reference(r)
            break

    cache: CurveCache = {}
    newly_computed: CurveCache = {}
    metrics = _ordered_metrics()
    for run in runs:
        if complete[run.run_name]:
            cache[run.run_name] = existing[run.run_name]
        else:
            curves = {spec.key: compute_metric(run, spec, ref=ref) for spec in metrics}
            curves[_EPS_ARCHIVE_SIDECAR_KEY] = eps_archive_used(run)
            cache[run.run_name] = curves
            newly_computed[run.run_name] = curves

    if newly_computed:
        save_curve_cache(cache_dir, problem, runs, newly_computed)

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
            ax.tick_params(axis="x", labelsize=10.5)
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
                    transform=ax.transAxes, fontsize=12, color="grey",
                )
            # Re-apply ticks/labels AFTER boxplot (which overrides them).
            ax.set_yticks(list(method_to_y.values()))
            ax.set_yticklabels(methods, fontsize=10.5)
            ax.set_ylim(0.5, len(methods) + 0.5)
            ax.invert_yaxis()
            if r_idx == 0:
                ax.set_title(_short_metric_label(spec), fontsize=12)
            if r_idx == n_rows - 1:
                ax.set_xlabel("area", fontsize=12)
            if c_idx == 0:
                ax.set_ylabel(f"{diff}\nmethod", fontsize=12)
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
        ax.tick_params(axis="x", labelsize=10.5)
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
                transform=ax.transAxes, fontsize=12, color="grey",
            )
        ax.set_yticks(list(method_to_y.values()))
        ax.set_yticklabels(methods, fontsize=10.5)
        ax.set_ylim(0.5, len(methods) + 0.5)
        ax.invert_yaxis()
        if r_idx == 0:
            ax.set_title("Hypervolume\n(product of metric areas, higher is better)", fontsize=12)
        if r_idx == n_rows - 1:
            ax.set_xlabel("product of areas", fontsize=12)
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
            ax.tick_params(axis="both", labelsize=10.5)
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
                        transform=ax.transAxes, fontsize=12, color="grey")
            if r_idx == 0:
                ax.set_title(_SHORT_CURVE_LABELS.get(spec.key, spec.label), fontsize=12)
            if r_idx == n_rows - 1:
                ax.set_xlabel(axis_label, fontsize=12)
            if c_idx == 0:
                ax.set_ylabel(f"{diff}", fontsize=12)

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
        ax.tick_params(axis="both", labelsize=10.5)
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
                    transform=ax.transAxes, fontsize=12, color="grey")
        if r_idx == 0:
            ax.set_title("Product\n(raw, higher is better)", fontsize=12)
        if r_idx == n_rows - 1:
            ax.set_xlabel(axis_label, fontsize=12)

        # Rank column (col N+2) — rank of the median product at each step
        # (1 = best), aligned across methods on the union-of-x-values grid
        # via forward-fill (see _rank_curves_on_union_grid).
        ax = axes[r_idx][-1]
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="both", labelsize=10.5)
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
                    transform=ax.transAxes, fontsize=12, color="grey")
        if r_idx == 0:
            ax.set_title("Product rank\n(1 = best)", fontsize=12)
        if r_idx == n_rows - 1:
            ax.set_xlabel(axis_label, fontsize=12)

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
            fontsize=12, bbox_to_anchor=(0.5, 0.0),
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
            ax.tick_params(axis="x", labelsize=10.5)
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
                    transform=ax.transAxes, fontsize=12, color="grey",
                )
            ax.set_yticks(list(method_to_y.values()))
            ax.set_yticklabels(all_methods, fontsize=10.5)
            ax.set_ylim(0.5, len(all_methods) + 0.5)
            ax.invert_yaxis()
            if r_idx == 0:
                ax.set_title(problem, fontsize=13.5)
            if r_idx == n_rows - 1:
                ax.set_xlabel("product of areas", fontsize=12)
            if c_idx == 0:
                ax.set_ylabel(f"{diff}\nmethod", fontsize=12)
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
    ax.tick_params(axis="both", labelsize=10.5)
    if not data:
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="grey")
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
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.invert_yaxis()


def _draw_metric_box_panel(
    ax,
    data: Optional[dict[str, list[float]]],
    methods: list[str],
    method_styles: dict[str, dict],
    method_labels: Optional[dict[str, str]],
    *,
    ascending_is_better: bool,
    reference_line: Optional[float] = None,
    sort_by_mean: bool = True,
    show_labels: bool = True,
) -> None:
    """Horizontal boxplot of one column's per-method distribution (rank/ratio spread across rows).

    Boxplot sibling of :func:`_draw_metric_bar_panel`: instead of a single
    bar for one already-averaged scalar per method, draws one box per method
    summarizing the spread of that method's own per-row values (e.g. from
    ``ranking.rank_lists_over_rows`` / ``ranking.relative_auc_ratio_lists_over_rows``)
    -- the distribution the bar chart's mean was hiding. By default
    (``sort_by_mean=True``) methods are sorted by their own mean value,
    exactly like the bar panel's sort (``ascending_is_better=True`` sorts
    smallest-mean-first, e.g. rank; ``False`` sorts largest-mean-first, e.g. a
    higher-is-better relative-AUC ratio); ``sort_by_mean=False`` instead keeps
    ``methods``' own given order top-to-bottom (a caller that wants a fixed,
    data-independent row order across every panel, e.g.
    :func:`_plot_relative_auc_box_grid_figure`, which always shows NDIG above
    NDIG-B regardless of which one scores higher in a given panel). By
    default (``show_labels=True``) each box gets a y-tick text label from
    ``method_labels``; ``show_labels=False`` omits both the tick marks and
    their text (e.g. when a caller draws one shared legend for the whole
    figure instead of repeating per-panel method-name labels).
    ``reference_line``, if given, draws the same dashed vertical guide as the
    bar panel. A method with a single-sample list still renders (a
    degenerate, zero-width box); a method with no data (missing or empty
    list) is simply omitted, mirroring the bar panel's ``present`` filter.
    """
    from matplotlib.colors import to_rgba  # lazy import, mirrors other plot helpers in this module

    ax.tick_params(axis="both", labelsize=10.5)
    present = [m for m in methods if data and data.get(m)]
    if not present:
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="grey")
        return
    if sort_by_mean:
        present_sorted = sorted(
            present, key=lambda m: sum(data[m]) / len(data[m]), reverse=not ascending_is_better
        )
    else:
        present_sorted = present
    y_pos = list(range(len(present_sorted)))
    values = [data[m] for m in present_sorted]
    labels = [
        (method_labels.get(m, m) if method_labels else m) for m in present_sorted
    ]
    bp = ax.boxplot(
        values, positions=y_pos, vert=False, patch_artist=True, showfliers=False, widths=0.6,
    )
    for m, box, whisker_lo, whisker_hi, cap_lo, cap_hi, median in zip(
        present_sorted, bp["boxes"],
        bp["whiskers"][0::2], bp["whiskers"][1::2],
        bp["caps"][0::2], bp["caps"][1::2],
        bp["medians"],
    ):
        color = method_styles[m]["color"]
        box.set_facecolor(to_rgba(color, alpha=0.6))
        box.set_edgecolor(color)
        for artist in (whisker_lo, whisker_hi, cap_lo, cap_hi, median):
            artist.set_color(color)
    if reference_line is not None:
        ax.axvline(reference_line, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.grid(True, axis="x", alpha=0.25)
    if show_labels:
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10.5)
    else:
        ax.set_yticks([])
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
        ax.set_title(f"{_SHORT_CURVE_LABELS.get(spec.key, spec.label)}", fontsize=12)
        if c_idx == 0:
            ax.set_ylabel(f"Avg rank across {n_rows} problems\n(best to worst)", fontsize=12)

    _draw_metric_bar_panel(
        axes[-1], avg_rank_row.get("product"), methods, method_styles, method_labels,
        ascending_is_better=True,
    )
    axes[-1].set_title("Product\n(raw)", fontsize=12)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_avg_rank_box_figure(
    avg_rank_lists: dict[str, dict[str, list[float]]],
    methods: list[str],
    method_styles: dict[str, dict],
    metrics_present: list[MetricSpec],
    n_rows: int,
    out_path: str | Path,
    method_labels: Optional[dict[str, str]] = None,
) -> Optional[Path]:
    """Standalone boxplot figure: rank distribution per metric (+ product), 1 = best.

    Exact structural sibling of :func:`_plot_avg_rank_figure` -- same figsize
    formula, same per-column titles, same y-label wording -- but draws each
    method's full per-row rank distribution (via :func:`_draw_metric_box_panel`)
    instead of collapsing it to a single averaged bar, so the spread across
    rows (``ranking.rank_lists_over_rows``) stays visible. ``ascending_is_better=True``
    throughout, exactly like the bar-chart version (rank 1 = best).
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
        _draw_metric_box_panel(
            ax, avg_rank_lists.get(spec.key), methods, method_styles, method_labels,
            ascending_is_better=True,
        )
        ax.set_title(f"{_SHORT_CURVE_LABELS.get(spec.key, spec.label)}", fontsize=12)
        if c_idx == 0:
            ax.set_ylabel(f"Avg rank across {n_rows} problems\n(best to worst)", fontsize=12)

    _draw_metric_box_panel(
        axes[-1], avg_rank_lists.get("product"), methods, method_styles, method_labels,
        ascending_is_better=True,
    )
    axes[-1].set_title("Product\n(raw)", fontsize=12)

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
        ax.set_title(f"{_SHORT_CURVE_LABELS.get(spec.key, spec.label)} {arrow}", fontsize=12)
        if c_idx == 0:
            ax.set_ylabel(
                f"Avg relative AUC across {n_rows} problems\n(1.0 = best)", fontsize=12
            )

    _draw_metric_bar_panel(
        axes[-1], relative_auc_row.get("product"), methods, method_styles, method_labels,
        ascending_is_better=False, reference_line=1.0,
    )
    axes[-1].set_title("Product ↑\n(raw)", fontsize=12)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_relative_auc_box_figure(
    relative_auc_lists: dict[str, dict[str, list[float]]],
    methods: list[str],
    method_styles: dict[str, dict],
    metrics_present: list[MetricSpec],
    n_rows: int,
    out_path: str | Path,
    method_labels: Optional[dict[str, str]] = None,
    x_axis_label: Optional[str] = None,
) -> Optional[Path]:
    """Standalone boxplot figure: relative-AUC ratio distribution per metric (+ product).

    Exact structural sibling of :func:`_plot_relative_auc_figure` -- same
    figsize formula, same per-column titles/direction arrows, same
    ``ascending_is_better``/``reference_line=1.0`` conventions -- but draws
    each method's full per-row ratio distribution (via
    :func:`_draw_metric_box_panel`, fed by
    ``ranking.relative_auc_ratio_lists_over_rows``) instead of collapsing it
    to a single averaged bar, so the spread across rows stays visible. See
    :func:`itcas.reporting.ranking.relative_auc_ratios_over_rows` for the
    exact definition of "ratio".

    ``x_axis_label``, when given, replaces the default ``ylabel``-positioned
    ``"Avg relative AUC across N problems (1.0 = best)"`` text with a plain,
    centered ``fig.supxlabel`` call using the given text verbatim (this is a
    *horizontal* box plot -- the ratio values are the x-axis, method names
    the y-tick labels -- so a caller that wants a short, literal x-axis title
    instead of the default verbose left-edge label can opt in per call site;
    omit it to keep every other caller's existing output byte-identical).
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
        _draw_metric_box_panel(
            ax, relative_auc_lists.get(spec.key), methods, method_styles, method_labels,
            ascending_is_better=not spec.higher_is_better, reference_line=1.0,
        )
        arrow = "↑" if spec.higher_is_better else "↓"
        ax.set_title(f"{_SHORT_CURVE_LABELS.get(spec.key, spec.label)} {arrow}", fontsize=12)
        if c_idx == 0 and x_axis_label is None:
            ax.set_ylabel(
                f"Avg relative AUC across {n_rows} problems\n(1.0 = best)", fontsize=12
            )

    _draw_metric_box_panel(
        axes[-1], relative_auc_lists.get("product"), methods, method_styles, method_labels,
        ascending_is_better=False, reference_line=1.0,
    )
    axes[-1].set_title("Product ↑\n(raw)", fontsize=12)

    if x_axis_label is not None:
        fig.supxlabel(x_axis_label, fontsize=12)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_relative_auc_box_grid_figure(
    relative_auc_lists: dict[str, dict[str, list[float]]],
    methods: list[str],
    method_styles: dict[str, dict],
    metrics_present: list[MetricSpec],
    out_path: str | Path,
    method_labels: Optional[dict[str, str]] = None,
) -> Optional[Path]:
    """Standalone 2x2-grid boxplot figure: relative-AUC ratio distribution, one panel per metric.

    A 2x2-grid sibling of :func:`_plot_relative_auc_box_figure`: same
    per-panel content (:func:`_draw_metric_box_panel`, same
    ``ascending_is_better``/``reference_line=1.0`` conventions, same
    ``ranking.relative_auc_ratio_lists_over_rows`` input shape and "ratio"
    definition), but laid out as a 2x2 grid of the (exactly four)
    :class:`MetricSpec`\\ s in ``metrics_present`` instead of one wide row --
    and, unlike that figure, never draws a **product** panel (there is no
    fifth panel to place in a 2x2 grid, and the product column has no
    ``MetricSpec``/direction of its own to plot alongside four fixed metric
    panels here).

    Panels are filled row-major (``metrics_present[0]`` top-left,
    ``metrics_present[1]`` top-right, ``metrics_present[2]`` bottom-left,
    ``metrics_present[3]`` bottom-right); any panel beyond the fourth is
    silently dropped and any short of four leaves the remaining grid cell(s)
    blank (axis turned off) rather than crashing, so this still degrades
    gracefully if a metric is ever absent from the data.

    Unlike :func:`_plot_relative_auc_box_figure`, each panel's boxes are
    drawn in ``methods``' own given order (top-to-bottom, not sorted by
    mean -- ``sort_by_mean=False``) and carry no per-panel y-tick labels
    (``show_labels=False``): the same two methods repeat in every one of the
    four panels, so their names are shown once, via a single shared legend
    below the grid, instead of four times.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Patch

    if not metrics_present:
        return None

    panels = metrics_present[:4]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), squeeze=False)
    flat_axes = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]

    for ax, spec in zip(flat_axes, panels):
        _draw_metric_box_panel(
            ax, relative_auc_lists.get(spec.key), methods, method_styles, method_labels,
            ascending_is_better=not spec.higher_is_better, reference_line=1.0,
            sort_by_mean=False, show_labels=False,
        )
        arrow = "↑" if spec.higher_is_better else "↓"
        ax.set_title(f"{_SHORT_CURVE_LABELS.get(spec.key, spec.label)} {arrow}", fontsize=12)

    for ax in flat_axes[len(panels):]:
        ax.axis("off")

    legend_handles = [
        Patch(
            facecolor=to_rgba(method_styles[m]["color"], alpha=0.6),
            edgecolor=method_styles[m]["color"],
            label=(method_labels.get(m, m) if method_labels else m),
        )
        for m in methods
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=len(methods),
        fontsize=12, bbox_to_anchor=(0.5, -0.05),
    )
    fig.text(
        0.5, -0.13,
        "Avg relative AUC across synthetic problems and difficulty (1.0 = best)",
        fontsize=12, ha="center",
    )

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def _draw_metric_line_panel(
    ax,
    relative_auc_by_level: list[tuple[str, dict[str, dict[str, list[float]]]]],
    column_key: str,
    methods: list[str],
    method_styles: dict[str, dict],
    method_labels: Optional[dict[str, str]],
    *,
    higher_is_better: bool,
) -> None:
    """One panel of :func:`_plot_relative_auc_by_difficulty_figure`: one line + IQR band per method.

    ``relative_auc_by_level`` is ``[(level_label, relative_auc_row), ...]``
    already in x-axis order (see that function's docstring), where each
    ``relative_auc_row`` is ``{column_key: {method: [ratio, ...]}}`` -- a
    *list* of per-seed (or per-row, depending on the caller) ratios per
    (column, method) at that level, not a single averaged scalar. Plots
    evenly spaced categorical x positions (``range(n)``, never the real
    difficulty value); at each level, a method's line point is the **mean**
    of its ratio list for ``column_key``, and a shaded band around the line
    shows the 25th-75th percentile (interquartile) spread of that same list
    (via ``numpy.percentile``), so a level with a single sample degenerates
    to a zero-width band instead of crashing. A ``float("nan")`` point (and
    correspondingly no band) is used for a (method, level) combination with
    no data at all, so the line breaks there instead of raising or silently
    skipping the method. Draws a dashed ``y=1.0`` reference line (mirrors
    ``_draw_metric_bar_panel``'s ``reference_line=1.0``); ``higher_is_better``
    is accepted for signature symmetry with the bar-panel helper but doesn't
    otherwise affect this panel's rendering (a line plot has no "sort
    direction" the way a bar chart does).
    """
    import numpy as np  # lazy import; numpy is already a hard dependency via matplotlib

    ax.tick_params(axis="both", labelsize=10.5)
    n = len(relative_auc_by_level)
    x = list(range(n))
    any_data = False
    for method in methods:
        ys: list[float] = []
        los: list[float] = []
        his: list[float] = []
        for _level_label, row in relative_auc_by_level:
            values = (row.get(column_key) or {}).get(method) or []
            if values:
                ys.append(float(sum(values) / len(values)))
                los.append(float(np.percentile(values, 25)))
                his.append(float(np.percentile(values, 75)))
            else:
                ys.append(float("nan"))
                los.append(float("nan"))
                his.append(float("nan"))
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
        los_arr = np.array(los, dtype=float)
        his_arr = np.array(his, dtype=float)
        if not np.all(np.isnan(los_arr)):  # skip fill_between for an all-NaN band
            ax.fill_between(
                x, los_arr, his_arr, color=style.get("color"), alpha=0.15, linewidth=0,
            )
    if not any_data:
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="grey")
        return
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_xticks(x)
    ax.set_xticklabels([lvl for lvl, _ in relative_auc_by_level], fontsize=10.5)


def _plot_relative_auc_by_difficulty_figure(
    relative_auc_by_level: list[tuple[str, dict[str, dict[str, list[float]]]]],
    methods: list[str],
    method_styles: dict[str, dict],
    metrics_present: list[MetricSpec],
    out_path: str | Path,
    method_labels: Optional[dict[str, str]] = None,
    x_axis_label: Optional[str] = None,
) -> Optional[Path]:
    """Standalone line-plot figure: relative-AUC ratio (+ IQR band) per metric, across difficulty.

    A sibling of :func:`_plot_relative_auc_figure` that shows the same
    per-(row, method, column) relative-AUC ratios (see
    :func:`itcas.reporting.ranking.relative_auc_ratios_over_rows`) *without*
    averaging away the per-difficulty breakdown: :func:`_plot_relative_auc_figure`
    collapses every row (difficulty level) into one number per method per
    column, whereas this figure draws one line per method per column, plotted
    across difficulty levels on the x-axis, so a method's trend as the
    problem gets harder/easier stays visible. ``x_axis_label``, when given,
    is drawn as every panel's own ``ax.set_xlabel`` (not one shared
    figure-level label) so it reads correctly however many panels this
    figure ends up with; omit it (the default) to reproduce the previous
    behavior of no axis title, just bare level tick labels -- appropriate
    for FF/CASD, whose difficulty levels are arbitrary numbered tiers, not a
    labelable quantity like a feasible-set proportion. Unlike the bar/box figures, the
    variation shown here is *not* collapsed away either: each line is
    surrounded by a shaded interquartile band (see :func:`_draw_metric_line_panel`)
    built directly from the same per-level ratio list, so a level's spread
    stays visible right alongside its trend.

    ``relative_auc_by_level`` is ``[(level_label, relative_auc_row), ...]``,
    already in the desired x-axis order, where each ``relative_auc_row`` is
    ``{column_key: {method: [ratio, ...]}}`` -- a *list* of ratios per
    (column, method) at that level (i.e. *not* averaged across levels, and
    not reduced to one scalar within a level either -- callers are
    responsible for computing each level's own ratio list independently; see
    ``ff_comparison``'s per-level ``ranking.relative_auc_seed_ratios_for_row``
    calls, whose list is each level's own per-seed spread, or
    ``casd_comparison``'s equivalent, or the synthetic pipeline's
    already-per-difficulty ``_combine_synthetic_summaries`` output, whose list
    is that difficulty's spread across *problems* rather than seeds -- see
    that pipeline's own docstring for why). ``level_label`` is used verbatim
    as that level's x-tick label and should already be short (unlike the
    verbose multi-line row labels used elsewhere in this package).

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
        ax.set_title(f"{_SHORT_CURVE_LABELS.get(spec.key, spec.label)} {arrow}", fontsize=12)
        if x_axis_label is not None:
            ax.set_xlabel(x_axis_label, fontsize=11)
        if c_idx == 0:
            ax.set_ylabel("Relative AUC by difficulty\n(1.0 = best)", fontsize=12)

    _draw_metric_line_panel(
        axes[-1], relative_auc_by_level, "product", methods, method_styles, method_labels,
        higher_is_better=True,
    )
    axes[-1].set_title("Product ↑\n(raw)", fontsize=12)
    if x_axis_label is not None:
        axes[-1].set_xlabel(x_axis_label, fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="lower center", ncol=min(len(labels), 6),
            fontsize=10.5, bbox_to_anchor=(0.5, -0.05),
        )

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_relative_auc_by_difficulty_grid_figure(
    relative_auc_by_level: list[tuple[str, dict[str, dict[str, list[float]]]]],
    methods: list[str],
    method_styles: dict[str, dict],
    metrics_present: list[MetricSpec],
    out_path: str | Path,
    method_labels: Optional[dict[str, str]] = None,
    x_axis_label: Optional[str] = None,
    legend_in_corner: bool = True,
) -> Optional[Path]:
    """Standalone 2x2-grid line-plot figure: relative-AUC ratio (+ IQR band) per metric, across difficulty.

    A 2x2-grid sibling of :func:`_plot_relative_auc_by_difficulty_figure`: same
    per-panel content (:func:`_draw_metric_line_panel`, same line + shaded
    IQR-band convention, same ``y=1.0`` reference line, same per-panel
    categorical difficulty x-ticks), but laid out as a 2x2 grid of the
    (exactly four) :class:`MetricSpec`\\ s in ``metrics_present`` instead of
    one wide row -- mirroring :func:`_plot_relative_auc_box_grid_figure`'s own
    2x2 layout choice for the same reason: this never draws a **product**
    panel either (there is no fifth panel to place in a 2x2 grid, and the
    product column has no ``MetricSpec``/direction of its own to plot
    alongside four fixed metric panels here).

    Panels are filled row-major (``metrics_present[0]`` top-left,
    ``metrics_present[1]`` top-right, ``metrics_present[2]`` bottom-left,
    ``metrics_present[3]`` bottom-right); any panel beyond the fourth is
    silently dropped and any short of four leaves the remaining grid cell(s)
    blank (axis turned off) rather than crashing. Each panel is forced
    **square** (``ax.set_box_aspect(1)``) regardless of the figure's own
    aspect ratio.

    One legend, shared across the whole figure, built from the first panel
    that actually plotted a line (not unconditionally the bottom-right one --
    a 2x2 grid panel can be the all-"(no data)" one while a later panel has
    real lines). ``legend_in_corner`` (default ``True``, the NDIG
    kernel-ablation report's own convention) draws it *inside* the last
    populated panel (the bottom-right one in the guaranteed-4-metric case
    this package always has), ``loc="upper right"``, single column so
    entries stack vertically (see
    :func:`_plot_normalized_avg_curve_grid_figure`'s sibling
    ``loc="lower right"`` placement in its own panel). ``legend_in_corner=False`` (the
    synthetic-comparison report's convention) instead draws one whole-figure
    horizontal legend below the grid (``loc="lower center"``, one row,
    ``bbox_to_anchor=(0.5, -0.1)`` -- the exact vertical offset
    :func:`_plot_normalized_avg_curve_figure` already uses for its own
    below-figure legend, so the two synthetic-comparison figures' legends sit
    the same visual distance below their panels' x-axis titles).
    ``x_axis_label``, when given, is drawn as every panel's own
    ``ax.set_xlabel`` (not one shared figure-level label), matching the
    one-row figure's own per-panel handling.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if not metrics_present:
        return None

    panels = metrics_present[:4]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 9.0), squeeze=False)
    flat_axes = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]

    for ax, spec in zip(flat_axes, panels):
        _draw_metric_line_panel(
            ax, relative_auc_by_level, spec.key, methods, method_styles, method_labels,
            higher_is_better=spec.higher_is_better,
        )
        arrow = "↑" if spec.higher_is_better else "↓"
        ax.set_title(f"{_SHORT_CURVE_LABELS.get(spec.key, spec.label)} {arrow}", fontsize=12)
        if x_axis_label is not None:
            ax.set_xlabel(x_axis_label, fontsize=11)
        ax.set_box_aspect(1)

    for ax in flat_axes[len(panels):]:
        ax.axis("off")

    handles, labels = [], []
    for ax in flat_axes:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    if handles and panels:
        if legend_in_corner:
            legend_ax = flat_axes[len(panels) - 1]
            legend_ax.legend(handles, labels, loc="upper right", ncol=1, fontsize=10.5)
        else:
            fig.legend(
                handles, labels, loc="lower center", ncol=min(len(labels), 6),
                fontsize=10.5, bbox_to_anchor=(0.5, -0.1),
            )

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Normalized average curve vs % of evaluation budget: one extra cross-row
# figure shared by every "grid of rows" comparison family (synthetic,
# ff_comparison, casd_comparison).
#
# Every other cross-row figure in this file (avg-rank box, relative-AUC box,
# relative-AUC-by-difficulty line+band) collapses each row down to *one
# scalar per (row, method, column)* -- an AUC, a rank, a ratio -- before ever
# averaging across rows. This figure instead keeps each row's full
# **iteration-by-iteration curve** and averages those curves across rows
# directly, so a reader can see not just "who wins on average" but *where in
# the search* (early/late) a method's advantage shows up. Two problems that
# scalar-AUC figures don't have to solve become unavoidable here:
#
# 1. **X-axis alignment.** A synthetic problem's own budget (``run.config
#    ["budget"]``) can be 100 or 200 depending on the problem, so plotting
#    raw ``x_evals`` and averaging positionally/by-value across rows would
#    silently blend a problem 40% through its search with another only 20%
#    through its own. FF/CASD share one fixed budget across every difficulty
#    row, so they have no such mismatch to begin with -- but reusing the same
#    "% of this run's own budget" x-axis (:func:`_run_budget`) for all three
#    families means one implementation serves everyone: for FF/CASD it's
#    simply a linear rescale of evals (harmless, budget is constant there),
#    and for synthetic it's the fix that makes cross-problem averaging valid.
#    ``run.config["budget"]`` itself only counts *acquisition* evaluations,
#    not the shared init dataset (``x_evals`` ranges ``[n_init,
#    n_init+budget]``, not ``[0, budget]`` -- see :func:`_run_budget`'s
#    docstring) -- :func:`_row_normalized_method_curves` subtracts each run's
#    own ``x_evals[0]`` before dividing by budget, so 0% always means "right
#    after the shared init dataset," the same point in the search for every
#    method/metric/row, instead of a row-varying offset that made curves
#    appear to start at different points for no real reason.
# 2. **Y-axis normalization.** A metric's raw scale (e.g. FCHV) can differ by
#    orders of magnitude between rows (problems/difficulty levels), so a
#    straight average of raw curves across rows would be dominated by
#    whichever row happens to have the largest numbers. This reuses the
#    ratio-to-row-best convention :mod:`itcas.reporting.ranking` already
#    established for AUCs (:func:`itcas.reporting.ranking.relative_auc_seed_ratios_for_row`):
#    within one row, divide every curve by that row's single most extreme
#    value seen anywhere (any method, any seed, any timestep) -- 1.0 means
#    "matched the best-ever value seen anywhere in this row." Exactly as in
#    that convention, this is *not* renormalized to "higher=better" for
#    lower-is-better metrics (FCFD): a ratio below 1.0 is worse there, not
#    better, matching every other relative-ratio figure in this file.
#
# Reading a run's curve off the shared ``pct_grid`` (see
# ``_row_normalized_method_curves`` below) uses linear interpolation between
# a run's own logged checkpoints (``numpy.interp``), NOT the piecewise-constant
# forward-fill ``_forward_fill_at`` uses for ``_rank_curves_on_union_grid``'s
# batch-vs-sequential rank alignment. The proposed method (``itcas_ndig``) is
# the only *batch* method in every family this figure covers, so it logs far
# fewer, more widely-spaced checkpoints than the sequential baselines;
# forward-filling those sparse checkpoints onto this figure's much finer
# 41-point grid turned them into a visible staircase that isn't present in
# any raw per-problem plot in this file (those just draw one straight
# ``matplotlib`` line segment between a method's own real checkpoints, i.e.
# implicit linear interpolation) -- switching to explicit linear
# interpolation here matches that existing visual convention instead of
# introducing a new artifact of the resampling step itself. Only the
# *interior* behavior changes: still NaN before a run's first checkpoint (no
# data yet to interpolate from) and still held flat at the last known value
# past a run's last checkpoint (both conventions agree there's nothing newer
# to show once a run ends).
# ---------------------------------------------------------------------------
def _default_pct_grid(n_grid_points: int = 41) -> list[float]:
    """Evenly spaced ``0..100`` percent-of-budget grid shared by every normalized-curve figure.

    A pure function of ``n_grid_points`` with no per-run/per-problem data
    dependency, so callers recompute it at render time (see
    :func:`_synthetic_problem_report`'s use of :func:`normalized_curve_grid_lists_over_rows`,
    which discards the grid it returns) rather than persisting it through
    JSON, where it would just be redundant state that has to stay in sync
    with this constant across every problem's intermediate metrics file.
    """
    import numpy as np  # lazy import, mirrors _draw_metric_line_panel's own numpy import

    return list(np.linspace(0, 100, n_grid_points))


def _run_budget(run: RunSeries) -> float:
    """This run's own **acquisition** budget -- the denominator for the shared % axis.

    ``run.config["budget"]`` (present on every run's ``.summary.json``
    ``"config"`` dict for every current experiment config) counts only the
    acquisition evaluations planned *after* the shared init dataset -- e.g. a
    run with ``n_init=40``/``budget=100`` reaches ``n_total=140`` individual
    evaluations total (``n_init`` + ``budget``, confirmed against a real
    ``.summary.json``: ``n_init=40, budget=100, n_total=140``), so ``x_evals``
    itself ranges ``[40, 140]``, not ``[0, 100]``. This is why
    :func:`_row_normalized_method_curves` converts to "% of budget" via
    ``(x - x_evals[0]) / budget``, not ``x / budget`` -- see that function.

    Falls back to this run's own logged evaluation *span*
    (``x_evals[-1] - x_evals[0]``, i.e. the acquisition evaluations actually
    performed) when the config key is absent or not a positive number
    (defensive only; should not be hit on real sweep data, but keeps this
    self-consistent with the offset subtracted at the call site rather than
    raising on a malformed/legacy run). Returns ``0.0`` when neither is
    available -- callers must treat that as "this run cannot be placed on a %
    axis" and skip it, exactly like every other "no data" skip in this module
    (never fabricate a budget).
    """
    budget = run.config.get("budget")
    if budget is not None:
        try:
            b = float(budget)
        except (TypeError, ValueError):
            b = 0.0
        if b > 0:
            return b
    if run.x_evals:
        b = float(run.x_evals[-1] - run.x_evals[0])
        if b > 0:
            return b
    return 0.0


def _row_normalized_method_curves(
    row_runs: list[RunSeries],
    cache: CurveCache,
    methods: list[str],
    column_key: str,
    higher_is_better: bool,
    pct_grid: list[float],
    compute_curve,
) -> dict[str, list[float]]:
    """One row's per-method mean curve, normalized by this row's own best-ever value.

    ``compute_curve`` is a ``(run) -> Optional[list[float]]`` callable
    producing a run's raw curve for the column being normalized -- a single
    metric's own curve (e.g. ``lambda run: (cache.get(run.run_name) or {}).get(spec.key)``)
    or the point-wise product curve (via :func:`_compute_seed_product_curve`)
    -- mirroring the callable-per-column pattern
    :func:`itcas.reporting.ranking._lookup_or_compute_auc` already uses for
    AUCs. ``cache``/``column_key`` are accepted (rather than baked only into
    ``compute_curve``) purely for signature/documentation parity with that
    same pattern; the actual values always come from calling ``compute_curve``.

    Two passes over ``row_runs`` (restricted to ``methods``, exactly like
    every other row helper in this module/``ranking.py``):

    1. **Find this row's best-ever value** (``row_best``): the max (if
       ``higher_is_better``) or min (otherwise) of every finite value in
       every run's curve, across every method being compared -- the same
       "extreme value anywhere in the row" the AUC-based ratio figures use
       (:func:`itcas.reporting.ranking.relative_auc_seed_ratios_for_row`),
       just taken over raw curve points instead of one AUC per seed. Returns
       ``{}`` if no finite value exists anywhere, or if ``row_best == 0``
       (an undefined ratio, exactly the same skip condition that function
       uses).
    2. **Normalize + align each run onto the shared ``pct_grid``**: divide
       the run's raw curve by ``row_best``, rescale its own ``x_evals`` to
       "% of its own budget" *measured from that run's own first checkpoint*
       (``100 * (x - x_evals[0]) / budget``, :func:`_run_budget`; the run is
       skipped if its budget is ``<= 0`` or it has no logged checkpoints at
       all) -- ``x_evals[0]`` is the shared init dataset, not the start of
       the acquisition budget (see :func:`_run_budget`'s docstring), so
       leaving it un-subtracted would put every run's first checkpoint at
       ``x_evals[0] / budget`` (e.g. 40%) instead of 0%, and -- since that
       ratio varies row to row with each row's own ``n_init``/``budget``
       mix -- would misalign different rows', methods', and even different
       metrics' curves against each other despite them all actually starting
       at the same point in the search (right after the shared init
       dataset). Then linearly interpolate (``numpy.interp``) the normalized
       curve onto every point
       of ``pct_grid`` -- NOT the piecewise-constant forward-fill
       (:func:`_forward_fill_at`) used elsewhere in this file, see this
       section's header comment for why (avoids a resampling-induced
       staircase on the proposed batch method's sparse checkpoints). ``left=
       nan`` keeps grid points before a run's first checkpoint undefined
       (never fabricated); ``numpy.interp``'s default ``right`` behavior
       (held at the curve's last value) matches forward-fill's own behavior
       past a run's last checkpoint, so nothing changes there. A method's
       per-run filled curves are then averaged elementwise, filtering NaN
       manually (``[v for v in col if v == v]``, matching
       ``_min_med_max``/the ``visualize.py`` mean/std helper's established
       avoidance of ``numpy.nanmean`` to sidestep all-NaN-slice
       ``RuntimeWarning`` spam) rather than with a numpy nan-function; a grid
       index with zero non-NaN contributions across every one of a method's
       runs stays NaN at that index.

    Returns ``{method: row_curve}`` (each ``row_curve`` the same length as
    ``pct_grid``) only for methods with at least one non-all-NaN curve --
    methods with no data for this row/column are simply absent, never given
    a fabricated all-NaN entry.
    """
    import numpy as np  # lazy import, mirrors this module's other lazy numpy imports

    method_set = set(methods)
    method_runs: dict[str, list[RunSeries]] = {}
    for run in row_runs:
        if run.method in method_set:
            method_runs.setdefault(run.method, []).append(run)

    all_values: list[float] = []
    for runs in method_runs.values():
        for run in runs:
            y = compute_curve(run)
            if y is None:
                continue
            all_values.extend(v for v in y if v == v)  # finite (non-NaN) only
    if not all_values:
        return {}
    row_best = max(all_values) if higher_is_better else min(all_values)
    if row_best == 0:
        return {}  # undefined ratio -- skip this row/column, mirrors ranking.py's best==0 skip

    out: dict[str, list[float]] = {}
    for method in methods:
        per_run_filled: list[list[float]] = []
        for run in method_runs.get(method, []):
            y = compute_curve(run)
            if y is None:
                continue
            budget = _run_budget(run)
            if budget <= 0 or not run.x_evals:
                continue
            normalized = [v / row_best for v in y]
            n = min(len(run.x_evals), len(normalized))
            # x_evals[0] is the shared init dataset (e.g. n_init=40), not the
            # start of the acquisition budget -- see _run_budget's docstring
            # (x_evals ranges [n_init, n_init+budget], not [0, budget]).
            # Subtracting it off is what puts every run's own first logged
            # checkpoint at 0% (not n_init/budget) regardless of that run's
            # own n_init/budget mix, which is also what makes every method
            # (they all share one init dataset per row) and every metric
            # (they all read off the same run's x_evals) line up at the same
            # 0% origin instead of starting mid-axis at a value that quietly
            # varied row to row.
            x0 = run.x_evals[0]
            x_pct = [100.0 * (x - x0) / budget for x in run.x_evals[:n]]
            # Linear interpolation, not forward-fill -- see this function's
            # docstring and this section's header comment for why.
            filled = np.interp(pct_grid, x_pct, normalized[:n], left=float("nan"))
            per_run_filled.append([float(v) for v in filled])
        if not per_run_filled:
            continue
        row_curve = []
        for col in zip(*per_run_filled):
            finite = [v for v in col if v == v]
            row_curve.append(sum(finite) / len(finite) if finite else float("nan"))
        if all(v != v for v in row_curve):
            continue  # all-NaN -- no grid point had any data for this method
        out[method] = row_curve
    return out


def normalized_curve_grid_lists_over_rows(
    rows: list,
    methods: list[str],
    metrics: list[MetricSpec] | None = None,
    n_grid_points: int = 41,
) -> tuple[list[float], dict[str, dict[str, list[list[float]]]]]:
    """Return ``(pct_grid, {column_key: {method: [row_curve, ...]}})`` across every row.

    ``rows`` is the same row-agnostic ``(row_label, runs, cache)`` triple
    every other function in this file/``ranking.py`` consumes -- a row is one
    problem for the synthetic pipeline, one difficulty level for
    ``ff_comparison``/``casd_comparison`` -- so this function does not need
    to know or care which. ``metrics`` defaults to :func:`_ordered_metrics`
    like every sibling row-aggregate function.

    For each row and each metric in ``metrics``, calls
    :func:`_row_normalized_method_curves` with that metric's own
    ``higher_is_better``; for the synthetic ``"product"`` column (the
    point-wise product across ``metrics``, via
    :func:`_compute_seed_product_curve`), calls it with
    ``higher_is_better=True`` -- the product is always higher-is-better by
    construction, exactly as every other product-column usage in this file
    already assumes. Both cases always read a run's curve on the ``"evals"``
    axis (``run.x_evals``, never ``run.x_steps``) -- the only fair shared
    x-axis between a batch method (one record per algorithmic step) and a
    sequential one (one record per individual evaluation), exactly the
    convention every other cross-method figure in this package already
    follows.

    The returned ``curve_lists[column_key][method]`` is a *list of that
    row's own curve*, one entry per row where that (column, method) had data
    -- the curve-valued sibling of ``ranking.rank_lists_over_rows``/
    ``ranking.relative_auc_ratio_lists_over_rows``'s "list of per-row
    contributions" shape, just one dimension richer (each list element is
    itself a curve of length ``len(pct_grid)``, not a scalar). The returned
    ``pct_grid`` (see :func:`_default_pct_grid`) is a pure function of
    ``n_grid_points`` with no data dependency; callers that don't need it
    (e.g. :func:`_synthetic_problem_report`, which only persists this
    function's second return value to JSON) are expected to discard it and
    recompute it at render time rather than round-trip it through storage.
    """
    if metrics is None:
        metrics = _ordered_metrics()
    pct_grid = _default_pct_grid(n_grid_points)

    curve_lists: dict[str, dict[str, list[list[float]]]] = {}
    for _row_label, row_runs, cache in rows:
        for spec in metrics:
            row_curves = _row_normalized_method_curves(
                row_runs, cache, methods, spec.key, spec.higher_is_better, pct_grid,
                lambda run, spec=spec, cache=cache: (cache.get(run.run_name) or {}).get(spec.key),
            )
            for method, curve in row_curves.items():
                curve_lists.setdefault(spec.key, {}).setdefault(method, []).append(curve)

        # Raw product of the metrics -- same per-row/per-method pipeline as
        # the metric columns above, always higher-is-better by construction.
        row_curves = _row_normalized_method_curves(
            row_runs, cache, methods, "product", True, pct_grid,
            lambda run, cache=cache: _compute_seed_product_curve(
                run, "evals", cache.get(run.run_name) or {}, metrics
            ),
        )
        for method, curve in row_curves.items():
            curve_lists.setdefault("product", {}).setdefault(method, []).append(curve)

    return pct_grid, curve_lists


def _draw_normalized_curve_panel(
    ax,
    pct_grid: list[float],
    curve_lists_col: dict[str, list[list[float]]],
    methods: list[str],
    method_styles: dict[str, dict],
    method_labels: Optional[dict[str, str]],
) -> bool:
    """One panel of :func:`_plot_normalized_avg_curve_figure`: one line + IQR band per method.

    ``curve_lists_col`` is ``{method: [row_curve, ...]}`` for one column --
    one entry of :func:`normalized_curve_grid_lists_over_rows`'s returned
    ``curve_lists`` dict. For each method with at least one row curve, at
    each grid index the finite values across that method's row curves are
    collected (manual NaN filter, same idiom as
    :func:`_row_normalized_method_curves`); the line point is their mean and
    the shaded band spans the 25th-75th percentile (``numpy.percentile`` on
    the plain filtered list, exactly the established ``np.percentile``
    idiom :func:`_draw_metric_line_panel` already uses -- only
    ``nanmean``/``nanpercentile`` *on arrays still containing NaN* are
    avoided in this module, not ``numpy.percentile`` itself). A row with a
    single contributing row-curve at some index still renders (a
    zero-width band there).

    Draws the axis grid only when at least one method actually plotted -- an
    empty panel is left otherwise bare so the caller can overlay the standard
    "(no data)" placeholder text exactly like every other panel in this file
    does, rather than this function drawing that text itself. Returns
    whether anything was plotted.
    """
    import numpy as np  # lazy import, mirrors _draw_metric_line_panel's own numpy import

    ax.tick_params(axis="both", labelsize=10.5)
    any_data = False
    for method in methods:
        row_curves = curve_lists_col.get(method) or []
        if not row_curves:
            continue
        ys: list[float] = []
        los: list[float] = []
        his: list[float] = []
        for col in zip(*row_curves):
            finite = [v for v in col if v == v]
            if finite:
                ys.append(float(sum(finite) / len(finite)))
                los.append(float(np.percentile(finite, 25)))
                his.append(float(np.percentile(finite, 75)))
            else:
                ys.append(float("nan"))
                los.append(float("nan"))
                his.append(float("nan"))
        if all(v != v for v in ys):  # all NaN -- no data anywhere for this method
            continue
        any_data = True
        style = method_styles.get(method, {})
        label = method_labels.get(method, method) if method_labels else method
        ax.plot(
            pct_grid, ys,
            color=style.get("color"), linestyle=style.get("linestyle", "-"),
            linewidth=1.2, label=label,
        )
        los_arr = np.array(los, dtype=float)
        his_arr = np.array(his, dtype=float)
        if not np.all(np.isnan(los_arr)):  # skip fill_between for an all-NaN band
            ax.fill_between(
                pct_grid, los_arr, his_arr, color=style.get("color"), alpha=0.15, linewidth=0,
            )
    if any_data:
        ax.grid(True, alpha=0.25)
    return any_data


def _plot_normalized_avg_curve_figure(
    pct_grid: list[float],
    curve_lists: dict[str, dict[str, list[list[float]]]],
    methods: list[str],
    method_styles: dict[str, dict],
    metrics_present: list[MetricSpec],
    out_path: str | Path,
    method_labels: Optional[dict[str, str]] = None,
) -> Optional[Path]:
    """Standalone line-plot figure: normalized metric curve (+ IQR band) vs % of budget, averaged across rows.

    Same single-row panel layout as :func:`_plot_relative_auc_by_difficulty_figure`
    (one column per metric in ``metrics_present`` plus a trailing Product
    column, each titled with a ``↑``/``↓`` direction arrow, ``fig_w = max(3.0
    * n_cols, 10.0)``, ``fig_h = 3.5``) and the same per-method
    color/linestyle from ``method_styles`` with one shared legend built from
    the first panel's handles -- but the x-axis here is **continuous and
    shared by construction** (every row already normalized onto the same
    ``pct_grid`` by :func:`normalized_curve_grid_lists_over_rows`), unlike
    that figure's categorical per-difficulty x-ticks, so every panel (not
    only the last) gets its own ``"% of evaluation budget"`` x-label.

    ``curve_lists`` is exactly :func:`normalized_curve_grid_lists_over_rows`'s
    second return value; ``pct_grid`` its first (or
    :func:`_default_pct_grid`'s output, when reloaded from a JSON summary
    that only persisted ``curve_lists`` -- see that function's docstring).
    Each panel is drawn by :func:`_draw_normalized_curve_panel`; when it
    reports nothing was plotted, this function overlays the standard
    "(no data)" placeholder text used throughout this file.

    Returns ``None`` (writing nothing) when ``metrics_present`` is empty,
    exactly the same "no metrics -> no figure" contract as every sibling
    ``_plot_*_figure``.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if not metrics_present:
        return None

    n_cols = len(metrics_present) + 1  # + product
    fig_w = max(3.0 * n_cols, 10.0)
    # Slightly taller than _plot_relative_auc_by_difficulty_figure's
    # fig_h=3.5: every panel here also carries its own "% of evaluation
    # budget" x-label (that figure only has categorical x-tick labels, no
    # axis label), so a little more vertical room keeps the legend from
    # sitting flush against it.
    fig_h = 3.75
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    axes = axes[0]

    for c_idx, spec in enumerate(metrics_present):
        ax = axes[c_idx]
        ok = _draw_normalized_curve_panel(
            ax, pct_grid, curve_lists.get(spec.key, {}), methods, method_styles, method_labels,
        )
        if not ok:
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="grey")
        arrow = "↑" if spec.higher_is_better else "↓"
        ax.set_title(f"{_SHORT_CURVE_LABELS.get(spec.key, spec.label)} {arrow}", fontsize=12)
        ax.set_xlabel("% of evaluation budget", fontsize=11)
        if c_idx == 0:
            ax.set_ylabel("Avg normalized metrics", fontsize=12)

    ax = axes[-1]
    ok = _draw_normalized_curve_panel(
        ax, pct_grid, curve_lists.get("product", {}), methods, method_styles, method_labels,
    )
    if not ok:
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="grey")
    ax.set_title("Product ↑\n(raw)", fontsize=12)
    ax.set_xlabel("% of evaluation budget", fontsize=11)

    # bbox_to_anchor's y is pushed a bit below the axes (vs. the -0.05 used
    # elsewhere in this module) to clear the per-panel x-axis labels above,
    # without leaving an oversized gap.
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="lower center", ncol=min(len(labels), 6),
            fontsize=10.5, bbox_to_anchor=(0.5, -0.1),
        )

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_normalized_avg_curve_grid_figure(
    pct_grid: list[float],
    curve_lists: dict[str, dict[str, list[list[float]]]],
    methods: list[str],
    method_styles: dict[str, dict],
    metrics_present: list[MetricSpec],
    out_path: str | Path,
    method_labels: Optional[dict[str, str]] = None,
) -> Optional[Path]:
    """Standalone 2x2-grid line-plot figure: normalized metric curve (+ IQR band) vs % of budget.

    A 2x2-grid sibling of :func:`_plot_normalized_avg_curve_figure`: same
    per-panel content (:func:`_draw_normalized_curve_panel`, same
    continuous shared ``pct_grid`` x-axis, same "% of evaluation budget"
    x-label on every panel), but laid out as a 2x2 grid of the (exactly four)
    :class:`MetricSpec`\\ s in ``metrics_present`` instead of one wide row --
    mirroring :func:`_plot_relative_auc_box_grid_figure`'s/
    :func:`_plot_relative_auc_by_difficulty_grid_figure`'s own 2x2 layout
    choice for the same reason: this never draws a **product** panel either.

    Panels are filled row-major; any panel beyond the fourth is silently
    dropped and any short of four leaves the remaining grid cell(s) blank
    (axis turned off) rather than crashing. Each panel is forced **square**
    (``ax.set_box_aspect(1)``) regardless of the figure's own aspect ratio.
    One legend, shared across the whole figure but drawn *inside* the
    bottom-right panel (``loc="lower right"``, single column so entries
    stack vertically) instead of a whole-figure legend below the grid --
    built from the first panel that actually plotted a line (not
    unconditionally the bottom-right one), but always attached to the last
    populated panel (the bottom-right one in the guaranteed-4-metric case
    this package always has) -- mirrors
    :func:`_plot_relative_auc_by_difficulty_grid_figure`'s own corner-legend
    placement (that figure uses ``"upper right"`` for its own panel; this one
    uses ``"lower right"`` so the legend sits beside the panel's own
    "% of evaluation budget" x-axis label instead of overlapping the top of
    the curve).

    Returns ``None`` (writing nothing) when ``metrics_present`` is empty,
    the same "no metrics -> no figure" contract as every sibling
    ``_plot_*_figure``.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if not metrics_present:
        return None

    panels = metrics_present[:4]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 9.0), squeeze=False)
    flat_axes = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]

    for ax, spec in zip(flat_axes, panels):
        ok = _draw_normalized_curve_panel(
            ax, pct_grid, curve_lists.get(spec.key, {}), methods, method_styles, method_labels,
        )
        if not ok:
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="grey")
        arrow = "↑" if spec.higher_is_better else "↓"
        ax.set_title(f"{_SHORT_CURVE_LABELS.get(spec.key, spec.label)} {arrow}", fontsize=12)
        ax.set_xlabel("% of evaluation budget", fontsize=11)
        ax.set_box_aspect(1)

    for ax in flat_axes[len(panels):]:
        ax.axis("off")

    handles, labels = [], []
    for ax in flat_axes:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    if handles and panels:
        legend_ax = flat_axes[len(panels) - 1]
        legend_ax.legend(handles, labels, loc="lower right", ncol=1, fontsize=10.5)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def normalized_avg_curve_figure_over_rows(
    rows: list,
    methods: list[str],
    method_styles: dict[str, dict],
    metrics_present: list[MetricSpec],
    out_path: str | Path,
    method_labels: Optional[dict[str, str]] = None,
    n_grid_points: int = 41,
) -> Optional[Path]:
    """Convenience wrapper for callers that already hold ``rows`` in memory (CASD/FF).

    Just :func:`normalized_curve_grid_lists_over_rows` followed by
    :func:`_plot_normalized_avg_curve_figure`. The synthetic pipeline cannot
    use this directly -- its per-problem/aggregate split means no single call
    site ever holds every problem's ``rows`` in memory at once (see
    :func:`_synthetic_problem_report`/:func:`_render_synthetic_aggregate`,
    which call :func:`normalized_curve_grid_lists_over_rows` and
    :func:`_plot_normalized_avg_curve_figure` separately, at different
    pipeline stages, with the intermediate ``curve_lists`` persisted to JSON
    in between).
    """
    pct_grid, curve_lists = normalized_curve_grid_lists_over_rows(
        rows, methods, metrics=metrics_present, n_grid_points=n_grid_points,
    )
    return _plot_normalized_avg_curve_figure(
        pct_grid, curve_lists, methods, method_styles, metrics_present, out_path,
        method_labels=method_labels,
    )


# ---------------------------------------------------------------------------
# Batch-improvement heatmap: % change in mean AUC, sequential -> batch.
# ---------------------------------------------------------------------------
def _plot_batch_improvement_heatmap(
    pct_by_column: dict[str, dict[str, dict[str, Optional[float]]]],
    families: list[tuple[str, str, str, str]],
    difficulties: list[str],
    metrics_present: list[MetricSpec],
    out_path: str | Path,
) -> Optional[Path]:
    """One PDF: a row of heatmap panels, one per metric (+ ``"product"``).

    Each panel is a ``len(families)`` x ``len(difficulties)`` grid of
    %-change-in-mean-AUC cells (see
    :mod:`itcas.reporting.batch_improvement_comparison` for how
    ``pct_by_column`` is computed -- aggregate-then-ratio across problems,
    FCFD sign-flipped so positive always means "batch improved", exactly like
    every other figure in this package's FCFD convention). ``pct_by_column``
    is ``{column_key: {family_key: {difficulty: pct | None}}}``; a ``None``
    cell (missing data, or an undefined ratio because the sequential side's
    aggregate AUC was exactly zero) renders as a fixed neutral grey rather
    than being drawn as (or confused with) a real 0% cell.

    ``families`` is ``[(family_key, sequential_method, batch_method,
    display_label), ...]`` in the desired top-to-bottom row order (proposed
    method first); only ``family_key`` (to look up ``pct_by_column``) and
    ``display_label`` (the y-tick text) are used here -- the method names
    themselves are irrelevant to rendering. ``display_label`` is drawn as a
    y-tick only on the **leftmost** panel -- every panel shares the same row
    order, so repeating it on all five would waste width without adding
    information; the freed-up space is what makes each cell large enough to
    carry its own printed value comfortably. ``difficulties`` is the desired
    left-to-right column order (raw difficulty tags, e.g. ``"p0_01"``);
    rendered as a feasible-set percent (``"1%"``, via :func:`_feasible_pct_label`)
    rather than the raw tag, with ``"Difficulty level"`` drawn as every
    panel's own ``ax.set_xlabel`` (not one shared ``fig.supxlabel``) so each
    panel reads correctly on its own.

    Each panel is independently colour-scaled (``TwoSlopeNorm(vcenter=0)``
    over that panel's own finite values' max absolute magnitude) rather than
    sharing one scale across panels, since different metrics have very
    different typical %-swing ranges -- forcing a shared scale would wash out
    a metric with naturally small swings under one with naturally large ones.
    White is therefore always exactly 0% in every panel; red = regression
    (batch worse), blue = improvement (batch better), the standard ``RdBu``
    diverging colormap. Each finite cell is also annotated with its signed
    percentage (``f"{v:+.0f}%"``), text color switched to white near either
    saturated end of that panel's own color scale (``0.3 <= norm(v) <= 0.7``
    stays black) so it stays legible against dark fills; a ``None``/masked
    cell gets no text, just its fixed grey fill.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.colors
    import matplotlib.pyplot as plt
    import numpy as np

    if not metrics_present:
        return None

    columns: list[tuple[str, str]] = [(s.key, f"{_SHORT_CURVE_LABELS.get(s.key, s.label)} {'↑' if s.higher_is_better else '↓'}") for s in metrics_present]
    columns.append(("product", "Product ↑\n(raw)"))

    n_cols = len(columns)
    n_rows_grid = len(families)
    n_diffs = len(difficulties)
    fig_w = max(3.2 * n_cols, 10.0)
    fig_h = max(0.55 * n_rows_grid + 1.8, 3.5)
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    axes = axes[0]

    cmap = matplotlib.colormaps.get_cmap("RdBu").copy()
    cmap.set_bad(color="lightgrey")

    for c_idx, (column_key, title) in enumerate(columns):
        ax = axes[c_idx]
        col_data = pct_by_column.get(column_key, {})

        grid = np.full((n_rows_grid, n_diffs), np.nan, dtype=float)
        for r_idx, (family_key, _seq, _batch, _label) in enumerate(families):
            fam_data = col_data.get(family_key, {})
            for d_idx, diff in enumerate(difficulties):
                v = fam_data.get(diff)
                if v is not None:
                    grid[r_idx, d_idx] = v

        masked = np.ma.masked_invalid(grid)
        finite_vals = grid[np.isfinite(grid)]
        m = float(np.max(np.abs(finite_vals))) if finite_vals.size else 1.0
        if m == 0.0:
            m = 1.0
        norm = matplotlib.colors.TwoSlopeNorm(vcenter=0.0, vmin=-m, vmax=m)

        im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(n_diffs))
        ax.set_xticklabels([_feasible_pct_label(d) for d in difficulties], fontsize=10.5)
        ax.set_yticks(range(n_rows_grid))
        if c_idx == 0:
            ax.set_yticklabels([f[3] for f in families], fontsize=10.5)
        else:
            # Every panel shares the same row order -- repeating the
            # family/method names on every panel would waste the width
            # freed up for larger cells without adding information, so only
            # the leftmost panel carries them.
            ax.set_yticklabels([])
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Difficulty level", fontsize=11)

        for r_idx in range(n_rows_grid):
            for d_idx in range(n_diffs):
                v = grid[r_idx, d_idx]
                if not np.isfinite(v):
                    continue
                # norm(v) is TwoSlopeNorm's [0, 1] output -- 0.5 is the
                # (white) center; near either saturated end (dark red/blue)
                # needs light text, near the white center needs dark text.
                shade = norm(v)
                text_color = "black" if 0.3 <= shade <= 0.7 else "white"
                ax.text(
                    d_idx, r_idx, f"{v:+.0f}%",
                    ha="center", va="center", fontsize=10.5, color=text_color,
                )

        fig.colorbar(im, ax=ax, shrink=0.7)

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
    auc_cache_dir: str | Path | None = None,
) -> tuple[dict[str, dict], list[str]]:
    """One problem's contribution to the synthetic comparison, per difficulty.

    Uses only ``runs``/``cache`` for this single ``problem`` -- never touches
    any other problem's data -- so this is safe to call from a per-problem
    job holding just one problem's ``RunSeries`` in memory.

    When ``out_dir`` is given, also renders this problem's own per-difficulty
    PDF (``<out_dir>/<difficulty>/synthetic_comparison_<problem>_vs_evaluations.pdf``),
    byte-for-byte the same file :func:`summarize_synthetic_comparison` used to
    render inline for this problem's row.

    When ``auc_cache_dir`` is given (see :func:`summarize_synthetic_comparison`
    / :func:`summarize_synthetic_comparison_problem`), every
    ``ranking.*`` call below is fed that directory's already-cached per-seed
    AUCs (:mod:`itcas.reporting.auc_cache`) so it can skip recomputing
    ``summary._curve_area`` for any ``(run, axis, metric)`` already on disk;
    newly computed AUCs for this problem's own runs are merged back in before
    returning. This never changes the returned summary or any figure's
    content -- a cached AUC is numerically identical to what ``_curve_area``
    would compute fresh (see ``ranking._lookup_or_compute_auc``).

    Returns ``(summary_by_diff, paths_written)`` where ``summary_by_diff`` is
    ``{difficulty: {"metrics_present": [metric_key, ...], "avg_rank":
    {column_key: {method: rank}}, "relative_auc": {column_key: {method:
    ratio}}, "auc": {method: [seed_aucs]}, "normalized_curve_lists":
    {column_key: {method: [row_curve]}}}}`` -- fully JSON-serializable (see
    :func:`save_synthetic_problem_metrics`) and exactly what
    :func:`_combine_synthetic_summaries` expects as one entry of its
    ``summaries_by_problem`` argument. ``normalized_curve_lists`` holds at
    most one curve per (column, method) here -- this problem's own single
    row -- exactly mirroring how ``avg_rank``/``relative_auc`` are also
    single-row contributions at this stage (see
    :func:`normalized_curve_grid_lists_over_rows`); the shared percent-grid
    itself is *not* stored here (it is a pure function of the number of grid
    points, not of any per-problem data -- see :func:`_default_pct_grid`) so
    it never needs to round-trip through this JSON-serializable summary.
    """
    from .batch_vs_sequential import _collect_product_auc_for_stats, _rows_by_problem, plot_group_grid
    from .method_labels import METHOD_ABBREVIATIONS
    from .ranking import average_ranks_over_rows, relative_auc_ratios_over_rows

    runs_by_problem = {problem: runs}
    caches_by_problem = {problem: cache}
    auc_by_diff = _collect_product_auc_for_stats(runs_by_problem, caches_by_problem).get(problem, {})

    existing_auc_cache = {}
    if runs and auc_cache_dir is not None:
        from .auc_cache import load_auc_cache_for_problem

        existing_auc_cache = load_auc_cache_for_problem(auc_cache_dir, problem, runs=runs)

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
        avg_rank = average_ranks_over_rows(
            rows, list(SYNTHETIC_METHODS), _SYNTHETIC_AXIS, auc_cache=existing_auc_cache
        )
        relative_auc = relative_auc_ratios_over_rows(
            rows, list(SYNTHETIC_METHODS), _SYNTHETIC_AXIS, auc_cache=existing_auc_cache
        )
        # Discard the returned pct_grid -- see this function's docstring on
        # `normalized_curve_lists` for why it's recomputed at render time
        # (`_default_pct_grid()`) rather than persisted here.
        normalized_curve_lists = normalized_curve_grid_lists_over_rows(rows, list(SYNTHETIC_METHODS))[1]
        summary[diff] = {
            "metrics_present": [s.key for s in metrics_present],
            "avg_rank": avg_rank,
            "relative_auc": relative_auc,
            "auc": auc_by_diff.get(diff, {}),
            "normalized_curve_lists": normalized_curve_lists,
        }

    if runs and auc_cache_dir is not None:
        from .auc_cache import compute_auc_table, save_auc_cache

        save_auc_cache(auc_cache_dir, problem, runs, compute_auc_table(runs, cache))

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

    Each problem contributes at most one "row" per difficulty. Rather than
    reducing those rows straight to a mean (the final step inside
    ``ranking.average_ranks_over_rows``/``relative_auc_ratios_over_rows``,
    see the section docstring above), this collects each difficulty's raw
    per-problem values as lists (``avg_rank_lists``/``relative_auc_lists``)
    -- exactly the shape ``ranking.rank_lists_over_rows``/
    ``ranking.relative_auc_ratio_lists_over_rows`` would produce had they
    been called directly over every problem's row at once, since a
    per-problem job's own ``avg_rank``/``relative_auc`` entry (see
    :func:`_synthetic_problem_report`) is already exactly one row's
    contribution to those lists. Reducing further to a mean, when wanted, is
    now the caller's job (:func:`_render_synthetic_aggregate` passes these
    lists straight to the boxplot figures, which do their own
    mean-based sort without needing a separate scalar reduction here).

    Returns ``{difficulty: {"metrics_present": [MetricSpec, ...],
    "avg_rank_lists": {column_key: {method: [rank, ...]}},
    "relative_auc_lists": {column_key: {method: [ratio, ...]}}, "n_rows":
    int, "auc_by_problem": {problem: {method: [seed_aucs]}},
    "normalized_curve_lists": {column_key: {method: [row_curve, ...]}}}}`` --
    ``n_rows`` is the number of problems present at that difficulty, matching
    ``len(rows)`` in the old monolithic pipeline exactly (a problem is
    "present" at a difficulty iff its own per-problem job found at least one
    run there, which is precisely when it wrote an entry for that difficulty
    in its summary). ``normalized_curve_lists`` is collected the same way as
    ``avg_rank_lists``/``relative_auc_lists`` -- one problem's at-most-one-row
    contribution appended per column/method -- except each appended element
    is itself a whole curve (that problem's own row curve) rather than a
    scalar, exactly reproducing what
    :func:`normalized_curve_grid_lists_over_rows` would return had it been
    called directly over every problem's row at once (see this module's
    "Normalized average curve" section docstring for why that equivalence
    holds).
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
        curve_lists: dict[str, dict[str, list[list[float]]]] = {}
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
            for column_key, method_curves in (entry.get("normalized_curve_lists") or {}).items():
                for method, curves in method_curves.items():
                    curve_lists.setdefault(column_key, {}).setdefault(method, []).extend(curves)
            if entry.get("auc"):
                auc_by_problem[problem] = entry["auc"]

        metrics_present = [s for s in _ordered_metrics() if s.key in metric_keys]

        out[diff] = {
            "metrics_present": metrics_present,
            "avg_rank_lists": rank_lists,
            "relative_auc_lists": ratio_lists,
            "n_rows": n_rows,
            "auc_by_problem": auc_by_problem,
            "normalized_curve_lists": curve_lists,
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
    ``synthetic_comparison_avg_rank_vs_evaluations.pdf`` (a **boxplot**, one
    box per method showing that method's rank distribution across the
    problems present at that difficulty -- see :func:`_plot_avg_rank_box_figure`),
    ``synthetic_comparison_relative_auc_vs_evaluations.pdf`` (likewise a
    boxplot of relative-AUC ratios -- see :func:`_plot_relative_auc_box_figure`),
    and ``synthetic_comparison_stats_report.{json,md}`` per difficulty, plus
    two top-level figures spanning every difficulty at once -- unlike the
    three per-difficulty outputs above, these two are written directly under
    ``output_dir``, not a per-difficulty subfolder:

    * ``synthetic_comparison_relative_auc_by_difficulty.pdf`` (a 2x2 grid of
      the 4 metrics, no product panel, horizontal legend below the grid --
      see :func:`_plot_relative_auc_by_difficulty_grid_figure`). Its shaded
      band shows
      spread **across problems** at each difficulty, not across seeds: a
      "row" in this split/aggregate pipeline already collapses each
      problem's own seeds down to one ratio inside its own per-problem job
      (see :func:`_synthetic_problem_report`), so the only variation left to
      show here is across the problems sharing a difficulty -- unlike
      ``ff_comparison``/``casd_comparison``, whose by-difficulty band shows
      spread across seeds within one real-world problem's own rows.
    * ``synthetic_comparison_normalized_avg_curve_vs_pct_budget.pdf`` (see
      :func:`_plot_normalized_avg_curve_figure` and this module's
      "Normalized average curve" section docstring): each metric's -- plus
      product's -- curve, normalized to its own (problem, difficulty) row's
      best and averaged, with a shaded IQR band, over *every* row across
      *every* problem and *every* difficulty at once -- unlike the two
      figures above, this one is not split per difficulty at all (every
      difficulty's own per-problem row curves, already computed once per
      problem in :func:`_synthetic_problem_report`, are pooled into a single
      list per method/column here), since its x-axis is already "% of each
      row's own evaluation budget" rather than difficulty, so folding
      difficulty into the same across-row pooling as problems needs no
      separate axis.
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
        ok = _plot_avg_rank_box_figure(
            entry["avg_rank_lists"], list(SYNTHETIC_METHODS), _SYNTHETIC_METHOD_STYLES,
            metrics_present, n_rows, avg_rank_path,
            method_labels=METHOD_ABBREVIATIONS,
        )
        if ok is not None:
            paths.append(str(ok))

        relative_auc_path = diff_dir / "synthetic_comparison_relative_auc_vs_evaluations.pdf"
        ok = _plot_relative_auc_box_figure(
            entry["relative_auc_lists"], list(SYNTHETIC_METHODS), _SYNTHETIC_METHOD_STYLES,
            metrics_present, n_rows, relative_auc_path,
            method_labels=METHOD_ABBREVIATIONS, x_axis_label="Avg relative AUC",
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

    # One top-level (not per-difficulty) 2x2-grid figure spanning every
    # difficulty at once -- see _plot_relative_auc_by_difficulty_grid_figure
    # and this function's own docstring. `combined[diff]["relative_auc_lists"]`
    # is already exactly one difficulty's own {column_key: {method: [ratio,
    # ...]}} -- one ratio per problem present at that difficulty, never
    # averaged across difficulties -- so no further per-level computation is
    # needed here; `_draw_metric_line_panel` does its own mean-point/IQR-band
    # reduction from this list. As noted in this function's docstring, the
    # resulting band shows spread across *problems*, not seeds. Laid out as a
    # 2x2 grid of the 4 metrics (no product panel, unlike the one-row
    # `_plot_relative_auc_by_difficulty_figure` other reports in this package
    # use) with a horizontal legend below the grid (`legend_in_corner=False`)
    # rather than the NDIG-ablation report's own inside-the-panel corner
    # legend -- `bbox_to_anchor=(0.5, -0.1)` matches
    # `_plot_normalized_avg_curve_figure`'s own below-figure legend spacing
    # (this synthetic pipeline's other top-level figure, rendered just below)
    # so both figures' legends sit the same visual distance under their
    # panels' x-axis titles.
    if combined:
        # Tick labels are the feasible-fraction percent (e.g. "1%" for the
        # p0_01/threshold_pct=0.01 level), not the raw key -- see
        # _feasible_pct_label; only valid for this shared standard-problem
        # difficulty scale (feasible-fraction thresholds), unlike FF/CASD's
        # own arbitrary numbered tiers.
        relative_auc_by_level = [
            (_feasible_pct_label(diff), combined[diff]["relative_auc_lists"])
            for diff in sorted(combined)
        ]
        metric_keys_all = {s.key for e in combined.values() for s in e["metrics_present"]}
        metrics_present_all = [s for s in _ordered_metrics() if s.key in metric_keys_all]
        by_diff_path = out_dir / "synthetic_comparison_relative_auc_by_difficulty.pdf"
        ok = _plot_relative_auc_by_difficulty_grid_figure(
            relative_auc_by_level, list(SYNTHETIC_METHODS), _SYNTHETIC_METHOD_STYLES,
            metrics_present_all, by_diff_path, method_labels=METHOD_ABBREVIATIONS,
            x_axis_label="Difficulty level", legend_in_corner=False,
        )
        if ok is not None:
            paths.append(str(ok))

        # Pool every (problem, difficulty) row's own normalized curve list
        # (each already computed once per problem in
        # _synthetic_problem_report and collected per-difficulty by
        # _combine_synthetic_summaries) into one {column_key: {method:
        # [row_curve, ...]}} spanning every problem and every difficulty at
        # once -- see this function's docstring for why this figure, unlike
        # the two above, is never split per difficulty.
        normalized_curve_lists: dict[str, dict[str, list[list[float]]]] = {}
        for entry in combined.values():
            for column_key, method_curves in (entry.get("normalized_curve_lists") or {}).items():
                for method, curves in method_curves.items():
                    normalized_curve_lists.setdefault(column_key, {}).setdefault(method, []).extend(curves)

        normalized_curve_path = out_dir / "synthetic_comparison_normalized_avg_curve_vs_pct_budget.pdf"
        ok = _plot_normalized_avg_curve_figure(
            _default_pct_grid(), normalized_curve_lists, list(SYNTHETIC_METHODS),
            _SYNTHETIC_METHOD_STYLES, metrics_present_all, normalized_curve_path,
            method_labels=METHOD_ABBREVIATIONS,
        )
        if ok is not None:
            paths.append(str(ok))

    return paths


def summarize_synthetic_comparison(
    input_dir: str | Path,
    problems_config: str | Path = _DEFAULT_PROBLEMS_CONFIG,
    output_dir: str | Path | None = None,
    alpha: float = 0.05,
    auc_cache_dir: str | Path | None = None,
) -> list[str]:
    """ITCAS (batch) vs 5 baselines on every synthetic problem, one folder per difficulty.

    ``auc_cache_dir`` (see :mod:`itcas.reporting.auc_cache`) defaults to
    ``Path(input_dir).parent / "auc_cache"`` when ``None`` -- a stable
    location shared across every report/problem regardless of
    ``output_dir`` (which varies per run, including throwaway smoke-test
    dirs). Passed through to :func:`_synthetic_problem_report` for each
    problem so its ``ranking.*`` calls can skip recomputing AUCs already on
    disk; this never changes any figure's content, only how fast it's
    produced.

    Writes, under ``<output_dir>/<difficulty>/``:

    * ``synthetic_comparison_<problem>_vs_evaluations.pdf`` -- one PDF **per
      problem** (not one combined grid): a single row of metric curves + raw
      product, no product-rank column (that per-iteration "who's ahead right
      now" column made sense when comparing across a shared grid; split
      one-problem-per-PDF it added nothing beyond the metric/product curves
      already shown).
    * ``synthetic_comparison_avg_rank_vs_evaluations.pdf`` -- one standalone
      **boxplot** figure: each metric's (+ product's) column shows one box per
      method summarizing that method's rank (1 = best) *distribution* across
      every problem present at this difficulty (see
      :func:`_plot_avg_rank_box_figure` and
      ``ranking.rank_lists_over_rows``), rather than collapsing that spread
      to a single averaged bar. This used to be a bottom row glued onto the
      combined grid; it is now its own PDF since there is no longer one
      combined grid for it to sit under.
    * ``synthetic_comparison_relative_auc_vs_evaluations.pdf`` -- one more
      standalone **boxplot** figure: for each metric (+ product) on each
      problem, the best AUC across every method/seed is found, every
      (method, seed) AUC is divided by that best and averaged over seeds to
      get one ratio per problem, then each method's distribution of those
      per-problem ratios (across every problem at this difficulty) is drawn
      as a box (see :func:`_plot_relative_auc_box_figure` and
      ``ranking.relative_auc_ratio_lists_over_rows``).
    * ``synthetic_comparison_normalized_avg_curve_vs_pct_budget.pdf`` -- a
      **line-plot** figure of a different shape than the two boxplots above:
      instead of collapsing each problem's curve to one scalar (an AUC, a
      rank, a ratio) before combining across problems, each metric's (+
      product's) raw curve is first normalized by this difficulty's own
      best-ever value seen anywhere (any method, any seed, any timestep --
      the same ratio-to-row-best convention the relative-AUC figures above
      already use, just applied point-wise to the whole curve instead of
      once to its AUC), then every problem's normalized curve is averaged
      point-wise on the shared "% of that problem's own evaluation budget"
      x-axis (:func:`_run_budget`; needed here specifically because
      synthetic problems' own budgets differ, 100 or 200 -- unlike
      ``ff_comparison``/``casd_comparison``, whose rows already share one
      fixed budget) -- one line + shaded IQR band per method per column, the
      band showing spread **across problems** at this difficulty (see
      :func:`_plot_normalized_avg_curve_figure` and
      :func:`normalized_curve_grid_lists_over_rows`).
    * ``synthetic_comparison_stats_report.{json,md}`` -- Friedman-gated,
      Holm-Bonferroni corrected one-sided paired Wilcoxon dominance test
      (``H1: itcas_ndig > baseline``) against each of the five baselines,
      one row per problem (see :func:`_synthetic_report_to_markdown`).

    Additionally writes one figure directly under ``<output_dir>`` (not a
    per-difficulty subfolder, since it spans every difficulty at once):

    * ``synthetic_comparison_relative_auc_by_difficulty.pdf`` -- a 2x2-grid
      line-plot sibling of the per-difficulty relative-AUC boxplot above (4
      metric panels, no product panel, horizontal legend below the grid):
      instead of showing each method's ratio distribution as a box at one
      difficulty, one line per method is drawn across difficulty levels on
      the x-axis (each line point the mean of that difficulty's per-problem
      ratios), so a method's trend as the problem gets harder/easier stays
      visible; a shaded band around each line shows the interquartile spread
      **across problems** at that difficulty (not across seeds -- each
      problem's own seeds are already averaged down to one ratio before this
      figure ever sees them, see :func:`_render_synthetic_aggregate`'s
      docstring) (see :func:`_plot_relative_auc_by_difficulty_grid_figure`).

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
    resolved_auc_cache_dir = (
        Path(auc_cache_dir) if auc_cache_dir is not None else Path(input_dir).parent / "auc_cache"
    )

    runs_by_problem = _collect_family_runs(input_path, problems, list(SYNTHETIC_METHODS))
    caches_by_problem = {
        p: _precompute_cached(runs, resolved_auc_cache_dir, p)
        for p, runs in runs_by_problem.items() if runs
    }

    summaries_by_problem: dict[str, dict[str, dict]] = {}
    paths: list[str] = []
    for problem in problems:
        runs = runs_by_problem.get(problem, [])
        if not runs:
            continue
        cache = caches_by_problem.get(problem, {})
        summary, problem_paths = _synthetic_problem_report(
            problem, runs, cache, out_dir=out_dir, auc_cache_dir=resolved_auc_cache_dir
        )
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
    auc_cache_dir: str | Path | None = None,
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
    ``auc_cache_dir`` (see :mod:`itcas.reporting.auc_cache`) defaults to
    ``Path(input_dir).parent / "auc_cache"`` when ``None``, mirroring
    :func:`summarize_synthetic_comparison`.
    """
    from .batch_vs_sequential import _collect_family_runs

    del alpha  # unused here, see docstring

    input_path = Path(input_dir)
    out_dir = Path(output_dir) if output_dir is not None else Path(_SYNTHETIC_OUTPUT_DIR)
    resolved_auc_cache_dir = (
        Path(auc_cache_dir) if auc_cache_dir is not None else Path(input_dir).parent / "auc_cache"
    )

    synthetic_problems = _synthetic_problems(problems_config)
    if problem not in synthetic_problems:
        raise ValueError(
            f"'{problem}' is not one of the synthetic problems in {problems_config} "
            f"(or is one of the excluded real-world problems: {_REAL_WORLD_PROBLEMS})"
        )

    runs = _collect_family_runs(input_path, [problem], list(SYNTHETIC_METHODS)).get(problem, [])
    if not runs:
        return []
    cache = _precompute_cached(runs, resolved_auc_cache_dir, problem)

    summary, paths = _synthetic_problem_report(
        problem, runs, cache, out_dir=out_dir, auc_cache_dir=resolved_auc_cache_dir
    )

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
    per difficulty found across those files, the same four aggregate
    outputs :func:`summarize_synthetic_comparison` writes today:
    ``synthetic_comparison_avg_rank_vs_evaluations.pdf`` (a boxplot of each
    method's rank distribution across problems),
    ``synthetic_comparison_relative_auc_vs_evaluations.pdf`` (likewise a
    boxplot of relative-AUC ratio distributions),
    ``synthetic_comparison_normalized_avg_curve_vs_pct_budget.pdf`` (each
    metric's -- plus product's -- row-best-normalized curve, averaged across
    problems on the shared "% of each problem's own evaluation budget"
    x-axis, with a shaded IQR band across problems), and
    ``synthetic_comparison_stats_report.{json,md}``, all under
    ``<output_dir>/<difficulty>/`` -- plus the same top-level
    ``synthetic_comparison_relative_auc_by_difficulty.pdf`` line-plot-plus-
    shaded-band figure (spread across problems at each difficulty, see
    :func:`summarize_synthetic_comparison`'s docstring), written directly
    under ``<output_dir>``.
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
    parser.add_argument(
        "--auc-cache-dir", type=str, default=None, dest="auc_cache_dir",
        help=(
            "Directory for the disk-backed per-seed metric-AUC cache (see "
            "itcas.reporting.auc_cache), used by --synthetic-comparison and "
            "--synthetic-comparison-problem. Defaults to <input-dir's parent>/auc_cache."
        ),
    )
    parser.add_argument(
        "--method-group", type=str, default=None, choices=["batch", "sequential"],
        dest="method_group",
        help=(
            "Which method-group comparison to run (see "
            "itcas.reporting.method_group_comparison): 'batch' compares the 5 batch "
            "methods against each other, 'sequential' the 5 sequential methods against "
            "each other. Required together with --method-group-comparison-problem or "
            "--method-group-comparison-aggregate-from."
        ),
    )
    parser.add_argument(
        "--method-group-comparison-problem", type=str, default=None,
        dest="method_group_comparison_problem",
        help=(
            "Per-problem half of the split method-group-comparison pipeline (see "
            "summarize_method_group_comparison_problem): loads only this one problem's "
            "runs for the --method-group given, and -- when --save-metrics is given -- "
            "writes <save-metrics>/<problem>_<method-group>_comparison_metrics.json for "
            "a later --method-group-comparison-aggregate-from run. Paired with "
            "--input-dir/--method-group/--save-metrics/--problems-config/--auc-cache-dir. "
            "Renders no per-problem plot -- --output-dir is unused here."
        ),
    )
    parser.add_argument(
        "--method-group-comparison-aggregate-from", type=str, default=None,
        dest="method_group_comparison_aggregate_from",
        help=(
            "Aggregate half of the split method-group-comparison pipeline (see "
            "summarize_method_group_comparison_aggregate): loads every "
            "*_<method-group>_comparison_metrics.json under this directory (written by "
            "--method-group-comparison-problem runs) and produces the combined "
            "relative-AUC/normalized-curve figures for that group. Paired with "
            "--output-dir/--method-group. Does not require --input-dir and never reads "
            "any run logs."
        ),
    )
    parser.add_argument(
        "--ndig-kernel-ablation-problem", type=str, default=None,
        dest="ndig_kernel_ablation_problem",
        help=(
            "Per-problem half of the split NDIG QD-DPP diversity-kernel "
            "ablation pipeline (see itcas.reporting.ndig_kernel_ablation_comparison."
            "summarize_ndig_kernel_ablation_problem): compares the proposed method's "
            "full batch NDIG acquisition (itcas_ndig) against its two kernel-ablated "
            "siblings (ndig_no_kobj_batch/ndig_no_kctx_batch). Loads only this one "
            "problem's runs and -- when --save-metrics is given -- writes "
            "<save-metrics>/<problem>_ndig_kernel_ablation_metrics.json for a later "
            "--ndig-kernel-ablation-aggregate-from run. Paired with --input-dir/"
            "--save-metrics/--problems-config/--auc-cache-dir. Renders no per-problem "
            "plot -- --output-dir is unused here."
        ),
    )
    parser.add_argument(
        "--ndig-kernel-ablation-aggregate-from", type=str, default=None,
        dest="ndig_kernel_ablation_aggregate_from",
        help=(
            "Aggregate half of the split NDIG QD-DPP diversity-kernel ablation "
            "pipeline (see summarize_ndig_kernel_ablation_aggregate): loads every "
            "*_ndig_kernel_ablation_metrics.json under this directory (written by "
            "--ndig-kernel-ablation-problem runs) and produces the combined "
            "relative-AUC-by-difficulty and normalized-curve figures. Paired with "
            "--output-dir. Does not require --input-dir and never reads any run logs."
        ),
    )
    args = parser.parse_args(argv)

    kw = dict(alpha=args.alpha)

    if args.ndig_kernel_ablation_problem is not None:
        from .ndig_kernel_ablation_comparison import summarize_ndig_kernel_ablation_problem

        if args.input_dir is None:
            parser.error("--input-dir is required for --ndig-kernel-ablation-problem")
        paths = summarize_ndig_kernel_ablation_problem(
            args.input_dir, args.ndig_kernel_ablation_problem,
            problems_config=args.problems_config, output_dir=args.output_dir,
            save_metrics_dir=args.save_metrics, auc_cache_dir=args.auc_cache_dir,
        )
        for p in paths:
            print(p)
        return 0

    if args.ndig_kernel_ablation_aggregate_from is not None:
        from .ndig_kernel_ablation_comparison import summarize_ndig_kernel_ablation_aggregate

        paths = summarize_ndig_kernel_ablation_aggregate(
            args.ndig_kernel_ablation_aggregate_from, output_dir=args.output_dir,
        )
        for p in paths:
            print(p)
        return 0

    if args.method_group_comparison_problem is not None:
        from .method_group_comparison import summarize_method_group_comparison_problem

        if args.input_dir is None:
            parser.error("--input-dir is required for --method-group-comparison-problem")
        if args.method_group is None:
            parser.error("--method-group is required for --method-group-comparison-problem")
        paths = summarize_method_group_comparison_problem(
            args.input_dir, args.method_group_comparison_problem, args.method_group,
            problems_config=args.problems_config, output_dir=args.output_dir,
            save_metrics_dir=args.save_metrics, auc_cache_dir=args.auc_cache_dir,
        )
        for p in paths:
            print(p)
        return 0

    if args.method_group_comparison_aggregate_from is not None:
        from .method_group_comparison import summarize_method_group_comparison_aggregate

        if args.method_group is None:
            parser.error("--method-group is required for --method-group-comparison-aggregate-from")
        paths = summarize_method_group_comparison_aggregate(
            args.method_group_comparison_aggregate_from, args.method_group,
            output_dir=args.output_dir,
        )
        for p in paths:
            print(p)
        return 0

    if args.synthetic_comparison_problem is not None:
        if args.input_dir is None:
            parser.error("--input-dir is required for --synthetic-comparison-problem")
        paths = summarize_synthetic_comparison_problem(
            args.input_dir, args.synthetic_comparison_problem,
            problems_config=args.problems_config, output_dir=args.output_dir,
            save_metrics_dir=args.save_metrics, auc_cache_dir=args.auc_cache_dir, **kw,
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
            output_dir=args.output_dir, auc_cache_dir=args.auc_cache_dir, **kw,
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
