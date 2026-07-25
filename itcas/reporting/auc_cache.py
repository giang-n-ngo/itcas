"""Disk-backed cache of per-seed metric AUCs, keyed by run.

Every report in this package that ranks methods (``ranking.py``) or tests
dominance (``batch_vs_sequential._collect_product_auc_for_stats``) only ever
consumes a per-``(run, axis, metric)`` **scalar** -- the area under that
run's metric curve, integrated via :func:`itcas.reporting.summary._curve_area`
-- never the raw per-iteration curve itself. Only the per-difficulty grid
PDFs (:func:`itcas.reporting.batch_vs_sequential.plot_group_grid`) need the
full curve.

Computing that scalar is cheap (a single trapezoidal pass); what's expensive
is producing the curve it integrates in the first place
(:func:`itcas.reporting.summary._precompute`, which calls
:func:`itcas.reporting.metrics.compute_metric` once per run per metric --
``feasible_convex_hull_volume``/``feasible_context_fill_distance`` are
genuinely slow, see ``itcas/metrics/metrics.py``). This module persists the
*already-integrated* scalars to disk so a later report invocation that only
needs the scalars (not the curves) can skip ``compute_metric``/``_precompute``
entirely for any run/metric/axis already cached.

This module is intentionally decoupled from any particular report: it knows
nothing about "rows", "problems", or "difficulty levels" beyond using
``summary._difficulty_of`` to tag each cached run for human debugging. Any
caller that already has ``runs``/``curve_cache`` (from
``summary._precompute``) can build and persist an :data:`AUCTable`; any
caller that wants to skip curve computation can load one first and pass it
through to ``ranking.py``'s ``auc_cache=`` parameter.

This module also persists the raw curves themselves (``save_curve_cache``/
``load_curve_cache``/``load_curve_cache_for_problem``, see the "Curve cache"
section below) so that ``_precompute``/``compute_metric`` -- the actual
expensive step the AUC cache above never touches, since it still needs the
raw curve for the grid PDFs -- can also be skipped on a hit; see
:func:`itcas.reporting.summary._precompute_cached`.

Import direction: this module imports from :mod:`itcas.reporting.summary` at
module level. ``summary.py`` must never import this module at its own module
level (only lazily inside function bodies), or the two modules would form an
import cycle.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

from .metrics import RunSeries, eps_archive_used
from .summary import (
    CurveCache,
    _compute_seed_product_curve,
    _curve_area,
    _difficulty_of,
    _ordered_metrics,
)

# run_name -> axis ("evals"|"steps") -> metric_key (a MetricSpec.key, or
# "product") -> AUC
AUCTable = dict[str, dict[str, dict[str, float]]]


def _finite_or_none(value: Optional[float]) -> Optional[float]:
    """Return ``value`` unless it's ``None``, NaN, or +/-Inf (then ``None``).

    Guards :func:`compute_auc_table`'s output against sneaking a non-finite
    float into the JSON cache: ``json.dump`` either raises
    (``allow_nan=False``) or silently emits the invalid ``NaN``/``Infinity``
    tokens (the default), neither of which is acceptable for a file meant to
    be reloaded by a strict JSON parser.
    """
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    return value


def compute_auc_table(
    runs: list[RunSeries],
    curve_cache: CurveCache,
    axes: tuple[str, ...] = ("evals", "steps"),
) -> AUCTable:
    """Integrate every run's already-computed curves into one :data:`AUCTable`.

    Cheap: ``curve_cache`` must already be built (via
    ``summary._precompute``); this only integrates it
    (``summary._curve_area``) per run/axis/metric, plus the raw point-wise
    product curve (``summary._compute_seed_product_curve``) under the
    ``"product"`` key -- mirrors exactly what ``ranking.py``'s per-row loops
    already compute inline, just materialized once per run instead of
    recomputed by every caller.

    Entries with no data (curve is ``None``, or ``_curve_area``/the product
    AUC is ``None`` or non-finite) are omitted from the output entirely,
    never stored as ``null``/``NaN``.
    """
    metrics = _ordered_metrics()
    table: AUCTable = {}
    for run in runs:
        run_curves = curve_cache.get(run.run_name) or {}
        per_axis: dict[str, dict[str, float]] = {}
        for axis in axes:
            x_axis = run.x_evals if axis == "evals" else run.x_steps
            per_metric: dict[str, float] = {}
            for spec in metrics:
                y = run_curves.get(spec.key)
                if y is None:
                    continue
                auc = _finite_or_none(_curve_area(x_axis, y))
                if auc is not None:
                    per_metric[spec.key] = auc

            prod = _compute_seed_product_curve(run, axis, run_curves, metrics)
            if prod is not None:
                auc = _finite_or_none(_curve_area(x_axis, prod))
                if auc is not None:
                    per_metric["product"] = auc

            if per_metric:
                per_axis[axis] = per_metric
        if per_axis:
            table[run.run_name] = per_axis
    return table


def _cache_path(cache_dir: str | Path, problem: str) -> Path:
    return Path(cache_dir) / f"{problem}_auc_cache.json"


def save_auc_cache(
    cache_dir: str | Path,
    problem: str,
    runs: list[RunSeries],
    auc_table: AUCTable,
) -> Path:
    """Write/merge ``<cache_dir>/<problem>_auc_cache.json``.

    File shape::

        {"problem": ..., "runs": {run_name: {"method": ..., "seed": ...,
         "difficulty": ..., "eps_archive_used": ...,
         "auc": {axis: {metric_key: value}}}}}

    ``method``/``seed``/``difficulty`` come from the matching entry in
    ``runs`` (``seed``/``method`` directly; ``difficulty`` via
    ``summary._difficulty_of``). ``eps_archive_used``
    (``itcas.reporting.metrics.eps_archive_used``) is the ``eps`` value
    actually in effect for this run *right now* -- stamped fresh on every
    save regardless of whether ``epsilon_archive_size``/``product`` are
    among the axis_map's keys for this call, so a later
    :func:`load_auc_cache` can tell whether this entire entry (every metric,
    not just the two whose AUC depends on ``eps``) was cached under a
    since-changed ``configs/experiments.json`` calibration and must be
    dropped -- see that function's docstring. If the file already exists,
    entries are MERGED at the ``run_name`` level (loaded, updated only for
    the ``run_name``\\ s present in ``auc_table``, written back) rather than
    the whole file being overwritten -- a second report script populating
    the same problem's cache with a different method subset must not clobber
    entries the first script already wrote for other methods.
    """
    path = _cache_path(cache_dir, problem)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_runs: dict[str, dict] = {}
    if path.exists():
        try:
            with path.open() as f:
                existing = json.load(f)
            if isinstance(existing, dict) and isinstance(existing.get("runs"), dict):
                existing_runs = existing["runs"]
        except (json.JSONDecodeError, OSError):
            existing_runs = {}

    run_by_name = {r.run_name: r for r in runs}
    for run_name, axis_map in auc_table.items():
        if not axis_map:
            continue
        run = run_by_name.get(run_name)
        existing_runs[run_name] = {
            "method": run.method if run is not None else None,
            "seed": run.seed if run is not None else None,
            "difficulty": _difficulty_of(run) if run is not None else None,
            "eps_archive_used": eps_archive_used(run) if run is not None else None,
            "auc": axis_map,
        }

    with path.open("w") as f:
        json.dump({"problem": problem, "runs": existing_runs}, f)
    return path


def load_auc_cache(path: str | Path, runs: Optional[list[RunSeries]] = None) -> AUCTable:
    """Load one ``<problem>_auc_cache.json``, returning just the AUC mapping.

    Returns ``{run_name: {axis: {metric_key: auc}}}`` -- the metadata fields
    written by :func:`save_auc_cache` (``method``/``seed``/``difficulty``)
    are for human debugging / that function's own merge step, not needed by
    callers wanting to look up an AUC.

    ``runs``, if given, enables the same eps-recalibration staleness check
    :func:`itcas.reporting.summary._precompute_cached` applies to the curve
    cache: an on-disk entry whose stored ``eps_archive_used`` no longer
    matches what's currently resolved (``itcas.reporting.metrics.
    eps_archive_used``) for the matching run in ``runs`` was cached under a
    since-changed ``configs/experiments.json`` calibration and is dropped in
    its entirety -- not just its ``epsilon_archive_size``/``product`` AUCs --
    matching the curve cache's "no partial top-up" policy, since this AUC
    table is a *separate* on-disk cache that can go stale independently of
    the curve cache (``ranking.py``'s ``_lookup_or_compute_auc`` prefers it
    over recomputing from an already-fresh curve cache). A run present on
    disk but absent from ``runs`` is left alone (nothing to compare against);
    pass ``runs=None`` (the default) to skip this check entirely and load
    every cached entry as-is.
    """
    with Path(path).open() as f:
        data = json.load(f)
    runs_data = data.get("runs", {}) if isinstance(data, dict) else {}
    run_by_name = {r.run_name: r for r in runs} if runs is not None else {}
    out: AUCTable = {}
    for run_name, entry in runs_data.items():
        if not isinstance(entry, dict):
            continue
        run = run_by_name.get(run_name)
        if run is not None and entry.get("eps_archive_used") != eps_archive_used(run):
            continue  # stale -- configs/experiments.json's eps_archive changed since this was cached
        auc = entry.get("auc")
        if auc:
            out[run_name] = auc
    return out


def load_auc_cache_for_problem(
    cache_dir: str | Path, problem: str, runs: Optional[list[RunSeries]] = None
) -> AUCTable:
    """Return ``{}`` if this problem's cache file doesn't exist yet.

    First-run/cache-miss case -- nothing has been cached for ``problem``
    yet -- instead of raising ``FileNotFoundError``. ``runs`` is forwarded to
    :func:`load_auc_cache` for the eps-recalibration staleness check; see its
    docstring.
    """
    path = _cache_path(cache_dir, problem)
    if not path.exists():
        return {}
    return load_auc_cache(path, runs=runs)


# ---------------------------------------------------------------------------
# Curve cache: the raw per-iteration curves themselves (not just their AUCs).
#
# Unlike the AUC cache above -- which only ever helps a caller that consumes
# scalars (``ranking.py``) -- this one lets ``summary._precompute`` itself be
# skipped on a hit, since the grid PDFs (``plot_group_grid``) need the full
# curve, not just its integral. See ``summary._precompute_cached``.
# ---------------------------------------------------------------------------
# run_name -> metric_key -> curve (list[float]) or None (computed, no data
# for this run -- e.g. FCFD on a problem with no context dims).
CurveCacheEntry = dict[str, Optional[list[float]]]


def _curve_cache_path(cache_dir: str | Path, problem: str) -> Path:
    return Path(cache_dir) / f"{problem}_curve_cache.json"


def save_curve_cache(
    cache_dir: str | Path,
    problem: str,
    runs: list[RunSeries],
    curve_cache: CurveCache,
) -> Path:
    """Write/merge ``<cache_dir>/<problem>_curve_cache.json``.

    File shape::

        {"problem": ..., "runs": {run_name: {"method": ..., "seed": ...,
         "difficulty": ..., "curves": {metric_key: [values] | null}}}}

    Same merge-by-``run_name`` semantics as :func:`save_auc_cache` (load
    existing, update only the ``run_name``\\ s present in ``curve_cache``,
    write back) -- a second report script populating a different method
    subset must not clobber entries the first script already wrote.

    Unlike :func:`compute_auc_table`'s output, a ``None`` curve (a metric
    genuinely has no data for this run, e.g. FCFD on a non-context problem)
    is written explicitly as JSON ``null`` rather than the key being omitted
    -- :func:`summary._precompute_cached`'s completeness check needs to tell
    "computed, no data" apart from "never computed", so every one of
    ``_ordered_metrics()``'s keys must be present for every cached run.
    Likewise, NaN values *inside* a curve's own list are left as-is (not
    sanitized the way :func:`_finite_or_none` sanitizes AUC scalars) --
    dropping a mid-list element would desync the curve positionally from
    ``run.x_evals``/``run.x_steps``. Python's ``json`` module round-trips
    ``NaN``/``Infinity`` tokens fine by default (non-standard JSON, but this
    is an internal-only cache with no cross-language consumer).
    """
    path = _curve_cache_path(cache_dir, problem)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_runs: dict[str, dict] = {}
    if path.exists():
        try:
            with path.open() as f:
                existing = json.load(f)
            if isinstance(existing, dict) and isinstance(existing.get("runs"), dict):
                existing_runs = existing["runs"]
        except (json.JSONDecodeError, OSError):
            existing_runs = {}

    run_by_name = {r.run_name: r for r in runs}
    for run_name, curves in curve_cache.items():
        run = run_by_name.get(run_name)
        existing_runs[run_name] = {
            "method": run.method if run is not None else None,
            "seed": run.seed if run is not None else None,
            "difficulty": _difficulty_of(run) if run is not None else None,
            "curves": curves,
        }

    with path.open("w") as f:
        json.dump({"problem": problem, "runs": existing_runs}, f)
    return path


def load_curve_cache(path: str | Path) -> CurveCache:
    """Load one ``<problem>_curve_cache.json``, returning just the curve mapping.

    Returns ``{run_name: {metric_key: [values] | None}}`` -- the metadata
    fields written by :func:`save_curve_cache` (``method``/``seed``/
    ``difficulty``) are for human debugging / that function's own merge step,
    not needed by callers wanting to look up a curve.
    """
    with Path(path).open() as f:
        data = json.load(f)
    runs = data.get("runs", {}) if isinstance(data, dict) else {}
    out: CurveCache = {}
    for run_name, entry in runs.items():
        if not isinstance(entry, dict):
            continue
        curves = entry.get("curves")
        if isinstance(curves, dict):
            out[run_name] = curves
    return out


def load_curve_cache_for_problem(cache_dir: str | Path, problem: str) -> CurveCache:
    """Return ``{}`` if this problem's curve cache file doesn't exist yet.

    First-run/cache-miss case -- nothing has been cached for ``problem``
    yet -- instead of raising ``FileNotFoundError``.
    """
    path = _curve_cache_path(cache_dir, problem)
    if not path.exists():
        return {}
    return load_curve_cache(path)
