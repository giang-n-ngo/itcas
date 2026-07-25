"""Method-group comparisons: batch methods against each other, sequential
methods against each other.

Two narrower siblings of :mod:`itcas.reporting.summary`'s synthetic-
comparison pipeline (``summarize_synthetic_comparison*``, which pits the one
proposed batch method against 5 forced-sequential baselines): instead of one
6-method table, this module compares the 5 **batch** methods among
themselves, and separately the 5 **sequential** methods among themselves --
using the exact same ``(family_key, sequential_method, batch_method,
display_label)`` pairing :mod:`itcas.reporting.batch_improvement_comparison`
already defines (``FAMILIES``, imported from there rather than re-typed here,
so the two modules can never drift apart):

    family        sequential                  batch                       label
    itcas         itcas_seq_ndig               itcas_ndig                  NDIG
    bes           bes_then_sample_lse10        bes_then_sample_lse10_batch BES-TS-10
    straddle      straddle_then_sample_lse10   straddle_then_sample_lse10_batch STR-TS-10
    eci           cas_eci                      cas_eci_batch               ECI
    moc_cas_hard  moc_cas_hard                 moc_cas_hard_batch          MOC-CAS

``random`` has no batch/sequential pair and is excluded from both groups --
same reasoning ``batch_improvement_comparison.py`` already gives for
excluding it there.

**Scope, narrower than the synthetic pipeline in three ways:**

1. **No per-problem plots.** ``summary._synthetic_problem_report`` renders a
   ``synthetic_comparison_<problem>_vs_evaluations.pdf`` grid per problem
   (via ``batch_vs_sequential.plot_group_grid``); this module never does --
   :func:`_method_group_problem_report` below only ever returns a
   JSON-serializable summary, no ``out_dir``/paths-written list.
2. **No avg-rank, no stats report.** Only the *relative-AUC* and
   *normalized-curve* figure families are produced -- the exact three
   ``summary._plot_relative_auc_box_figure`` /
   ``summary._plot_relative_auc_by_difficulty_figure`` /
   ``summary._plot_normalized_avg_curve_figure`` calls
   ``summary._render_synthetic_aggregate`` also makes, just restricted to one
   group's own 5 methods -- never the ``summary._plot_avg_rank_box_figure``
   boxplot, and never a Friedman/Wilcoxon ``run_stats``/``report_to_json``/
   markdown report. This mirrors ``batch_improvement_comparison.py``'s own
   "No statistics" scope decision (see its module docstring): a deliberate
   cut, not an oversight -- a stats report could be added later as a
   companion output if wanted.
3. **Ratios/curves are relative to the group's own best, not the other
   group's.** Every ``ranking.relative_auc_ratios_over_rows`` /
   ``summary.normalized_curve_grid_lists_over_rows`` call below is fed only
   that group's own 5-method list (never the union of both groups), so the
   "row best" each ratio/curve normalizes against is always found among that
   group's own methods -- see ``ranking.relative_auc_ratios_over_rows``'s and
   ``summary._row_normalized_method_curves``'s own docstrings for why
   restricting the ``methods`` argument is sufficient to restrict the pool.

Otherwise this reuses the synthetic pipeline's scope verbatim: the same
``summary._synthetic_problems()`` problem suite (every problem in
``configs/final_problems.json`` except the two real-world problems,
``spacecraft_formation_flying_a1``/``casd_llm``), the same shared
``p0_01``/``p0_05``/``p0_10``/``p0_20`` difficulty scale, evaluations axis
only (``RunSeries.x_evals`` -- the only fair shared axis between a method's
sequential and batch variant, see ``batch_vs_sequential.py``'s module
docstring), and the same disk-backed curve/AUC caching conventions
(``auc_cache_dir`` defaulting to ``Path(input_dir).parent / "auc_cache"``).

**Split, memory-bounded pipeline.** Exactly like
``summary.summarize_synthetic_comparison_problem`` /
``summarize_synthetic_comparison_aggregate`` (see that module's section
docstring on why the monolithic form OOMs on the full sweep -- job 3427762,
noted in ``scripts/synthetic_comparison.sbatch``'s header comment), this
module only ever exposes the two-stage split form: a **per-problem** stage
(:func:`summarize_method_group_comparison_problem`, loads one problem's runs
only, writes ``<problem>_<group>_comparison_metrics.json``) and an
**aggregate** stage (:func:`summarize_method_group_comparison_aggregate`,
reads only those small JSON files, never a raw run log). There is no
monolithic ``summarize_method_group_comparison`` counterpart here -- unlike
``summary.summarize_synthetic_comparison``, which still exists for small
local runs, this module was written after that lesson was already learned,
so only the split form is offered.

**CLI.** Reachable via ``python -m itcas.summarize`` alongside the
``--synthetic-comparison-*`` flags (see ``summary.main``), not its own
``python -m`` entry point -- ``itcas.summarize`` already wires in every
Slurm-invoked pipeline in this package, so a second parallel CLI here would
just be a second place to keep in sync.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from .batch_improvement_comparison import FAMILIES
from .metrics import RunSeries
from .method_labels import METHOD_ABBREVIATIONS
from .summary import SYNTHETIC_PROPOSED_METHOD, CurveCache, _SYNTHETIC_METHOD_STYLES

Group = Literal["batch", "sequential"]

_DEFAULT_PROBLEMS_CONFIG = "configs/final_problems.json"
_AXIS = "evals"  # evaluations only -- see module docstring.

# {group: (method, ...)} -- proposed method first, derived from FAMILIES so
# this can never drift from batch_improvement_comparison.py's own table.
# `random` is excluded from both (see module docstring).
METHOD_GROUPS: dict[str, tuple[str, ...]] = {
    "batch": tuple(batch for _fam, _seq, batch, _label in FAMILIES),
    "sequential": tuple(seq for _fam, seq, _batch, _label in FAMILIES),
}

PROPOSED_METHOD: dict[str, str] = {
    "batch": FAMILIES[0][2],
    "sequential": FAMILIES[0][1],
}

_DEFAULT_OUTPUT_DIR: dict[str, str] = {
    "batch": "results/batch_comparison",
    "sequential": "results/sequential_comparison",
}

# One color per family, reused verbatim from summary._SYNTHETIC_METHOD_STYLES
# (keeping this report's palette identical to every other report's for the
# same method/family, see module docstring) -- never invented fresh here.
# `itcas` maps to itcas_ndig's color there since that's the proposed method's
# synthetic-comparison key; the other four family keys already match
# _SYNTHETIC_METHOD_STYLES's own baseline keys verbatim (those are always the
# *sequential* name, e.g. "cas_eci", "moc_cas_hard").
_FAMILY_STYLE_SOURCE: dict[str, dict] = {
    "itcas": _SYNTHETIC_METHOD_STYLES[SYNTHETIC_PROPOSED_METHOD],
    "bes": _SYNTHETIC_METHOD_STYLES["bes_then_sample_lse10"],
    "straddle": _SYNTHETIC_METHOD_STYLES["straddle_then_sample_lse10"],
    "eci": _SYNTHETIC_METHOD_STYLES["cas_eci"],
    "moc_cas_hard": _SYNTHETIC_METHOD_STYLES["moc_cas_hard"],
}

# {group: {method: style}} -- same shape as summary._SYNTHETIC_METHOD_STYLES,
# restricted to one group's own 5 methods and keyed by that group's own
# method names (batch names for "batch", sequential names for "sequential").
METHOD_STYLES: dict[str, dict[str, dict]] = {
    "batch": {batch: _FAMILY_STYLE_SOURCE[fam] for fam, _seq, batch, _label in FAMILIES},
    "sequential": {seq: _FAMILY_STYLE_SOURCE[fam] for fam, seq, _batch, _label in FAMILIES},
}


def _validate_group(group: str) -> None:
    if group not in METHOD_GROUPS:
        raise ValueError(f"group must be one of {sorted(METHOD_GROUPS)}, got {group!r}")


# ---------------------------------------------------------------------------
# Per-problem report + JSON round-trip (mirrors summary._synthetic_problem_report
# / save_synthetic_problem_metrics / load_synthetic_problem_metrics, trimmed).
# ---------------------------------------------------------------------------
def _method_group_problem_report(
    problem: str,
    runs: list[RunSeries],
    cache: CurveCache,
    methods: list[str],
    *,
    auc_cache_dir: str | Path | None = None,
) -> dict[str, dict]:
    """One problem's contribution to a method-group comparison, per difficulty.

    Trimmed sibling of ``summary._synthetic_problem_report``: computes only
    ``relative_auc`` (:func:`itcas.reporting.ranking.relative_auc_ratios_over_rows`)
    and ``normalized_curve_lists``
    (:func:`itcas.reporting.summary.normalized_curve_grid_lists_over_rows`) --
    no ``avg_rank`` (that only ever fed the now-out-of-scope avg-rank
    boxplot), no rendered PDF (there is no ``out_dir`` parameter here at all
    -- this report never renders a per-problem plot, see module docstring),
    and no ``auc`` entry either (that only ever fed the now-out-of-scope
    stats report). ``methods`` is one group's own 5-method list (see
    :data:`METHOD_GROUPS`): passing only those methods into both ``ranking``/
    ``summary`` calls below means every ratio/curve is computed relative to
    the best value found *within that group*, never pooled against the other
    group's methods (see module docstring point 3).

    Uses only ``runs``/``cache`` for this single ``problem`` -- never touches
    any other problem's data -- so this is safe to call from a per-problem
    job holding just one problem's ``RunSeries``/``CurveCache`` in memory.

    When ``auc_cache_dir`` is given (see :mod:`itcas.reporting.auc_cache`),
    the ``relative_auc`` call is fed that directory's already-cached per-seed
    AUCs so it can skip recomputing ``summary._curve_area`` for any ``(run,
    axis, metric)`` already on disk; newly computed AUCs for this problem's
    own runs are merged back in before returning -- exactly mirroring
    ``_synthetic_problem_report``'s own caching contract.

    Returns ``{difficulty: {"metrics_present": [metric_key, ...],
    "relative_auc": {column_key: {method: ratio}}, "normalized_curve_lists":
    {column_key: {method: [row_curve]}}}}`` -- fully JSON-serializable (see
    :func:`save_method_group_problem_metrics`) and exactly what
    :func:`_combine_method_group_summaries` expects as one entry of its
    ``summaries_by_problem`` argument.
    """
    from .batch_vs_sequential import _rows_by_problem
    from .ranking import relative_auc_ratios_over_rows
    from .summary import _difficulty_of, _metrics_present_in_rows, normalized_curve_grid_lists_over_rows

    runs_by_problem = {problem: runs}
    caches_by_problem = {problem: cache}

    existing_auc_cache = {}
    if runs and auc_cache_dir is not None:
        from .auc_cache import load_auc_cache_for_problem

        existing_auc_cache = load_auc_cache_for_problem(auc_cache_dir, problem, runs=runs)

    diffs = sorted({_difficulty_of(r) for r in runs})
    summary: dict[str, dict] = {}
    for diff in diffs:
        rows = _rows_by_problem([problem], runs_by_problem, caches_by_problem, diff)
        if not rows:
            continue

        metrics_present = _metrics_present_in_rows(rows)
        relative_auc = relative_auc_ratios_over_rows(
            rows, methods, _AXIS, auc_cache=existing_auc_cache
        )
        # Discard the returned pct_grid -- see _synthetic_problem_report's
        # docstring for why it's recomputed at render time instead of
        # persisted here (a pure function of n_grid_points, no data
        # dependency).
        normalized_curve_lists = normalized_curve_grid_lists_over_rows(rows, methods)[1]
        summary[diff] = {
            "metrics_present": [s.key for s in metrics_present],
            "relative_auc": relative_auc,
            "normalized_curve_lists": normalized_curve_lists,
        }

    if runs and auc_cache_dir is not None:
        from .auc_cache import compute_auc_table, save_auc_cache

        save_auc_cache(auc_cache_dir, problem, runs, compute_auc_table(runs, cache))

    return summary


def save_method_group_problem_metrics(
    metrics_dir: str | Path,
    problem: str,
    group: str,
    summary_by_diff: dict[str, dict],
) -> Path:
    """Write one problem's :func:`_method_group_problem_report` summary to JSON.

    Mirrors :func:`summary.save_synthetic_problem_metrics`, applied to this
    module's trimmed summary shape. ``group`` is stashed in the payload so
    :func:`load_method_group_problem_metrics` can validate a later aggregate
    run is reading the group it thinks it is (see that function's
    docstring) -- catches an operator accidentally pointing the aggregate
    step at the wrong group's metrics dir.
    """
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / f"{problem}_{group}_comparison_metrics.json"
    with out_path.open("w") as f:
        json.dump({"problem": problem, "group": group, "difficulties": summary_by_diff}, f)
    return out_path


def load_method_group_problem_metrics(
    path: str | Path, expected_group: str
) -> tuple[str, dict[str, dict]]:
    """Load one problem's summary written by :func:`save_method_group_problem_metrics`.

    Raises ``ValueError`` if the file's stored ``"group"`` doesn't match
    ``expected_group`` -- guards against an aggregate run accidentally
    pointed at a metrics directory written for the other group (batch vs.
    sequential metrics files can otherwise look identical: same
    ``<problem>_..._comparison_metrics.json`` shape, same problem names).

    Returns ``(problem_name, summary_by_diff)`` -- see
    :func:`_method_group_problem_report` for the shape of ``summary_by_diff``.
    """
    with Path(path).open() as f:
        data = json.load(f)
    stored_group = data.get("group")
    if stored_group != expected_group:
        raise ValueError(
            f"'{path}' was written for group '{stored_group}', expected '{expected_group}' -- "
            "refusing to mix batch and sequential metrics into one aggregate."
        )
    return data["problem"], data.get("difficulties", {})


# ---------------------------------------------------------------------------
# Cross-problem combination + rendering (mirrors
# summary._combine_synthetic_summaries / _render_synthetic_aggregate, trimmed
# to only relative_auc_lists / normalized_curve_lists).
# ---------------------------------------------------------------------------
def _combine_method_group_summaries(
    summaries_by_problem: dict[str, dict[str, dict]],
) -> dict[str, dict]:
    """Combine every problem's :func:`_method_group_problem_report` summary.

    Trimmed sibling of ``summary._combine_synthetic_summaries``: accumulates
    only ``relative_auc_lists``/``normalized_curve_lists`` (no
    ``avg_rank_lists``/``auc_by_problem`` -- those only ever fed the
    now-out-of-scope avg-rank plot and stats report, see module docstring).

    Returns ``{difficulty: {"metrics_present": [MetricSpec, ...],
    "relative_auc_lists": {column_key: {method: [ratio, ...]}}, "n_rows":
    int, "normalized_curve_lists": {column_key: {method: [row_curve,
    ...]}}}}`` -- ``n_rows`` is the number of problems present at that
    difficulty, exactly mirroring ``_combine_synthetic_summaries``'s own
    ``n_rows``.
    """
    from .summary import _ordered_metrics

    diffs: set[str] = set()
    for diff_map in summaries_by_problem.values():
        diffs.update(diff_map.keys())

    out: dict[str, dict] = {}
    for diff in sorted(diffs):
        metric_keys: set[str] = set()
        ratio_lists: dict[str, dict[str, list[float]]] = {}
        curve_lists: dict[str, dict[str, list[list[float]]]] = {}
        n_rows = 0
        for _problem, diff_map in summaries_by_problem.items():
            entry = diff_map.get(diff)
            if entry is None:
                continue
            n_rows += 1
            metric_keys.update(entry.get("metrics_present") or [])
            for column_key, method_vals in (entry.get("relative_auc") or {}).items():
                for method, val in method_vals.items():
                    ratio_lists.setdefault(column_key, {}).setdefault(method, []).append(val)
            for column_key, method_curves in (entry.get("normalized_curve_lists") or {}).items():
                for method, curves in method_curves.items():
                    curve_lists.setdefault(column_key, {}).setdefault(method, []).extend(curves)

        metrics_present = [s for s in _ordered_metrics() if s.key in metric_keys]
        out[diff] = {
            "metrics_present": metrics_present,
            "relative_auc_lists": ratio_lists,
            "n_rows": n_rows,
            "normalized_curve_lists": curve_lists,
        }
    return out


def _render_method_group_aggregate(
    summaries_by_problem: dict[str, dict[str, dict]],
    group: str,
    output_dir: str | Path,
) -> list[str]:
    """Render the cross-problem outputs from combined per-problem summaries.

    Shared by both stages the same way ``summary._render_synthetic_aggregate``
    is (only :func:`summarize_method_group_comparison_aggregate` calls this
    today, since there is no monolithic form here -- see module docstring).
    Writes, restricted to ``group``'s own 5 methods:

    * ``<output_dir>/<difficulty>/<group>_comparison_relative_auc_vs_evaluations.pdf``
      -- a boxplot of each method's relative-AUC ratio distribution across
      the problems present at that difficulty (see
      :func:`summary._plot_relative_auc_box_figure`).
    * ``<output_dir>/<group>_comparison_relative_auc_by_difficulty.pdf`` --
      one top-level line-plot figure (not per-difficulty) spanning every
      difficulty, one line + IQR band per method (see
      :func:`summary._plot_relative_auc_by_difficulty_figure`); the band
      shows spread across problems at each difficulty, same convention as
      the synthetic pipeline's own by-difficulty figure.
    * ``<output_dir>/<group>_comparison_normalized_avg_curve_vs_pct_budget.pdf``
      -- one more top-level figure, each metric's (+ product's) raw curve
      normalized to its own (problem, difficulty) row's best-within-group
      value and averaged on the shared "% of that row's own evaluation
      budget" x-axis (see :func:`summary._plot_normalized_avg_curve_figure`).

    Deliberately does **not** call ``summary._plot_avg_rank_box_figure`` or
    produce any ``run_stats``/``report_to_json``/markdown stats report -- see
    module docstring.
    """
    from .summary import (
        _default_pct_grid,
        _feasible_pct_label,
        _ordered_metrics,
        _plot_normalized_avg_curve_figure,
        _plot_relative_auc_box_figure,
        _plot_relative_auc_by_difficulty_figure,
    )

    out_dir = Path(output_dir)
    combined = _combine_method_group_summaries(summaries_by_problem)
    methods = list(METHOD_GROUPS[group])
    styles = METHOD_STYLES[group]

    paths: list[str] = []
    for diff, entry in combined.items():
        diff_dir = out_dir / diff
        metrics_present = entry["metrics_present"]
        n_rows = entry["n_rows"]

        relative_auc_path = diff_dir / f"{group}_comparison_relative_auc_vs_evaluations.pdf"
        ok = _plot_relative_auc_box_figure(
            entry["relative_auc_lists"], methods, styles,
            metrics_present, n_rows, relative_auc_path,
            method_labels=METHOD_ABBREVIATIONS, x_axis_label="Avg relative AUC",
        )
        if ok is not None:
            paths.append(str(ok))

    if combined:
        relative_auc_by_level = [
            (_feasible_pct_label(diff), combined[diff]["relative_auc_lists"])
            for diff in sorted(combined)
        ]
        metric_keys_all = {s.key for e in combined.values() for s in e["metrics_present"]}
        metrics_present_all = [s for s in _ordered_metrics() if s.key in metric_keys_all]

        by_diff_path = out_dir / f"{group}_comparison_relative_auc_by_difficulty.pdf"
        ok = _plot_relative_auc_by_difficulty_figure(
            relative_auc_by_level, methods, styles, metrics_present_all, by_diff_path,
            method_labels=METHOD_ABBREVIATIONS,
            x_axis_label="Proportion of the feasible set (i.e., difficulty level)",
        )
        if ok is not None:
            paths.append(str(ok))

        normalized_curve_lists: dict[str, dict[str, list[list[float]]]] = {}
        for entry in combined.values():
            for column_key, method_curves in (entry.get("normalized_curve_lists") or {}).items():
                for method, curves in method_curves.items():
                    normalized_curve_lists.setdefault(column_key, {}).setdefault(method, []).extend(curves)

        normalized_curve_path = (
            out_dir / f"{group}_comparison_normalized_avg_curve_vs_pct_budget.pdf"
        )
        ok = _plot_normalized_avg_curve_figure(
            _default_pct_grid(), normalized_curve_lists, methods, styles,
            metrics_present_all, normalized_curve_path,
            method_labels=METHOD_ABBREVIATIONS,
        )
        if ok is not None:
            paths.append(str(ok))

    return paths


# ---------------------------------------------------------------------------
# Public API -- two-stage split pipeline (see module docstring).
# ---------------------------------------------------------------------------
def summarize_method_group_comparison_problem(
    input_dir: str | Path,
    problem: str,
    group: str,
    problems_config: str | Path = _DEFAULT_PROBLEMS_CONFIG,
    output_dir: str | Path | None = None,
    save_metrics_dir: str | Path | None = None,
    auc_cache_dir: str | Path | None = None,
) -> list[str]:
    """Per-problem half of the split method-group-comparison pipeline.

    Loads only ``problem``'s own runs, restricted to ``group``'s 5 methods
    (see :data:`METHOD_GROUPS`) -- never any other problem's ``.jsonl`` logs,
    never the other group's methods -- and, when ``save_metrics_dir`` is
    given, writes ``<save_metrics_dir>/<problem>_<group>_comparison_metrics.json``
    (see :func:`save_method_group_problem_metrics`) so a later
    :func:`summarize_method_group_comparison_aggregate` call can combine
    every problem's contribution without re-reading any run logs.

    Raises ``ValueError`` if ``group`` is not ``"batch"``/``"sequential"``, or
    if ``problem`` is not one of the synthetic problems in ``problems_config``
    (i.e. is one of the excluded real-world problems).

    Never renders a per-problem plot -- unlike
    ``summary.summarize_synthetic_comparison_problem``, this report has no
    per-problem PDF at all (see module docstring), so ``output_dir`` is
    accepted only for CLI/signature symmetry with the aggregate stage and the
    synthetic-comparison pipeline; it is otherwise unused here.

    ``auc_cache_dir`` (see :mod:`itcas.reporting.auc_cache`) defaults to
    ``Path(input_dir).parent / "auc_cache"`` when ``None``, mirroring every
    other split pipeline in this package.
    """
    del output_dir  # accepted for signature symmetry only, see docstring

    from .batch_vs_sequential import _collect_family_runs
    from .summary import _precompute_cached, _synthetic_problems

    _validate_group(group)

    input_path = Path(input_dir)
    resolved_auc_cache_dir = (
        Path(auc_cache_dir) if auc_cache_dir is not None else Path(input_dir).parent / "auc_cache"
    )

    synthetic_problems = _synthetic_problems(problems_config)
    if problem not in synthetic_problems:
        raise ValueError(
            f"'{problem}' is not one of the synthetic problems in {problems_config} "
            "(or is one of the excluded real-world problems)."
        )

    methods = list(METHOD_GROUPS[group])
    runs = _collect_family_runs(input_path, [problem], methods).get(problem, [])
    if not runs:
        return []
    cache = _precompute_cached(runs, resolved_auc_cache_dir, problem)

    summary = _method_group_problem_report(
        problem, runs, cache, methods, auc_cache_dir=resolved_auc_cache_dir
    )

    paths: list[str] = []
    if save_metrics_dir is not None:
        metrics_path = save_method_group_problem_metrics(save_metrics_dir, problem, group, summary)
        paths.append(str(metrics_path))
    return paths


def summarize_method_group_comparison_aggregate(
    metrics_dir: str | Path,
    group: str,
    output_dir: str | Path | None = None,
) -> list[str]:
    """Aggregate half of the split method-group-comparison pipeline.

    Reads every ``*_<group>_comparison_metrics.json`` under ``metrics_dir``
    (written by :func:`summarize_method_group_comparison_problem` for this
    same ``group``) -- no ``.jsonl`` run logs, no ``RunSeries``
    reconstruction, for any problem -- and produces the outputs documented in
    :func:`_render_method_group_aggregate`.

    Raises ``ValueError`` if ``group`` is not ``"batch"``/``"sequential"``,
    or (via :func:`load_method_group_problem_metrics`) if a metrics file
    under ``metrics_dir`` was written for the other group.

    ``output_dir`` defaults to ``results/batch_comparison`` for the batch
    group and ``results/sequential_comparison`` for the sequential group
    (see :data:`_DEFAULT_OUTPUT_DIR`) when ``None``.
    """
    _validate_group(group)

    metrics_path = Path(metrics_dir)
    out_dir = Path(output_dir) if output_dir is not None else Path(_DEFAULT_OUTPUT_DIR[group])

    summaries_by_problem: dict[str, dict[str, dict]] = {}
    for f in sorted(metrics_path.glob(f"*_{group}_comparison_metrics.json")):
        problem, diffs = load_method_group_problem_metrics(f, group)
        summaries_by_problem[problem] = diffs

    if not summaries_by_problem:
        return []

    return _render_method_group_aggregate(summaries_by_problem, group, out_dir)
