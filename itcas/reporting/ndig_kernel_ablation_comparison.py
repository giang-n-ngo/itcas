"""NDIG QD-DPP diversity-kernel ablation comparison: the proposed method's
full batch NDIG acquisition (``itcas_ndig``, i.e. ``method="itcas"``,
``quality="ndig"``) against its two kernel-ablated siblings
``ndig_no_kobj_batch``/``ndig_no_kctx_batch`` (see
``itcas/algorithms/continuous.py``'s ``disable_kernel`` /
``itcas/pipeline/loop.py``'s ``NDIG_KERNEL_ABLATION``). Each ablation drops
one half of the QD-DPP joint diversity kernel ``k_obj(mu_i, mu_j) *
k_ctx(c_i, c_j)`` from the L-ensemble -- ``ndig_no_kobj_batch`` keeps only
``k_ctx`` (diversity via context alone), ``ndig_no_kctx_batch`` keeps only
``k_obj`` (diversity via the objective-space prediction alone) -- so this
report answers "how much does each half of the joint kernel actually
contribute to NDIG's batch performance?"

A narrower sibling of :mod:`itcas.reporting.method_group_comparison` (which
compares 5 *different* method families' own batch, or sequential, variants):
this module always compares the same fixed 3-method set within a single
family (:data:`METHODS`), so there is no ``group`` parameter to plumb through.
Otherwise it reuses that module's scope decisions verbatim:

* **Two output pairs, each a 2x2 grid** -- relative-AUC-by-difficulty and
  normalized average curve (see :func:`_render_ablation_aggregate`), each
  laid out as a 2x2 grid of the 4 metrics (via
  :func:`itcas.reporting.summary._plot_relative_auc_by_difficulty_grid_figure`/
  :func:`itcas.reporting.summary._plot_normalized_avg_curve_grid_figure`,
  the 2x2-grid siblings of the one-row figures every other report in this
  package uses) rather than one wide row -- no per-problem plot, no
  relative-AUC-vs-evaluations boxplot, no avg-rank boxplot, no
  Friedman/Wilcoxon stats report. This mirrors
  ``method_group_comparison.py``'s own "No statistics"/"No avg-rank" cuts;
  unlike that module this one also skips the relative-AUC-vs-evaluations
  boxplot (its third output) and any per-problem rendering -- a deliberate
  narrower scope, not an oversight.

  Each of the two figures is rendered in **two variants**: the base
  :data:`METHODS` (3 batch methods only) and a second ``_with_seq`` variant
  that additionally includes ``itcas_seq_ndig`` (:data:`METHODS_WITH_SEQ`) --
  the proposed method's forced-sequential sibling (``method="itcas_seq"``,
  ``quality="ndig"``, no QD-DPP batch diversity at all). The ``_with_seq``
  variant answers a different question than the 3-method one: not "which
  half of the joint kernel matters more" but "how much does QD-DPP batch
  diversity (in any form) help over no diversity mechanism at all" -- so it
  is an *additional* pair of outputs, not a replacement; both are always
  rendered together from the same per-problem data.
* **Scope: standard synthetic problems, 4 shared difficulty levels.** Uses
  :func:`itcas.reporting.summary._synthetic_problems` (default
  ``configs/final_problems.json``, excludes the two real-world problems,
  ``spacecraft_formation_flying_a1``/``casd_llm``) and the shared
  ``p0_01``/``p0_05``/``p0_10``/``p0_20`` difficulty scale -- the same scope
  ``scripts/jobs.json``'s ablation sweep job actually ran these two methods
  under.
* **Axis.** Evaluations only (``RunSeries.x_evals``) -- the 3-method
  :data:`METHODS` variant compares only batch methods with the same
  ``batch_size``, so it isn't a batch-vs-sequential axis mismatch the way
  ``batch_vs_sequential.py``'s docstring describes; the ``_with_seq`` variant
  *does* mix a batch family with its sequential sibling, exactly the same
  mismatch every other report in this package already resolves by using
  ``"evals"`` (total individual evaluations) rather than ``"steps"``
  (algorithmic iterations) -- the shared convention this report also uses.
* **Split, memory-bounded pipeline.** Exactly like
  ``method_group_comparison.py`` (see that module's docstring for the OOM
  history motivating this), only the two-stage split form is offered: a
  **per-problem** stage (:func:`summarize_ndig_kernel_ablation_problem`,
  loads one problem's runs only, writes
  ``<problem>_ndig_kernel_ablation_metrics.json``) and an **aggregate** stage
  (:func:`summarize_ndig_kernel_ablation_aggregate`, reads only those small
  JSON files, never a raw run log). There is no monolithic single-call
  counterpart.

**CLI.** Reachable via ``python -m itcas.summarize --ndig-kernel-ablation-problem``/
``--ndig-kernel-ablation-aggregate-from``, alongside every other split
pipeline in this package -- not its own ``python -m`` entry point, for the
same reason ``method_group_comparison.py`` gives (a second parallel CLI here
would just be a second place to keep in sync).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .metrics import RunSeries
from .method_labels import METHOD_ABBREVIATIONS
from .summary import CurveCache

_DEFAULT_PROBLEMS_CONFIG = "configs/final_problems.json"
_DEFAULT_OUTPUT_DIR = "results/ndig_kernel_ablation_comparison"
_AXIS = "evals"  # the only axis this report ever plots -- see module docstring.

# Proposed method first, then its two kernel-ablated siblings -- see module
# docstring. These are the real ``RunSeries.method`` names (``itcas_ndig`` is
# the parsed label for ``method="itcas", quality="ndig"``, see
# ``visualize._load_run``); ``ndig_no_kobj_batch``/``ndig_no_kctx_batch`` are
# real ``--method`` values (``itcas/pipeline/loop.py``'s
# ``CONTINUOUS_BASELINE_QUALITY``/``NDIG_KERNEL_ABLATION``).
METHODS: list[str] = ["itcas_ndig", "ndig_no_kobj_batch", "ndig_no_kctx_batch"]

# One color per method, distinct from every other family's color in
# summary._SYNTHETIC_METHOD_STYLES / method_group_comparison.METHOD_STYLES so
# this report's figures never accidentally imply a cross-report color
# convention that doesn't exist. itcas_ndig keeps the proposed-method blue
# used everywhere else in this package for visual continuity; the two
# ablations get the next two tab10 colors already used elsewhere in this
# file's sibling reports for a different family (orange, green), reused here
# since this 3-method comparison never appears in the same figure as those.
METHOD_STYLES: dict[str, dict] = {
    "itcas_ndig": {"color": "#1f77b4", "linestyle": "-"},
    "ndig_no_kobj_batch": {"color": "#ff7f0e", "linestyle": "-"},
    "ndig_no_kctx_batch": {"color": "#2ca02c", "linestyle": "-"},
}

# The "_with_seq" variant (see module docstring): METHODS plus itcas_seq_ndig,
# the proposed method's forced-sequential sibling (no QD-DPP batch diversity
# at all -- the natural "zero diversity mechanism" baseline for this
# ablation). Appended last so the 3 batch methods keep their existing
# left-to-right/legend order in both variants.
METHODS_WITH_SEQ: list[str] = METHODS + ["itcas_seq_ndig"]

# itcas_seq_ndig reuses itcas_ndig's own blue hue (same underlying method,
# quality="ndig", just without QD-DPP batch selection) but a dashed
# linestyle, so the two read as "the same algorithm, batch vs. sequential"
# rather than an unrelated 4th color.
METHOD_STYLES_WITH_SEQ: dict[str, dict] = {
    **METHOD_STYLES,
    "itcas_seq_ndig": {"color": "#1f77b4", "linestyle": "--"},
}

# The shared "standard problem" difficulty scale (see module docstring) --
# NOT the FF/CASD reports' own per-problem scales.
DIFFICULTIES: tuple[str, ...] = ("p0_01", "p0_05", "p0_10", "p0_20")


# ---------------------------------------------------------------------------
# Per-problem report + JSON round-trip (mirrors
# method_group_comparison._method_group_problem_report /
# save_method_group_problem_metrics / load_method_group_problem_metrics,
# fixed to METHODS instead of a per-group method list).
# ---------------------------------------------------------------------------
def _ablation_problem_report(
    problem: str,
    runs: list[RunSeries],
    cache: CurveCache,
    *,
    auc_cache_dir: Optional[str | Path] = None,
) -> dict[str, dict]:
    """One problem's contribution to the ablation comparison, per difficulty.

    Computes ``relative_auc``/``normalized_curve_lists``
    (:func:`itcas.reporting.ranking.relative_auc_ratios_over_rows` /
    :func:`itcas.reporting.summary.normalized_curve_grid_lists_over_rows`)
    restricted to :data:`METHODS`, so every ratio/curve is computed relative
    to the best value found among these 3 methods only -- plus a second
    ``_with_seq``-suffixed pair of the same two quantities restricted to
    :data:`METHODS_WITH_SEQ` (adding ``itcas_seq_ndig``), computed from the
    exact same ``rows`` so this never re-reads or re-caches anything twice
    (see module docstring's "Two output pairs" section). No rendered PDF, no
    ``avg_rank``/``auc`` entry -- see module docstring for the out-of-scope
    outputs this intentionally omits.

    Uses only ``runs``/``cache`` for this single ``problem``, so this is safe
    to call from a per-problem job holding just one problem's data in memory.
    When ``auc_cache_dir`` is given (see :mod:`itcas.reporting.auc_cache`),
    reuses already-cached per-seed AUCs and merges newly computed ones back in
    before returning -- mirrors every other split pipeline's caching contract.

    Returns ``{difficulty: {"metrics_present": [metric_key, ...],
    "relative_auc": {column_key: {method: ratio}}, "normalized_curve_lists":
    {column_key: {method: [row_curve]}}, "relative_auc_with_seq": {...},
    "normalized_curve_lists_with_seq": {...}}}``.
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
            rows, METHODS, _AXIS, auc_cache=existing_auc_cache
        )
        normalized_curve_lists = normalized_curve_grid_lists_over_rows(rows, METHODS)[1]
        relative_auc_with_seq = relative_auc_ratios_over_rows(
            rows, METHODS_WITH_SEQ, _AXIS, auc_cache=existing_auc_cache
        )
        normalized_curve_lists_with_seq = normalized_curve_grid_lists_over_rows(
            rows, METHODS_WITH_SEQ
        )[1]
        summary[diff] = {
            "metrics_present": [s.key for s in metrics_present],
            "relative_auc": relative_auc,
            "normalized_curve_lists": normalized_curve_lists,
            "relative_auc_with_seq": relative_auc_with_seq,
            "normalized_curve_lists_with_seq": normalized_curve_lists_with_seq,
        }

    if runs and auc_cache_dir is not None:
        from .auc_cache import compute_auc_table, save_auc_cache

        save_auc_cache(auc_cache_dir, problem, runs, compute_auc_table(runs, cache))

    return summary


def save_ablation_problem_metrics(
    metrics_dir: str | Path,
    problem: str,
    summary_by_diff: dict[str, dict],
) -> Path:
    """Write one problem's :func:`_ablation_problem_report` summary to JSON."""
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / f"{problem}_ndig_kernel_ablation_metrics.json"
    with out_path.open("w") as f:
        json.dump({"problem": problem, "difficulties": summary_by_diff}, f)
    return out_path


def load_ablation_problem_metrics(path: str | Path) -> tuple[str, dict[str, dict]]:
    """Load one problem's summary written by :func:`save_ablation_problem_metrics`.

    Returns ``(problem_name, summary_by_diff)`` -- see
    :func:`_ablation_problem_report` for the shape of ``summary_by_diff``.
    """
    with Path(path).open() as f:
        data = json.load(f)
    return data["problem"], data.get("difficulties", {})


# ---------------------------------------------------------------------------
# Cross-problem combination + rendering (mirrors
# method_group_comparison._combine_method_group_summaries /
# _render_method_group_aggregate, restricted to the two requested outputs).
# ---------------------------------------------------------------------------
def _combine_ablation_summaries(
    summaries_by_problem: dict[str, dict[str, dict]],
    *,
    variant_key: str = "",
) -> dict[str, dict]:
    """Combine every problem's :func:`_ablation_problem_report` summary.

    ``variant_key`` selects which pair of stored per-difficulty entries to
    combine: ``""`` (default) reads ``"relative_auc"``/``"normalized_curve_lists"``
    (the base :data:`METHODS` variant); ``"_with_seq"`` reads
    ``"relative_auc_with_seq"``/``"normalized_curve_lists_with_seq"`` (the
    :data:`METHODS_WITH_SEQ` variant) -- see module docstring's "Two output
    pairs" section. ``"metrics_present"`` is shared by both variants (it is
    method-agnostic, just which metrics have any data at all) so it is never
    suffixed.

    Returns ``{difficulty: {"metrics_present": [MetricSpec, ...],
    "relative_auc_lists": {column_key: {method: [ratio, ...]}}, "n_rows":
    int, "normalized_curve_lists": {column_key: {method: [row_curve,
    ...]}}}}`` -- ``n_rows`` is the number of problems present at that
    difficulty.
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
            for column_key, method_vals in (entry.get(f"relative_auc{variant_key}") or {}).items():
                for method, val in method_vals.items():
                    ratio_lists.setdefault(column_key, {}).setdefault(method, []).append(val)
            for column_key, method_curves in (
                entry.get(f"normalized_curve_lists{variant_key}") or {}
            ).items():
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


def _render_ablation_variant(
    summaries_by_problem: dict[str, dict[str, dict]],
    out_dir: Path,
    *,
    methods: list[str],
    method_styles: dict[str, dict],
    variant_key: str,
    filename_suffix: str,
) -> list[str]:
    """Render one variant's pair of cross-problem outputs (shared by both calls in
    :func:`_render_ablation_aggregate`).

    Writes, restricted to ``methods``, each as a 2x2 grid of the 4 metrics
    (no product panel -- see
    :func:`itcas.reporting.summary._plot_relative_auc_box_grid_figure`'s
    docstring for why a 2x2 layout has no room for a fifth panel):

    * ``<out_dir>/ndig_kernel_ablation_relative_auc_by_difficulty<filename_suffix>.pdf``
      -- one line + IQR band per method spanning every difficulty (see
      :func:`itcas.reporting.summary._plot_relative_auc_by_difficulty_grid_figure`);
      the band shows spread across problems at each difficulty.
    * ``<out_dir>/ndig_kernel_ablation_normalized_avg_curve_vs_pct_budget<filename_suffix>.pdf``
      -- each metric's raw curve normalized to its own (problem, difficulty)
      row's best-within-``methods`` value and averaged on the shared "% of
      that row's own evaluation budget" x-axis (see
      :func:`itcas.reporting.summary._plot_normalized_avg_curve_grid_figure`).

    Deliberately does not render the relative-AUC-vs-evaluations boxplot, any
    avg-rank plot, or a stats report -- see module docstring.
    """
    from .summary import (
        _default_pct_grid,
        _feasible_pct_label,
        _ordered_metrics,
        _plot_normalized_avg_curve_grid_figure,
        _plot_relative_auc_by_difficulty_grid_figure,
    )

    combined = _combine_ablation_summaries(summaries_by_problem, variant_key=variant_key)

    paths: list[str] = []
    if not combined:
        return paths

    relative_auc_by_level = [
        (_feasible_pct_label(diff), combined[diff]["relative_auc_lists"])
        for diff in sorted(combined)
    ]
    metric_keys_all = {s.key for e in combined.values() for s in e["metrics_present"]}
    metrics_present_all = [s for s in _ordered_metrics() if s.key in metric_keys_all]

    by_diff_path = out_dir / f"ndig_kernel_ablation_relative_auc_by_difficulty{filename_suffix}.pdf"
    ok = _plot_relative_auc_by_difficulty_grid_figure(
        relative_auc_by_level, methods, method_styles, metrics_present_all, by_diff_path,
        method_labels=METHOD_ABBREVIATIONS,
        x_axis_label="Difficulty level",
    )
    if ok is not None:
        paths.append(str(ok))

    normalized_curve_lists: dict[str, dict[str, list[list[float]]]] = {}
    for entry in combined.values():
        for column_key, method_curves in (entry.get("normalized_curve_lists") or {}).items():
            for method, curves in method_curves.items():
                normalized_curve_lists.setdefault(column_key, {}).setdefault(method, []).extend(curves)

    normalized_curve_path = (
        out_dir / f"ndig_kernel_ablation_normalized_avg_curve_vs_pct_budget{filename_suffix}.pdf"
    )
    ok = _plot_normalized_avg_curve_grid_figure(
        _default_pct_grid(), normalized_curve_lists, methods, method_styles,
        metrics_present_all, normalized_curve_path,
        method_labels=METHOD_ABBREVIATIONS,
    )
    if ok is not None:
        paths.append(str(ok))

    return paths


def _render_ablation_aggregate(
    summaries_by_problem: dict[str, dict[str, dict]],
    output_dir: str | Path,
) -> list[str]:
    """Render both variants' cross-problem outputs from combined per-problem summaries.

    Calls :func:`_render_ablation_variant` twice (see module docstring's "Two
    output pairs" section): once for the base :data:`METHODS` (3 batch
    methods, unsuffixed filenames -- unchanged from before this variant was
    added) and once for :data:`METHODS_WITH_SEQ` (adds ``itcas_seq_ndig``,
    ``"_with_seq"``-suffixed filenames), both reading from the *same*
    ``summaries_by_problem`` (populated with both variants' data by
    :func:`_ablation_problem_report`) -- no extra disk I/O for the second
    variant.
    """
    out_dir = Path(output_dir)
    paths = _render_ablation_variant(
        summaries_by_problem, out_dir,
        methods=METHODS, method_styles=METHOD_STYLES,
        variant_key="", filename_suffix="",
    )
    paths.extend(_render_ablation_variant(
        summaries_by_problem, out_dir,
        methods=METHODS_WITH_SEQ, method_styles=METHOD_STYLES_WITH_SEQ,
        variant_key="_with_seq", filename_suffix="_with_seq",
    ))
    return paths


# ---------------------------------------------------------------------------
# Public API -- two-stage split pipeline (see module docstring).
# ---------------------------------------------------------------------------
def summarize_ndig_kernel_ablation_problem(
    input_dir: str | Path,
    problem: str,
    problems_config: str | Path = _DEFAULT_PROBLEMS_CONFIG,
    output_dir: str | Path | None = None,
    save_metrics_dir: str | Path | None = None,
    auc_cache_dir: str | Path | None = None,
) -> list[str]:
    """Per-problem half of the split ablation-comparison pipeline.

    Loads only ``problem``'s own runs, restricted to :data:`METHODS_WITH_SEQ`
    (the union needed by both output variants, see module docstring) --
    never any other problem's ``.jsonl`` logs -- and, when
    ``save_metrics_dir`` is given, writes
    ``<save_metrics_dir>/<problem>_ndig_kernel_ablation_metrics.json`` (see
    :func:`save_ablation_problem_metrics`) so a later
    :func:`summarize_ndig_kernel_ablation_aggregate` call can combine every
    problem's contribution without re-reading any run logs.

    Raises ``ValueError`` if ``problem`` is not one of the synthetic problems
    in ``problems_config`` (i.e. is one of the excluded real-world problems).

    Never renders a per-problem plot (see module docstring); ``output_dir``
    is accepted only for CLI/signature symmetry with the aggregate stage and
    every other split pipeline in this package, and is otherwise unused here.

    ``auc_cache_dir`` (see :mod:`itcas.reporting.auc_cache`) defaults to
    ``Path(input_dir).parent / "auc_cache"`` when ``None``.
    """
    del output_dir  # accepted for signature symmetry only, see docstring

    from .batch_vs_sequential import _collect_family_runs
    from .summary import _precompute_cached, _synthetic_problems

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

    runs = _collect_family_runs(input_path, [problem], METHODS_WITH_SEQ).get(problem, [])
    if not runs:
        return []
    cache = _precompute_cached(runs, resolved_auc_cache_dir, problem)

    summary = _ablation_problem_report(problem, runs, cache, auc_cache_dir=resolved_auc_cache_dir)

    paths: list[str] = []
    if save_metrics_dir is not None:
        metrics_path = save_ablation_problem_metrics(save_metrics_dir, problem, summary)
        paths.append(str(metrics_path))
    return paths


def summarize_ndig_kernel_ablation_aggregate(
    metrics_dir: str | Path,
    output_dir: str | Path | None = None,
) -> list[str]:
    """Aggregate half of the split ablation-comparison pipeline.

    Reads every ``*_ndig_kernel_ablation_metrics.json`` under ``metrics_dir``
    (written by :func:`summarize_ndig_kernel_ablation_problem`) -- no
    ``.jsonl`` run logs, no ``RunSeries`` reconstruction, for any problem --
    and produces the outputs documented in :func:`_render_ablation_aggregate`.

    ``output_dir`` defaults to ``results/ndig_kernel_ablation_comparison``
    when ``None``.
    """
    metrics_path = Path(metrics_dir)
    out_dir = Path(output_dir) if output_dir is not None else Path(_DEFAULT_OUTPUT_DIR)

    summaries_by_problem: dict[str, dict[str, dict]] = {}
    for f in sorted(metrics_path.glob("*_ndig_kernel_ablation_metrics.json")):
        problem, diffs = load_ablation_problem_metrics(f)
        summaries_by_problem[problem] = diffs

    if not summaries_by_problem:
        return []

    return _render_ablation_aggregate(summaries_by_problem, out_dir)
