"""NDIG-B acquisition-*component* ablation comparison (reviewer-requested):
the proposed method's full batch NDIG acquisition (``itcas_ndig``, i.e.
``method="itcas"``, ``quality="ndig"``) against four ablations of its own
quality score ``q(z) = depth(z) * info_gain(z)`` (as opposed to
:mod:`itcas.reporting.ndig_kernel_ablation_comparison`, which ablates the
QD-DPP batch-*diversity* mechanism, an orthogonal axis untouched here):

* ``itcas_ndig_no_infogain`` -- "remove information gain": normalized depth
  alone, no info-gain multiplier (``itcas/algorithms/quality.py``'s
  ``ndig_no_infogain``).
* ``itcas_edig`` -- "remove normalization of expected depth": EDIG *is*
  exactly NDIG's unbounded, unnormalized depth term times info gain, reused
  unchanged (``--quality edig``).
* ``itcas_efig`` -- "PoF times information gain": EFIG *is* exactly this
  formulation, reused unchanged (``--quality efig``).
* ``itcas_ndig_pof_entropy`` -- "binary entropy of PoF": no depth or
  info-gain term at all (``itcas/algorithms/quality.py``'s
  ``ndig_pof_entropy``).

A narrower sibling of :mod:`itcas.reporting.ndig_kernel_ablation_comparison`
(read that module's docstring first; this one reuses its scope decisions
verbatim except where noted):

* **Two output pairs.** Unlike the kernel-ablation report, there is only
  *one* variant here (the 5-method set below) -- no ``_with_seq`` sibling.
  Every method compared is already a batch (``method="itcas"``) run with the
  same ``batch_size``, so there is no "how much does batch diversity itself
  help" question in scope; this report only ever asks "which quality
  component matters." Still renders the same two grid figures per the
  kernel-ablation report (relative-AUC-by-difficulty, normalized average
  curve), never a per-problem plot, boxplot, avg-rank, or stats report.
* **Scope: standard synthetic problems, 4 shared difficulty levels, evals
  axis.** Same as the kernel-ablation report -- see its docstring.
* **Split, memory-bounded pipeline.** Same per-problem/aggregate split as
  every other report in this package (OOM history motivating this pattern
  lives in ``method_group_comparison.py``'s docstring).

**CLI.** Reachable via ``python -m itcas.summarize
--ndig-b-component-ablation-problem``/``--ndig-b-component-ablation-aggregate-from``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .metrics import RunSeries
from .method_labels import METHOD_ABBREVIATIONS
from .summary import CurveCache

_DEFAULT_PROBLEMS_CONFIG = "configs/final_problems.json"
_DEFAULT_OUTPUT_DIR = "results/ndig_b_component_ablation_comparison"
_AXIS = "evals"  # the only axis this report ever plots -- see module docstring.

# Proposed method first, then its four component ablations -- see module
# docstring. ``itcas_ndig``/``itcas_edig``/``itcas_efig``/
# ``itcas_ndig_no_infogain``/``itcas_ndig_pof_entropy`` are all the parsed
# labels for ``method="itcas", quality=<name>`` (see ``visualize._load_run``).
METHODS: list[str] = [
    "itcas_ndig",
    "itcas_ndig_no_infogain",
    "itcas_edig",
    "itcas_efig",
    "itcas_ndig_pof_entropy",
]

# One color per method, distinct from every other family's color in
# summary._SYNTHETIC_METHOD_STYLES / method_group_comparison.METHOD_STYLES /
# ndig_kernel_ablation_comparison.METHOD_STYLES so this report's figures
# never accidentally imply a cross-report color convention that doesn't
# exist. itcas_ndig keeps the proposed-method blue used everywhere else in
# this package; the four ablations get the next four tab10 colors.
METHOD_STYLES: dict[str, dict] = {
    "itcas_ndig": {"color": "#1f77b4", "linestyle": "-"},
    "itcas_ndig_no_infogain": {"color": "#ff7f0e", "linestyle": "-"},
    "itcas_edig": {"color": "#2ca02c", "linestyle": "-"},
    "itcas_efig": {"color": "#d62728", "linestyle": "-"},
    "itcas_ndig_pof_entropy": {"color": "#9467bd", "linestyle": "-"},
}

# The shared "standard problem" difficulty scale (see module docstring) --
# NOT the FF/CASD reports' own per-problem scales.
DIFFICULTIES: tuple[str, ...] = ("p0_01", "p0_05", "p0_10", "p0_20")


# ---------------------------------------------------------------------------
# Per-problem report + JSON round-trip (mirrors
# ndig_kernel_ablation_comparison._ablation_problem_report /
# save_ablation_problem_metrics / load_ablation_problem_metrics, fixed to
# this module's own 5-method METHODS and with no "_with_seq" variant).
# ---------------------------------------------------------------------------
def _component_ablation_problem_report(
    problem: str,
    runs: list[RunSeries],
    cache: CurveCache,
    *,
    auc_cache_dir: Optional[str | Path] = None,
) -> dict[str, dict]:
    """One problem's contribution to the component-ablation comparison, per difficulty.

    Computes ``relative_auc``/``normalized_curve_lists``
    (:func:`itcas.reporting.ranking.relative_auc_ratios_over_rows` /
    :func:`itcas.reporting.summary.normalized_curve_grid_lists_over_rows`)
    restricted to :data:`METHODS`, so every ratio/curve is computed relative
    to the best value found among these 5 methods only. No rendered PDF, no
    ``avg_rank``/``auc`` entry -- see module docstring for the out-of-scope
    outputs this intentionally omits.

    Uses only ``runs``/``cache`` for this single ``problem``, so this is safe
    to call from a per-problem job holding just one problem's data in memory.
    When ``auc_cache_dir`` is given (see :mod:`itcas.reporting.auc_cache`),
    reuses already-cached per-seed AUCs and merges newly computed ones back in
    before returning -- mirrors every other split pipeline's caching contract.

    Returns ``{difficulty: {"metrics_present": [metric_key, ...],
    "relative_auc": {column_key: {method: ratio}}, "normalized_curve_lists":
    {column_key: {method: [row_curve]}}}}``.
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
        summary[diff] = {
            "metrics_present": [s.key for s in metrics_present],
            "relative_auc": relative_auc,
            "normalized_curve_lists": normalized_curve_lists,
        }

    if runs and auc_cache_dir is not None:
        from .auc_cache import compute_auc_table, save_auc_cache

        save_auc_cache(auc_cache_dir, problem, runs, compute_auc_table(runs, cache))

    return summary


def save_component_ablation_problem_metrics(
    metrics_dir: str | Path,
    problem: str,
    summary_by_diff: dict[str, dict],
) -> Path:
    """Write one problem's :func:`_component_ablation_problem_report` summary to JSON."""
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / f"{problem}_ndig_b_component_ablation_metrics.json"
    with out_path.open("w") as f:
        json.dump({"problem": problem, "difficulties": summary_by_diff}, f)
    return out_path


def load_component_ablation_problem_metrics(path: str | Path) -> tuple[str, dict[str, dict]]:
    """Load one problem's summary written by :func:`save_component_ablation_problem_metrics`.

    Returns ``(problem_name, summary_by_diff)`` -- see
    :func:`_component_ablation_problem_report` for the shape of ``summary_by_diff``.
    """
    with Path(path).open() as f:
        data = json.load(f)
    return data["problem"], data.get("difficulties", {})


# ---------------------------------------------------------------------------
# Cross-problem combination + rendering (mirrors
# ndig_kernel_ablation_comparison._combine_ablation_summaries /
# _render_ablation_aggregate, with no "_with_seq" variant so the two are
# merged into single, unsuffixed functions).
# ---------------------------------------------------------------------------
def _combine_component_ablation_summaries(
    summaries_by_problem: dict[str, dict[str, dict]],
) -> dict[str, dict]:
    """Combine every problem's :func:`_component_ablation_problem_report` summary.

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


def _render_ablation_aggregate(
    summaries_by_problem: dict[str, dict[str, dict]],
    output_dir: str | Path,
) -> list[str]:
    """Render this report's pair of cross-problem outputs from combined per-problem summaries.

    Writes, restricted to :data:`METHODS`, each as a near-square grid of
    every registered metric (no product panel -- see
    :func:`itcas.reporting.summary._grid_shape` and
    :func:`itcas.reporting.summary._plot_relative_auc_box_grid_figure`'s
    docstring):

    * ``<output_dir>/ndig_b_component_ablation_relative_auc_by_difficulty.pdf``
      -- one line + IQR band per method spanning every difficulty (see
      :func:`itcas.reporting.summary._plot_relative_auc_by_difficulty_grid_figure`);
      the band shows spread across problems at each difficulty.
    * ``<output_dir>/ndig_b_component_ablation_normalized_avg_curve_vs_pct_budget.pdf``
      -- each metric's raw curve normalized to its own (problem, difficulty)
      row's best-within-:data:`METHODS` value and averaged on the shared "%
      of that row's own evaluation budget" x-axis (see
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

    out_dir = Path(output_dir)
    combined = _combine_component_ablation_summaries(summaries_by_problem)

    paths: list[str] = []
    if not combined:
        return paths

    relative_auc_by_level = [
        (_feasible_pct_label(diff), combined[diff]["relative_auc_lists"])
        for diff in sorted(combined)
    ]
    metric_keys_all = {s.key for e in combined.values() for s in e["metrics_present"]}
    metrics_present_all = [s for s in _ordered_metrics() if s.key in metric_keys_all]

    by_diff_path = out_dir / "ndig_b_component_ablation_relative_auc_by_difficulty.pdf"
    ok = _plot_relative_auc_by_difficulty_grid_figure(
        relative_auc_by_level, METHODS, METHOD_STYLES, metrics_present_all, by_diff_path,
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
        out_dir / "ndig_b_component_ablation_normalized_avg_curve_vs_pct_budget.pdf"
    )
    ok = _plot_normalized_avg_curve_grid_figure(
        _default_pct_grid(), normalized_curve_lists, METHODS, METHOD_STYLES,
        metrics_present_all, normalized_curve_path,
        method_labels=METHOD_ABBREVIATIONS,
    )
    if ok is not None:
        paths.append(str(ok))

    return paths


# ---------------------------------------------------------------------------
# Public API -- two-stage split pipeline (see module docstring).
# ---------------------------------------------------------------------------
def summarize_ndig_b_component_ablation_problem(
    input_dir: str | Path,
    problem: str,
    problems_config: str | Path = _DEFAULT_PROBLEMS_CONFIG,
    output_dir: str | Path | None = None,
    save_metrics_dir: str | Path | None = None,
    auc_cache_dir: str | Path | None = None,
) -> list[str]:
    """Per-problem half of the split component-ablation-comparison pipeline.

    Loads only ``problem``'s own runs, restricted to :data:`METHODS` -- never
    any other problem's ``.jsonl`` logs -- and, when ``save_metrics_dir`` is
    given, writes ``<save_metrics_dir>/<problem>_ndig_b_component_ablation_
    metrics.json`` (see :func:`save_component_ablation_problem_metrics`) so a
    later :func:`summarize_ndig_b_component_ablation_aggregate` call can
    combine every problem's contribution without re-reading any run logs.

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

    runs = _collect_family_runs(input_path, [problem], METHODS).get(problem, [])
    if not runs:
        return []
    cache = _precompute_cached(runs, resolved_auc_cache_dir, problem)

    summary = _component_ablation_problem_report(
        problem, runs, cache, auc_cache_dir=resolved_auc_cache_dir
    )

    paths: list[str] = []
    if save_metrics_dir is not None:
        metrics_path = save_component_ablation_problem_metrics(save_metrics_dir, problem, summary)
        paths.append(str(metrics_path))
    return paths


def summarize_ndig_b_component_ablation_aggregate(
    metrics_dir: str | Path,
    output_dir: str | Path | None = None,
) -> list[str]:
    """Aggregate half of the split component-ablation-comparison pipeline.

    Reads every ``*_ndig_b_component_ablation_metrics.json`` under
    ``metrics_dir`` (written by
    :func:`summarize_ndig_b_component_ablation_problem`) -- no ``.jsonl`` run
    logs, no ``RunSeries`` reconstruction, for any problem -- and produces
    the outputs documented in :func:`_render_ablation_aggregate`.

    ``output_dir`` defaults to ``results/ndig_b_component_ablation_comparison``
    when ``None``.
    """
    metrics_path = Path(metrics_dir)
    out_dir = Path(output_dir) if output_dir is not None else Path(_DEFAULT_OUTPUT_DIR)

    summaries_by_problem: dict[str, dict[str, dict]] = {}
    for f in sorted(metrics_path.glob("*_ndig_b_component_ablation_metrics.json")):
        problem, diffs = load_component_ablation_problem_metrics(f)
        summaries_by_problem[problem] = diffs

    if not summaries_by_problem:
        return []

    return _render_ablation_aggregate(summaries_by_problem, out_dir)
