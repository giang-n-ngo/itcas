"""Shared row/grid-building infrastructure for "one algorithm's sequential
variant vs its own batch variant" comparisons.

This module used to also *produce* a batch-vs-sequential report of its own
(grid PDFs per family group plus a per-family Wilcoxon significance report
and a cross-family ``SUMMARY.md`` synthesis) -- that report-generation code
has been removed and replaced by
:mod:`itcas.reporting.batch_improvement_comparison` (a single %-change
heatmap answering "how much does batch improve each method", rather than
this module's former per-(problem, difficulty) "is batch significantly
better" hypothesis test). What remains here is purely shared plumbing that
several *other* per-comparison reports (``ff_comparison.py``,
``casd_comparison.py``, ``lse_pure_comparison.py``, ``lse_ratio_comparison.py``,
and :mod:`itcas.reporting.summary`'s synthetic-comparison pipeline) build on:

* :func:`_collect_family_runs` -- discover + filter runs to a method set,
  grouped by problem, every difficulty kept.
* :data:`_Row` / :func:`_rows_by_problem` / :func:`_rows_by_difficulty` --
  the ``(row_label, runs, cache)`` triple every grid-building report in this
  package uses, and the two ways to build a list of them: "standard layout"
  (one row per problem sharing a difficulty) or "real-world layout" (one row
  per difficulty level of a single problem with its own per-problem
  difficulty scale, e.g. ``spacecraft_formation_flying_a1``/``casd_llm``, see
  :data:`_REAL_WORLD_PROBLEMS`).
* :func:`plot_group_grid` -- the grid-of-panels renderer (rows = ``_Row``\\ s,
  columns = metric curves + raw product [+ product rank]) every one of the
  reports above uses, in some cases with report-specific row selection/
  labeling layered on top.
* :func:`_collect_product_auc_for_stats` -- per-seed area-under-the-product-
  curve, bucketed by ``{problem: {difficulty: {method: [auc, ...]}}}``, used
  by ``ff_comparison.py``/``casd_comparison.py``/``summary.py``'s synthetic
  pipeline for their own Friedman/Wilcoxon dominance tests.

**Evaluations, not steps, is the correct shared axis** whenever a report here
compares a method's own sequential and batch variants (as opposed to
comparing *different* algorithms, which is :mod:`itcas.reporting.school_comparison`'s
concern and has its own per-setting axis-choice logic): total individual
evaluations (``RunSeries.x_evals``) is populated by
:func:`itcas.reporting.visualize._load_run` as the cumulative count of
individually evaluated points (``n_eval_total = X_run.shape[0]``, or the
record's own ``n_eval_total``/``n_eval_this_iter`` fallback) for *every* run
regardless of ``batch_size``, whereas ``x_steps`` only counts algorithmic
iterations (one per batch, however large) -- so it would silently rescale a
sequential run's x-axis relative to its own batch sibling by a factor that
has nothing to do with sample efficiency. Every function in this module that
bakes in an axis choice (:func:`plot_group_grid`, :func:`_collect_product_auc_for_stats`)
uses ``"evals"`` for exactly this reason.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .metrics import RunSeries
from .summary import (
    CurveCache,
    _SHORT_CURVE_LABELS,
    _compute_seed_product_curve,
    _curve_area,
    _difficulty_of,
    _min_med_max,
    _ordered_metrics,
    _per_method_product_curves,
    _rank_curves_on_union_grid,
)
from .visualize import _discover_runs

# Problems that use their own per-problem difficulty scale (p1_00..pN_00, via
# --threshold_pct) instead of the shared p0_01/p0_05/p0_10/p0_20 scale used by
# every other problem in configs/final_problems.json. Each is backed by a
# real-world evaluator (Basilisk simulation for the former, a live LLM +
# judge-model server for the latter) rather than a closed-form objective --
# see ff_comparison.py / casd_comparison.py, the dedicated per-problem reports
# for each. The level *count* differs per problem (10 for FF, 4 for CASD;
# see configs/thresholds.json) but every helper below (_rows_by_difficulty,
# _difficulties_present, ...) is already agnostic to how many levels a given
# problem has, so adding a second entry here is the only change needed to
# extend the "rows = difficulty level" layout to a new real-world problem.
_REAL_WORLD_PROBLEMS: tuple[str, ...] = ("spacecraft_formation_flying_a1", "casd_llm")

_AXIS = "evals"  # the only axis this module's functions ever plot/test on -- see module docstring.


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def _collect_family_runs(
    input_dir: Path,
    problems: list[str],
    methods: list[str],
) -> dict[str, list[RunSeries]]:
    """Runs grouped by problem, filtered to ``methods`` (every difficulty kept).

    Mirrors ``school_comparison._collect_setting_runs`` but without a
    difficulty filter -- callers here want every difficulty present on disk,
    not just one "default".
    """
    out: dict[str, list[RunSeries]] = {p: [] for p in problems}
    method_set = set(methods)
    for problem in problems:
        problem_dir = input_dir / problem
        runs = _discover_runs(problem_dir if problem_dir.is_dir() else input_dir, benchmark=problem)
        out[problem] = [r for r in runs if r.method in method_set]
    return out


def _difficulties_present(runs_by_problem: dict[str, list[RunSeries]]) -> list[str]:
    diffs = {_difficulty_of(r) for runs in runs_by_problem.values() for r in runs}
    return sorted(diffs)


def _filter_to_difficulty(
    problems: list[str],
    runs_by_problem: dict[str, list[RunSeries]],
    caches_by_problem: dict[str, CurveCache],
    diff: str,
) -> tuple[list[str], dict[str, list[RunSeries]], dict[str, CurveCache]]:
    """Project the (all-difficulty) discovery down to one difficulty, in memory."""
    filtered_runs: dict[str, list[RunSeries]] = {}
    filtered_caches: dict[str, CurveCache] = {}
    problems_present: list[str] = []
    for p in problems:
        runs = [r for r in runs_by_problem.get(p, []) if _difficulty_of(r) == diff]
        if not runs:
            continue
        problems_present.append(p)
        filtered_runs[p] = runs
        full_cache = caches_by_problem.get(p, {})
        filtered_caches[p] = {
            r.run_name: full_cache[r.run_name] for r in runs if r.run_name in full_cache
        }
    return problems_present, filtered_runs, filtered_caches


# Row = (row_label, runs (mixed methods, single problem+difficulty slice),
#        cache restricted to those runs' run_names). The plotting code below
# only ever needs this triple, whether a row represents "one problem at a
# fixed difficulty" (the standard layout) or "one difficulty of a single
# problem" (the spacecraft layout) -- see module docstring.
_Row = tuple[str, list[RunSeries], CurveCache]


def _rows_by_problem(
    problems: list[str],
    runs_by_problem: dict[str, list[RunSeries]],
    caches_by_problem: dict[str, CurveCache],
    diff: str,
) -> list[_Row]:
    """Standard layout: one row per problem present at ``diff``."""
    problems_present, diff_runs, diff_caches = _filter_to_difficulty(
        problems, runs_by_problem, caches_by_problem, diff
    )
    return [(p, diff_runs[p], diff_caches.get(p, {})) for p in problems_present]


def _rows_by_difficulty(
    runs: list[RunSeries],
    cache: CurveCache,
) -> list[_Row]:
    """Spacecraft layout: one row per difficulty level of a single problem."""
    diffs = sorted({_difficulty_of(r) for r in runs})
    rows: list[_Row] = []
    for diff in diffs:
        diff_runs = [r for r in runs if _difficulty_of(r) == diff]
        if not diff_runs:
            continue
        diff_cache = {r.run_name: cache[r.run_name] for r in diff_runs if r.run_name in cache}
        rows.append((diff, diff_runs, diff_cache))
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_group_grid(
    methods: list[str],
    method_styles: dict[str, dict],
    rows: list[_Row],
    title: Optional[str],
    out_path: str | Path,
    avg_rank_row: Optional[dict[str, dict[str, float]]] = None,
    include_product_rank_column: bool = True,
    method_labels: Optional[dict[str, str]] = None,
) -> Optional[Path]:
    """Render one grid figure: rows = ``rows``, cols = metrics + product [+ rank].

    Each panel holds one line per entry in ``methods`` (color+linestyle from
    ``method_styles``), plotted against total individual evaluations (see
    module docstring for why). Layout mirrors
    ``school_comparison.plot_school_comparison`` /
    ``summary._plot_problem_curves``.

    ``title`` is accepted for backward compatibility with existing callers
    but is never rendered -- every plot in this package omits its figure
    ``suptitle`` (per-panel column headers and axis labels still carry the
    same information).

    ``method_labels`` is an optional ``{real_method_name: short_display_label}``
    map (e.g. :mod:`itcas.reporting.method_labels`) used *only* for the text
    drawn in legends and the average-rank row's y-tick labels -- every
    internal lookup (``method_runs``, ``method_styles``, ``avg_rank_row``)
    still keys strictly on the real method name passed in ``methods``. Methods
    absent from the map keep their real name as the display label, and
    omitting ``method_labels`` entirely (the default) reproduces the old
    behavior of using real names everywhere.

    Each metric column's header (and the raw-product column's) is suffixed
    with ``↑`` (higher is better) or ``↓`` (lower is better), taken from
    ``MetricSpec.higher_is_better`` (always ``↑`` for the raw-product column,
    which is higher-is-better by construction).

    ``include_product_rank_column`` defaults to ``True`` (existing behavior,
    unchanged for every current caller): the trailing product-rank column
    (per-iteration rank of the product curve, 1 = best) is rendered. Pass
    ``False`` to drop that column entirely -- e.g. when a report's rows
    already carry enough per-row context that a fourth derived column is
    redundant -- leaving just the metric columns + the raw-product column.

    ``avg_rank_row`` is optional and fully backward compatible: when omitted
    (the default), output is unchanged from before this parameter existed.
    When given -- keyed ``{column_key: {method: avg_rank}}`` (one key per
    metric plus the synthetic ``"product"`` key), exactly
    :func:`itcas.reporting.ranking.average_ranks_over_rows`'s return shape --
    one extra row is appended at the bottom of the grid. Under each metric
    column, and under the raw-product column, that row draws a horizontal bar
    chart of each method's average rank (1 = best) across ``rows``, bars
    colored via ``method_styles``, sorted best-to-worst, with one shared title
    ("Average rank across N rows (1 = best)"), anchored above the middle
    metric column so it reads as centered over the row, instead of a
    per-panel x-label. The product-rank column is left blank in that
    row: it already visualizes a rank (of the product, over time) in every
    data row above, so a redundant AUC-rank bar duplicating the adjacent
    product column's new bottom cell would add no information.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if not rows:
        return None

    metrics = _ordered_metrics()
    seen_keys: set[str] = set()
    for _, _, cache in rows:
        for run_curves in cache.values():
            for key, y in run_curves.items():
                if y is not None:
                    seen_keys.add(key)
    metrics_present = [s for s in metrics if s.key in seen_keys]
    if not metrics_present:
        return None

    def _label(method: str) -> str:
        return method_labels.get(method, method) if method_labels else method

    n_data_rows = len(rows)
    n_rows = n_data_rows + (1 if avg_rank_row is not None else 0)
    n_cols = len(metrics_present) + (2 if include_product_rank_column else 1)
    product_col_idx = -2 if include_product_rank_column else -1

    fig_w = max(4.0 * n_cols, 12.0)
    fig_h = max(2.5 * n_rows + 1.0, 5.0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    for r_idx, (row_label, row_runs, cache) in enumerate(rows):
        method_runs: dict[str, list[RunSeries]] = {}
        for run in row_runs:
            method_runs.setdefault(run.method, []).append(run)

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
                    y = (cache.get(run.run_name) or {}).get(spec.key)
                    if y is None:
                        continue
                    xs = run.x_evals
                    n = min(len(xs), len(y))
                    curves.append([float(v) for v in y[:n]])
                    if x_ref is None:
                        x_ref = list(xs[:n])
                if not curves or x_ref is None:
                    continue
                lo, med, hi = _min_med_max(curves)
                n = min(len(med), len(x_ref))
                x_plot = x_ref[:n]
                style = method_styles[method]
                ax.plot(
                    x_plot, med[:n], color=style["color"], linestyle=style["linestyle"],
                    linewidth=1.5, label=_label(method),
                )
                if len(curves) > 1:
                    ax.fill_between(x_plot, lo[:n], hi[:n], color=style["color"], alpha=0.15)
                plotted = True
            if not plotted:
                ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12, color="grey")
            if r_idx == 0:
                arrow = "↑" if spec.higher_is_better else "↓"
                ax.set_title(f"{_SHORT_CURVE_LABELS.get(spec.key, spec.label)} {arrow}", fontsize=12)
            if r_idx == n_data_rows - 1:
                ax.set_xlabel("Total individual evaluations", fontsize=12)
            if c_idx == 0:
                ax.set_ylabel(row_label, fontsize=12)

        method_x, method_med, method_lo, method_hi = _per_method_product_curves(
            method_runs, methods, _AXIS, cache, metrics_present,
        )

        # Raw product column
        ax = axes[r_idx][product_col_idx]
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=10.5)
        plotted = False
        for method in methods:
            if method not in method_med:
                continue
            style = method_styles[method]
            x_plot = method_x[method]
            ax.plot(
                x_plot, method_med[method], color=style["color"], linestyle=style["linestyle"],
                linewidth=1.5, label=_label(method),
            )
            if len(method_runs.get(method, [])) > 1:
                ax.fill_between(x_plot, method_lo[method], method_hi[method],
                                color=style["color"], alpha=0.15)
            plotted = True
        if not plotted:
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="grey")
        if r_idx == 0:
            ax.set_title("Product ↑\n(raw)", fontsize=12)
        if r_idx == n_data_rows - 1:
            ax.set_xlabel("Total individual evaluations", fontsize=12)

        # Rank column (omitted entirely when include_product_rank_column=False)
        if include_product_rank_column:
            ax = axes[r_idx][-1]
            ax.grid(True, axis="y", alpha=0.25)
            ax.tick_params(axis="both", labelsize=10.5)
            plotted = False
            if method_med:
                grid, rank_curves = _rank_curves_on_union_grid(method_x, method_med)
                for method in methods:
                    if method not in rank_curves:
                        continue
                    style = method_styles[method]
                    ax.plot(grid, rank_curves[method], color=style["color"],
                            linestyle=style["linestyle"], linewidth=1.5, label=_label(method))
                    plotted = True
                n_ranked = len(method_med)
                ax.set_ylim(n_ranked + 0.5, 0.5)
                ax.set_yticks(list(range(1, n_ranked + 1)))
            if not plotted:
                ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12, color="grey")
            if r_idx == 0:
                ax.set_title("Product rank\n(1 = best)", fontsize=12)
            if r_idx == n_data_rows - 1:
                ax.set_xlabel("Total individual evaluations", fontsize=12)

    # Optional bottom row: average rank per metric across `rows` (see
    # avg_rank_row in the docstring above). Absent by default, so this whole
    # block is a no-op -- and therefore byte-for-byte inert -- for every
    # existing caller that doesn't pass avg_rank_row.
    if avg_rank_row is not None:
        r_idx = n_data_rows

        def _draw_avg_rank_bar(ax, data: Optional[dict[str, float]]) -> None:
            ax.tick_params(axis="both", labelsize=10.5)
            if not data:
                ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12, color="grey")
                return
            # Best-to-worst (ascending avg rank); bars start at x=0 so a
            # method's bar length grows with its avg rank -- rank-1
            # (best) methods therefore have the shortest bars, naturally
            # keeping "smaller/better" toward the left of the axis.
            present = [m for m in methods if m in data]
            present_sorted = sorted(present, key=lambda m: data[m])
            y_pos = list(range(len(present_sorted)))
            colors = [method_styles[m]["color"] for m in present_sorted]
            values = [data[m] for m in present_sorted]
            ax.barh(y_pos, values, color=colors)
            ax.grid(True, axis="x", alpha=0.25)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([_label(m) for m in present_sorted], fontsize=10.5)
            ax.invert_yaxis()  # best (lowest avg rank) at top

        for c_idx, spec in enumerate(metrics_present):
            ax = axes[r_idx][c_idx]
            _draw_avg_rank_bar(ax, avg_rank_row.get(spec.key))
            if c_idx == 0:
                ax.set_ylabel("Avg rank", fontsize=12)

        # One shared title for the whole row, set via a single axes' own
        # set_title (rather than fig.text after layout) so tight_layout --
        # called further below -- actually reserves vertical room for it;
        # fig.text added post-layout was found to collide with the previous
        # row's "Total individual evaluations" x-label, which sits in the
        # same inter-row gap and isn't accounted for by a bare fig.text.
        # Anchored on the middle metric column so it reads as roughly
        # centered over the row without needing a spanning artist.
        mid_c = len(metrics_present) // 2
        axes[r_idx][mid_c].set_title(
            f"Average rank across {n_data_rows} rows (1 = best)", fontsize=15, fontweight="bold",
        )

        # Raw-product column gets the same treatment, keyed "product" (see
        # ranking.average_ranks_over_rows). Product-rank column, when present,
        # stays blank (see docstring): it's already a rank-over-time
        # visualization in every data row above, so a redundant AUC-rank bar
        # there would just duplicate this column's new bottom cell.
        _draw_avg_rank_bar(axes[r_idx][product_col_idx], avg_rank_row.get("product"))
        if include_product_rank_column:
            axes[r_idx][-1].axis("off")

    # Shared legend
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
        fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.4, h_pad=0.5)
    else:
        fig.tight_layout(w_pad=0.4, h_pad=0.5)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Statistics: per-seed area under the product curve
#
# Kept here (rather than moving to summary.py) since it's shared by several
# reports that also depend on the row/grid infrastructure above
# (ff_comparison.py, casd_comparison.py, summary.py's synthetic-comparison
# pipeline), even though this module no longer produces a report of its own.
# ---------------------------------------------------------------------------
def _collect_product_auc_for_stats(
    runs_by_problem: dict[str, list[RunSeries]],
    caches_by_problem: dict[str, CurveCache],
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """Per-seed area-under-the-product-curve, bucketed for statistical testing.

    Returns ``{problem: {difficulty: {method: [auc_seed_0, …]}}}`` -- evals
    axis only, since ``x_steps`` is not a meaningful shared axis between a
    sequential and a batch variant here (see module docstring). Mirrors
    ``summary._collect_hv_for_stats``'s bucketing convention, but integrates
    the point-wise **product curve itself**
    (:func:`itcas.reporting.summary._compute_seed_product_curve`, exactly
    what the "Product" plot column shows) via
    :func:`itcas.reporting.summary._curve_area`, rather than the product of
    each metric's own area used by ``summary``'s "hypervolume".
    """
    all_metrics = _ordered_metrics()
    out: dict[str, dict[str, dict[str, list[float]]]] = {}
    for problem, runs in runs_by_problem.items():
        if not runs:
            continue
        cache = caches_by_problem.get(problem, {})
        seen_keys: set[str] = set()
        for run in runs:
            for key, y in (cache.get(run.run_name) or {}).items():
                if y is not None:
                    seen_keys.add(key)
        metrics_present = [s for s in all_metrics if s.key in seen_keys]
        if not metrics_present:
            continue
        for run in runs:
            diff = _difficulty_of(run)
            run_curves = cache.get(run.run_name) or {}
            prod = _compute_seed_product_curve(run, _AXIS, run_curves, metrics_present)
            if prod is None:
                continue
            auc = _curve_area(run.x_evals, prod)
            if auc is None:
                continue
            out.setdefault(problem, {}).setdefault(diff, {}).setdefault(run.method, []).append(auc)
    return out
