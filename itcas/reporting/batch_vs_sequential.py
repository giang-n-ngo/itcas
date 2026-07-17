"""Batch vs. sequential comparison for the same acquisition-method family.

For each "plot group" -- either the two proposed CAS-family acquisitions
(``eci`` + ``moc_cas_hard``, viewed together since both are the ITCAS
proposal, see ``contexts/cas.md`` / ``contexts/moccas.md``) or the
``straddle`` baseline on its own -- this module renders grid figures (rows =
problems or difficulty levels, columns = metric curves + raw product +
product rank) comparing every family's batch variant against its own
sequential sibling, 4 lines per panel for the combined CAS group (2 families
x 2 variants) and 2 lines per panel for the standalone straddle group.

Method pairs are *not* hardcoded here: they are read from the ``"families"``
mapping in ``configs/final_methods.json`` (see :func:`variant_pair`), taking
the first two entries of each family's method list as
``(sequential, batch)`` -- this holds for every family in that config today
(``eci``, ``moc_cas_hard``, ``straddle``, ``bes``, ``random``, ...): the
plain method comes first, its ``_batch``-suffixed sibling second, with any
further LSE-then-sample siblings (irrelevant to this report) trailing after.

Unlike :mod:`itcas.reporting.school_comparison` (which compares *different*
algorithms and therefore has to pick one x-axis per "setting" because
LSE-then-sample's sequential mode burns a fixed Stage-1 budget that doesn't
correspond to algorithmic steps the same way batch mode does), this report
compares one algorithm's own sequential and batch variants, so **total
individual evaluations** (``RunSeries.x_evals``) is the correct shared
x-axis for both: :func:`itcas.reporting.visualize._load_run` populates
``x_evals`` as the cumulative count of individually evaluated points
(``n_eval_total = X_run.shape[0]``, or the record's own ``n_eval_total``/
``n_eval_this_iter`` fallback) for *every* run regardless of ``batch_size``,
whereas ``x_steps`` only counts algorithmic iterations (one per batch,
however large). So evaluations, not steps, are the common currency here --
including for the statistical testing below, which is why the per-seed
area-under-the-product-curve collector only ever uses this axis.

Output layout (per plot group ``{cas, straddle}``):

* One PDF per "standard" difficulty (``p0_01``, ``p0_05``, ``p0_10``,
  ``p0_20``) with rows = the 15 problems that share that difficulty scale.
* One PDF for ``spacecraft_formation_flying_a1`` -- the only problem using a
  distinct 10-level difficulty scale (``p1_00``..``p10_00``) -- with rows =
  its own difficulty levels instead of rows = problems.

Statistical testing (reinstated; see ``contexts/metrics.md``) stays
per-*family* (``eci``, ``moc_cas_hard``, ``straddle``) even though the plots
above are grouped in pairs, because a 4-way combined significance test across
two unrelated acquisition families doesn't make sense -- each family is
tested only against its own batch sibling. For every (problem, difficulty)
the per-seed scalar under test is the **area under the point-wise product
curve** (:func:`itcas.reporting.summary._compute_seed_product_curve`
integrated via :func:`itcas.reporting.summary._curve_area` -- exactly what
the "Product" plot column shows, not the product-of-each-metric's-own-area
"hypervolume" used by :mod:`itcas.reporting.summary`). Because there are
only two conditions (sequential vs batch) the Friedman omnibus gate (which
needs >= 3 methods) is skipped and a one-sided paired Wilcoxon signed-rank
test (``H1: batch > sequential``) is run directly, Holm-Bonferroni corrected
(trivial with a single comparison, kept for consistency with
:mod:`itcas.reporting.stats`). A ``SUMMARY.md`` synthesises the per-family
reports into per-problem / per-difficulty tallies and a headline conclusion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from .metrics import RunSeries
from .stats import (
    GroupResult,
    PairwiseResult,
    StatsReport,
    _fmt,
    _holm_bonferroni,
    _sig_marker,
    _wilcoxon_greater,
    report_to_json,
)
from .summary import (
    CurveCache,
    _SHORT_CURVE_LABELS,
    _compute_seed_product_curve,
    _curve_area,
    _difficulty_of,
    _min_med_max,
    _ordered_metrics,
    _per_method_product_curves,
    _precompute,
    _rank_curves_on_union_grid,
)
from .visualize import _discover_runs

_DEFAULT_METHODS_CONFIG = "configs/final_methods.json"
_DEFAULT_PROBLEMS_CONFIG = "configs/final_problems.json"
_DEFAULT_OUTPUT_DIR = "results/batch_vs_sequential"

# The three families this report targets (see module docstring). Kept as a
# tuple here (not re-deriving the seq/batch method names -- those are read
# from the families config via `variant_pair`) so the CLI default doesn't
# silently pick up every other family (bes, random, ...) also present there.
DEFAULT_FAMILIES: tuple[str, ...] = ("eci", "moc_cas_hard", "straddle")

# Which families' plots are drawn *together* in one set of panels (see module
# docstring). Statistics remain per-family regardless of this grouping.
_DEFAULT_PLOT_GROUPS: dict[str, list[str]] = {
    "cas": ["eci", "moc_cas_hard"],
    "straddle": ["straddle"],
}

# The one problem that uses a distinct per-problem difficulty scale
# (p1_00..p10_00, 10 levels) instead of the shared p0_01/p0_05/p0_10/p0_20
# scale used by every other problem in configs/final_problems.json.
_SPACECRAFT_PROBLEM = "spacecraft_formation_flying_a1"

_AXIS = "evals"  # the only axis this report ever plots/tests on -- see module docstring.

# Stable hue per family (tab10), independent of any matplotlib import at
# module scope -- sequential is plotted solid, batch dashed, in the same hue.
_TAB10: tuple[str, ...] = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def _load_json(path: str | Path) -> dict:
    with Path(path).open() as f:
        return json.load(f)


def load_families(config_path: str | Path = _DEFAULT_METHODS_CONFIG) -> dict[str, list[str]]:
    """Return the ``"families"`` mapping from the final-methods config."""
    return _load_json(config_path)["families"]


def load_problems(config_path: str | Path = _DEFAULT_PROBLEMS_CONFIG) -> list[str]:
    return _load_json(config_path)["problems"]


def variant_pair(families: dict[str, list[str]], family: str) -> tuple[str, str]:
    """Return ``(sequential_method, batch_method)`` for one family.

    Takes the first two entries of ``families[family]`` -- every family in
    ``configs/final_methods.json`` lists its plain (sequential) method first
    and its ``_batch``-suffixed sibling second, with any further
    LSE-then-sample variants trailing after (irrelevant here). Raises if that
    convention doesn't hold, rather than silently pairing the wrong methods.
    """
    entries = families.get(family)
    if not entries or len(entries) < 2:
        raise ValueError(f"family '{family}' has fewer than 2 methods: {entries!r}")
    sequential, batch = entries[0], entries[1]
    if batch != f"{sequential}_batch":
        raise ValueError(
            f"family '{family}': expected entries[1] == entries[0] + '_batch' "
            f"(got sequential={sequential!r}, batch={batch!r}); the families "
            "config's ordering convention may have changed -- update "
            "variant_pair() rather than silently mispairing methods."
        )
    return sequential, batch


def _resolve_plot_groups(target_families: list[str]) -> dict[str, list[str]]:
    """Bucket ``target_families`` into plot groups (see module docstring).

    Families named in ``_DEFAULT_PLOT_GROUPS`` are grouped as configured
    there (filtered to those actually requested). Any requested family not
    covered by that mapping falls back to its own singleton group, so custom
    ``--families`` selections never silently disappear.
    """
    target_set = set(target_families)
    assigned: set[str] = set()
    groups: dict[str, list[str]] = {}
    for group_name, fams in _DEFAULT_PLOT_GROUPS.items():
        sel = [f for f in fams if f in target_set]
        if sel:
            groups[group_name] = sel
            assigned.update(sel)
    for f in target_families:
        if f not in assigned:
            groups[f] = [f]
            assigned.add(f)
    return groups


def _group_method_styles(
    families_in_group: list[str],
    families_cfg: dict[str, list[str]],
) -> tuple[list[str], dict[str, dict]]:
    """Return ``(methods, styles)`` for one plot group.

    ``methods`` lists every (sequential, batch) pair in family order.
    ``styles[method]`` is ``{"color": ..., "linestyle": ...}`` -- each family
    gets one stable hue (tab10, in the given family order); its sequential
    variant is solid, its batch variant dashed in that same hue, so "same
    family = same color, sequential vs batch = same color different style".
    """
    methods: list[str] = []
    styles: dict[str, dict] = {}
    for i, fam in enumerate(families_in_group):
        sequential, batch = variant_pair(families_cfg, fam)
        color = _TAB10[i % len(_TAB10)]
        styles[sequential] = {"color": color, "linestyle": "-"}
        styles[batch] = {"color": color, "linestyle": "--"}
        methods.extend([sequential, batch])
    return methods, styles


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
            ax.tick_params(axis="both", labelsize=7)
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
                        transform=ax.transAxes, fontsize=8, color="grey")
            if r_idx == 0:
                arrow = "↑" if spec.higher_is_better else "↓"
                ax.set_title(f"{_SHORT_CURVE_LABELS.get(spec.key, spec.label)} {arrow}", fontsize=8)
            if r_idx == n_data_rows - 1:
                ax.set_xlabel("Total individual evaluations", fontsize=8)
            if c_idx == 0:
                ax.set_ylabel(row_label, fontsize=8)

        method_x, method_med, method_lo, method_hi = _per_method_product_curves(
            method_runs, methods, _AXIS, cache, metrics_present,
        )

        # Raw product column
        ax = axes[r_idx][product_col_idx]
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=7)
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
                    transform=ax.transAxes, fontsize=8, color="grey")
        if r_idx == 0:
            ax.set_title("Product ↑\n(raw)", fontsize=8)
        if r_idx == n_data_rows - 1:
            ax.set_xlabel("Total individual evaluations", fontsize=8)

        # Rank column (omitted entirely when include_product_rank_column=False)
        if include_product_rank_column:
            ax = axes[r_idx][-1]
            ax.grid(True, axis="y", alpha=0.25)
            ax.tick_params(axis="both", labelsize=7)
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
                        transform=ax.transAxes, fontsize=8, color="grey")
            if r_idx == 0:
                ax.set_title("Product rank\n(1 = best)", fontsize=8)
            if r_idx == n_data_rows - 1:
                ax.set_xlabel("Total individual evaluations", fontsize=8)

    # Optional bottom row: average rank per metric across `rows` (see
    # avg_rank_row in the docstring above). Absent by default, so this whole
    # block is a no-op -- and therefore byte-for-byte inert -- for every
    # existing caller that doesn't pass avg_rank_row.
    if avg_rank_row is not None:
        r_idx = n_data_rows

        def _draw_avg_rank_bar(ax, data: Optional[dict[str, float]]) -> None:
            ax.tick_params(axis="both", labelsize=7)
            if not data:
                ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="grey")
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
            ax.set_yticklabels([_label(m) for m in present_sorted], fontsize=7)
            ax.invert_yaxis()  # best (lowest avg rank) at top

        for c_idx, spec in enumerate(metrics_present):
            ax = axes[r_idx][c_idx]
            _draw_avg_rank_bar(ax, avg_rank_row.get(spec.key))
            if c_idx == 0:
                ax.set_ylabel("Avg rank", fontsize=8)

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
            f"Average rank across {n_data_rows} rows (1 = best)", fontsize=10, fontweight="bold",
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
            fontsize=8, bbox_to_anchor=(0.5, 0.0),
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
# Per-group orchestration (plots)
# ---------------------------------------------------------------------------
def summarize_group(
    group_name: str,
    families_in_group: list[str],
    input_dir: str | Path,
    problems: list[str],
    families_cfg: dict[str, list[str]],
    output_dir: str | Path | None = None,
) -> tuple[list[str], dict[str, list[RunSeries]], dict[str, CurveCache]]:
    """Produce every grid PDF for one plot group (see module docstring).

    Returns ``(paths, runs_by_problem, caches_by_problem)`` -- the latter two
    so callers can reuse the already-discovered runs / already-computed
    metric curves for the per-family statistical tests without re-reading
    run logs or recomputing metrics.
    """
    input_path = Path(input_dir)
    out_dir = Path(output_dir) if output_dir is not None else Path(_DEFAULT_OUTPUT_DIR)

    methods, method_styles = _group_method_styles(families_in_group, families_cfg)
    runs_by_problem = _collect_family_runs(input_path, problems, methods)
    caches_by_problem = {p: _precompute(runs) for p, runs in runs_by_problem.items() if runs}

    fam_desc = " + ".join(families_in_group)
    paths: list[str] = []

    # Standard layout: rows = problems, one PDF per shared difficulty level.
    standard_problems = [p for p in problems if p != _SPACECRAFT_PROBLEM]
    standard_runs = {p: runs_by_problem.get(p, []) for p in standard_problems}
    for diff in _difficulties_present(standard_runs):
        rows = _rows_by_problem(standard_problems, standard_runs, caches_by_problem, diff)
        if not rows:
            continue
        title = f"Batch vs sequential — {fam_desc} ({diff}) vs total individual evaluations"
        out_path = out_dir / f"{group_name}_{diff}_vs_evaluations.pdf"
        ok = plot_group_grid(methods, method_styles, rows, title, out_path)
        if ok is not None:
            paths.append(str(ok))

    # Spacecraft layout: rows = difficulty levels of this one problem.
    if _SPACECRAFT_PROBLEM in problems:
        sc_runs = runs_by_problem.get(_SPACECRAFT_PROBLEM, [])
        sc_cache = caches_by_problem.get(_SPACECRAFT_PROBLEM, {})
        rows = _rows_by_difficulty(sc_runs, sc_cache)
        if rows:
            title = (
                f"Batch vs sequential — {fam_desc} ({_SPACECRAFT_PROBLEM}, "
                "rows = difficulty level) vs total individual evaluations"
            )
            out_path = out_dir / f"{group_name}_{_SPACECRAFT_PROBLEM}_vs_evaluations.pdf"
            ok = plot_group_grid(methods, method_styles, rows, title, out_path)
            if ok is not None:
                paths.append(str(ok))

    return paths, runs_by_problem, caches_by_problem


# ---------------------------------------------------------------------------
# Statistics: per-seed area under the product curve
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


def _family_auc_data(
    family: str,
    families_cfg: dict[str, list[str]],
    runs_by_problem: dict[str, list[RunSeries]],
    caches_by_problem: dict[str, CurveCache],
) -> tuple[str, str, dict[str, dict[str, dict[str, list[float]]]]]:
    """Return ``(sequential, batch, auc_data)`` for one family.

    Filters ``runs_by_problem`` down to just this family's two methods before
    collecting AUCs -- ``runs_by_problem``/``caches_by_problem`` may cover a
    whole plot group (e.g. both ``eci`` and ``moc_cas_hard``), so this avoids
    re-discovering runs / re-precomputing metric curves per family.
    """
    sequential, batch = variant_pair(families_cfg, family)
    method_set = {sequential, batch}
    fam_runs = {
        p: [r for r in runs if r.method in method_set]
        for p, runs in runs_by_problem.items()
    }
    auc_data = _collect_product_auc_for_stats(fam_runs, caches_by_problem)
    return sequential, batch, auc_data


_MIN_SEEDS_FOR_WILCOXON = 5  # matches stats._wilcoxon_greater's own floor


def build_family_stats_report(
    family: str,
    sequential: str,
    batch: str,
    auc_by_problem: dict[str, dict[str, dict[str, list[float]]]],
    alpha: float = 0.05,
) -> StatsReport:
    """One-sided paired Wilcoxon (``batch > sequential``) per (problem, difficulty).

    Skips the Friedman omnibus gate (only 2 conditions here, see module
    docstring) and goes straight to the pairwise test, Holm-Bonferroni
    corrected (trivial with a single comparison, kept for consistency with
    :mod:`itcas.reporting.stats`). ``proposed_method`` is set to ``batch``
    since that's the side this test can ever find "significant" (see
    :func:`_wilcoxon_greater`'s one-sided ``alternative="greater"``).
    """
    report = StatsReport(alpha=alpha, proposed_method=batch)
    for problem in sorted(auc_by_problem):
        for diff in sorted(auc_by_problem[problem]):
            method_aucs = auc_by_problem[problem][diff]
            seq_vals = method_aucs.get(sequential, [])
            batch_vals = method_aucs.get(batch, [])
            n = min(len(seq_vals), len(batch_vals))
            group = GroupResult(
                problem=problem,
                difficulty=diff,
                axis=_AXIS,
                n_seeds=n,
                methods=[sequential, batch],
                proposed_method=batch,
                friedman_stat=float("nan"),
                friedman_p=1.0,
                friedman_significant=False,
            )
            if n < _MIN_SEEDS_FOR_WILCOXON:
                group.note = f"insufficient paired seeds ({n} < {_MIN_SEEDS_FOR_WILCOXON}) for Wilcoxon"
                report.groups.append(group)
                continue
            x = np.asarray(batch_vals[:n], dtype=float)
            y = np.asarray(seq_vals[:n], dtype=float)
            stat, p_raw = _wilcoxon_greater(x, y)
            p_adj = _holm_bonferroni([p_raw])[0]
            med_diff = float(np.nanmedian(x - y))
            group.pairwise = [
                PairwiseResult(
                    baseline=sequential,
                    stat=stat,
                    p_raw=p_raw,
                    p_adj=p_adj,
                    significant=p_adj < alpha,
                    effect_median_diff=med_diff,
                )
            ]
            report.groups.append(group)
    return report


def _family_report_to_markdown(
    report: StatsReport,
    family: str,
    sequential: str,
    batch: str,
) -> str:
    """Custom Markdown renderer (not ``stats.report_to_markdown``).

    ``stats.report_to_markdown`` hardcodes wording for the *lower*-hypervolume-
    is-better itcas-vs-baselines comparisons; this report tests the opposite
    direction (*higher* product-curve AUC is better, one-sided ``batch >
    sequential``), so it needs its own correctly-worded table rather than
    reusing that text verbatim.
    """
    lines: list[str] = []
    lines.append(f"# Batch vs Sequential — `{family}` family")
    lines.append("")
    lines.append(
        f"**Sequential:** `{sequential}` &nbsp;|&nbsp; **Batch:** `{batch}` "
        f"&nbsp;|&nbsp; **α =** {report.alpha}"
    )
    lines.append("")
    lines.append(
        "> **Note:** this report has only two conditions (sequential vs batch), so the "
        "Friedman omnibus gate (designed for ≥3 methods, see `contexts/metrics.md`) is not "
        "meaningful and is skipped throughout. Every (problem, difficulty) group below goes "
        "directly to a **one-sided paired Wilcoxon signed-rank test** "
        "(`H1: batch > sequential`) on the **area under the point-wise product curve** "
        "(`summary._compute_seed_product_curve` integrated via `summary._curve_area`) — "
        "higher is better (see `contexts/metrics.md`: higher-is-better metrics multiply "
        "directly into the product, FCFD contributes as a reciprocal), Holm-Bonferroni "
        "corrected (trivial here, a single comparison per group)."
    )
    lines.append("")
    lines.append(
        "Significance markers: `***` p_adj < 0.001, `**` p_adj < 0.01, `*` p_adj < 0.05, "
        "`ns` not significant. Because the test is one-sided in favor of batch, it can only "
        "ever flag *batch significantly better*; it can never flag sequential as "
        "significantly better even when its median AUC happens to be higher (see "
        "`SUMMARY.md` for that caveat spelled out again at the synthesis level)."
    )
    lines.append("")
    lines.append(
        "| Problem | Difficulty | Paired seeds | W stat | p (raw) | p (adj) | Sig | "
        "Median Δ (batch − sequential AUC) |"
    )
    lines.append(
        "|:--------|:-----------|--------------:|-------:|--------:|--------:|:---:|"
        "-----------------------------------:|"
    )
    for g in sorted(report.groups, key=lambda x: (x.problem, x.difficulty)):
        if g.note or not g.pairwise:
            note = g.note or "no pairwise result"
            lines.append(
                f"| `{g.problem}` | `{g.difficulty}` | {g.n_seeds} | — | — | — | — | _{note}_ |"
            )
            continue
        pw = g.pairwise[0]
        sig_str = _sig_marker(pw.significant, pw.p_adj)
        lines.append(
            f"| `{g.problem}` | `{g.difficulty}` | {g.n_seeds} "
            f"| {_fmt(pw.stat, 2)} | {_fmt(pw.p_raw)} | {_fmt(pw.p_adj)} | {sig_str} "
            f"| {_fmt(pw.effect_median_diff)} |"
        )
    lines.append("")

    n_tested = sum(1 for g in report.groups if g.pairwise)
    n_sig = sum(1 for g in report.groups if g.pairwise and g.pairwise[0].significant)
    n_insufficient = sum(1 for g in report.groups if not g.pairwise)
    pct = (100.0 * n_sig / n_tested) if n_tested else float("nan")
    lines.append(
        f"**Summary:** batch significantly outperforms sequential in **{n_sig} / {n_tested}** "
        f"tested (problem, difficulty) groups ({_fmt(pct, 1)}%, α={report.alpha})"
        + (f"; {n_insufficient} group(s) had too few paired seeds to test." if n_insufficient else ".")
    )
    lines.append("")
    return "\n".join(lines)


def write_family_stats_report(
    report: StatsReport,
    family: str,
    sequential: str,
    batch: str,
    out_dir: Path,
) -> tuple[Path, Path]:
    """Write ``<family>_stats_report.{json,md}`` to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{family}_stats_report.json"
    md_path = out_dir / f"{family}_stats_report.md"
    json_path.write_text(report_to_json(report), encoding="utf-8")
    md_path.write_text(_family_report_to_markdown(report, family, sequential, batch), encoding="utf-8")
    return json_path, md_path


# ---------------------------------------------------------------------------
# SUMMARY.md: cross-family synthesis
# ---------------------------------------------------------------------------
def _categorize_group(g: GroupResult) -> str:
    """One of ``"batch_significant"``, ``"no_significant_difference"``, ``"insufficient_data"``.

    There is no ``"sequential_significant"`` bucket: the test is one-sided in
    favor of batch (see :func:`build_family_stats_report`), so it can never
    produce that verdict by construction -- not because sequential never
    wins on raw median, but because that direction was never tested.
    """
    if g.note or not g.pairwise:
        return "insufficient_data"
    return "batch_significant" if g.pairwise[0].significant else "no_significant_difference"


def _tally(groups: list[GroupResult]) -> dict[str, int]:
    counts = {"batch_significant": 0, "no_significant_difference": 0, "insufficient_data": 0}
    for g in groups:
        counts[_categorize_group(g)] += 1
    return counts


def _tally_row(label: str, groups: list[GroupResult]) -> str:
    c = _tally(groups)
    total = sum(c.values())
    return (
        f"| `{label}` | {total} | {c['batch_significant']} | "
        f"{c['no_significant_difference']} | {c['insufficient_data']} |"
    )


def write_overall_summary(
    family_reports: dict[str, StatsReport],
    family_pairs: dict[str, tuple[str, str]],
    problems: list[str],
    out_dir: Path,
    alpha: float = 0.05,
) -> Path:
    """Write ``SUMMARY.md``: per-problem / per-difficulty tallies + headline conclusion.

    "Better" is stated in the direction defined by ``contexts/metrics.md`` and
    ``summary._compute_seed_product_curve``: higher area under the point-wise
    product curve is better (higher-is-better metrics multiply directly;
    FCFD, the one lower-is-better metric, contributes as a reciprocal).
    """
    families = sorted(family_reports)
    lines: list[str] = []
    lines.append("# Batch vs Sequential — Synthesis Summary")
    lines.append("")
    lines.append(
        f"Families tested: {', '.join(f'`{f}` (`{family_pairs[f][0]}` vs `{family_pairs[f][1]}`)' for f in families)}. "
        f"α = {alpha}."
    )
    lines.append("")
    lines.append(
        "Each (problem, difficulty) group is classified from a one-sided paired Wilcoxon "
        "test on the **area under the point-wise product curve** (higher = better, per "
        "`contexts/metrics.md`; see the per-family `*_stats_report.md` files for the raw "
        "numbers):"
    )
    lines.append("")
    lines.append(
        "* **Batch significantly better** — Holm-Bonferroni adjusted p < α for `H1: batch > sequential`."
    )
    lines.append(
        "* **No significant difference** — test ran but did not reach significance."
    )
    lines.append(
        "* **Insufficient data** — fewer than 5 paired seeds available."
    )
    lines.append(
        "* There is intentionally **no** \"sequential significantly better\" bucket: the test "
        "is one-sided in favor of batch, so it can never produce that verdict, regardless of "
        "which method's median AUC happens to be higher."
    )
    lines.append("")

    # --- Per-problem tallies -------------------------------------------------
    lines.append("## Per-problem breakdown")
    lines.append("")
    lines.append(
        "For each problem, across all its difficulty levels, tallied separately per family."
    )
    lines.append("")
    for problem in problems:
        prob_groups_any = [
            g for f in families for g in family_reports[f].groups if g.problem == problem
        ]
        if not prob_groups_any:
            continue
        lines.append(f"### `{problem}`")
        lines.append("")
        lines.append("| Family | Difficulties tested | Batch sig. better | No sig. diff. | Insufficient data |")
        lines.append("|:-------|--------------------:|-------------------:|---------------:|-------------------:|")
        for f in families:
            groups = [g for g in family_reports[f].groups if g.problem == problem]
            if groups:
                lines.append(_tally_row(f, groups))
        lines.append("")

    # --- Per-difficulty tallies ----------------------------------------------
    lines.append("## Per-difficulty-level breakdown")
    lines.append("")
    lines.append(
        "For each difficulty level, across every problem that has it, tallied separately per "
        "family. Note `spacecraft_formation_flying_a1` uses its own difficulty scale "
        "(`p1_00`..`p10_00`) shared with no other problem, so those rows necessarily reflect "
        "that single problem only."
    )
    lines.append("")
    all_diffs = sorted({g.difficulty for f in families for g in family_reports[f].groups})
    for diff in all_diffs:
        diff_groups_any = [
            g for f in families for g in family_reports[f].groups if g.difficulty == diff
        ]
        if not diff_groups_any:
            continue
        lines.append(f"### `{diff}`")
        lines.append("")
        lines.append("| Family | Problems tested | Batch sig. better | No sig. diff. | Insufficient data |")
        lines.append("|:-------|-----------------:|-------------------:|---------------:|-------------------:|")
        for f in families:
            groups = [g for g in family_reports[f].groups if g.difficulty == diff]
            if groups:
                lines.append(_tally_row(f, groups))
        lines.append("")

    # --- Overall conclusion ---------------------------------------------------
    lines.append("## Overall conclusion")
    lines.append("")
    family_pct: dict[str, float] = {}
    for f in families:
        groups = family_reports[f].groups
        c = _tally(groups)
        n_tested = c["batch_significant"] + c["no_significant_difference"]
        total = sum(c.values())
        pct = (100.0 * c["batch_significant"] / n_tested) if n_tested else float("nan")
        family_pct[f] = pct
        if n_tested == 0:
            lines.append(f"* **`{f}`** — no (problem, difficulty) group had enough paired seeds to test.")
            continue
        sentence = (
            f"* **`{f}`** — batch significantly outperforms sequential in "
            f"**{c['batch_significant']} / {n_tested}** tested combinations "
            f"({_fmt(pct, 1)}%, out of {total} total); sequential is never flagged as "
            "significantly better (the test is one-sided by construction)"
        )
        if c["insufficient_data"]:
            sentence += f"; {c['insufficient_data']} combination(s) had insufficient paired seeds"
        sentence += "."
        lines.append(sentence)
    lines.append("")

    # --- Anomalies / notable exceptions ---------------------------------------
    anomalies: list[str] = []
    for f in families:
        overall_pct = family_pct.get(f, float("nan"))
        if overall_pct != overall_pct:
            continue
        for problem in problems:
            groups = [g for g in family_reports[f].groups if g.problem == problem]
            c = _tally(groups)
            n_tested = c["batch_significant"] + c["no_significant_difference"]
            if n_tested == 0:
                continue
            prob_pct = 100.0 * c["batch_significant"] / n_tested
            if abs(prob_pct - overall_pct) >= 40.0:
                anomalies.append(
                    f"`{problem}` for `{f}`: {_fmt(prob_pct, 1)}% batch-significant vs "
                    f"{_fmt(overall_pct, 1)}% overall for that family"
                )
    valid_pcts = {f: p for f, p in family_pct.items() if p == p}
    if len(valid_pcts) >= 2:
        spread = max(valid_pcts.values()) - min(valid_pcts.values())
        if spread >= 30.0:
            best_f = max(valid_pcts, key=valid_pcts.get)
            worst_f = min(valid_pcts, key=valid_pcts.get)
            anomalies.append(
                f"family-level spread: `{best_f}` ({_fmt(valid_pcts[best_f], 1)}%) shows batch "
                f"significance far more often than `{worst_f}` ({_fmt(valid_pcts[worst_f], 1)}%) "
                "-- batch's advantage is not uniform across families."
            )
    if anomalies:
        lines.append("**Notable exceptions / anomalies:**")
        lines.append("")
        for a in anomalies:
            lines.append(f"* {a}")
    else:
        lines.append(
            "**Notable exceptions / anomalies:** none detected -- every problem's per-family "
            "batch-significance rate stayed within 40 percentage points of that family's "
            "overall rate, and the three families' overall rates stayed within 30 points of "
            "each other."
        )
    lines.append("")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "SUMMARY.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
def summarize_batch_vs_sequential(
    input_dir: str | Path,
    methods_config: str | Path = _DEFAULT_METHODS_CONFIG,
    problems_config: str | Path = _DEFAULT_PROBLEMS_CONFIG,
    families: Optional[list[str]] = None,
    output_dir: str | Path | None = None,
    alpha: float = 0.05,
) -> list[str]:
    families_cfg = load_families(methods_config)
    problems = load_problems(problems_config)
    target_families = list(families) if families is not None else list(DEFAULT_FAMILIES)
    out_dir = Path(output_dir) if output_dir is not None else Path(_DEFAULT_OUTPUT_DIR)

    groups = _resolve_plot_groups(target_families)

    paths: list[str] = []
    family_reports: dict[str, StatsReport] = {}
    family_pairs: dict[str, tuple[str, str]] = {}

    for group_name, fams in groups.items():
        group_paths, runs_by_problem, caches_by_problem = summarize_group(
            group_name, fams, input_dir, problems, families_cfg, output_dir=out_dir,
        )
        paths.extend(group_paths)

        for fam in fams:
            sequential, batch, auc_data = _family_auc_data(
                fam, families_cfg, runs_by_problem, caches_by_problem
            )
            report = build_family_stats_report(fam, sequential, batch, auc_data, alpha=alpha)
            family_reports[fam] = report
            family_pairs[fam] = (sequential, batch)
            json_p, md_p = write_family_stats_report(report, fam, sequential, batch, out_dir)
            paths.extend([str(json_p), str(md_p)])

    if family_reports:
        summary_path = write_overall_summary(
            family_reports, family_pairs, problems, out_dir, alpha=alpha
        )
        paths.append(str(summary_path))

    return paths


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser("itcas.batch_vs_sequential")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument(
        "--methods-config", type=str, default=_DEFAULT_METHODS_CONFIG,
        help="Path to the final-methods config holding the 'families' mapping.",
    )
    parser.add_argument(
        "--problems-config", type=str, default=_DEFAULT_PROBLEMS_CONFIG,
        help="Path to the final-problems config listing the problem suite.",
    )
    parser.add_argument(
        "--families", type=str, default=None,
        help=(
            "Comma-separated family names to compare (must be keys of the "
            "families mapping). Default: eci,moc_cas_hard,straddle. Plot "
            "grouping (cas={eci,moc_cas_hard}, straddle={straddle}) is fixed "
            "regardless of which subset is requested; statistics stay per-family."
        ),
    )
    parser.add_argument(
        "--output-dir", type=str, default=_DEFAULT_OUTPUT_DIR,
        help=f"Directory to write grid PDFs and stats reports to (default: {_DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance level for the Holm-Bonferroni corrected Wilcoxon tests.",
    )
    args = parser.parse_args(argv)

    families = args.families.split(",") if args.families else None
    paths = summarize_batch_vs_sequential(
        args.input_dir,
        methods_config=args.methods_config,
        problems_config=args.problems_config,
        families=families,
        output_dir=args.output_dir,
        alpha=args.alpha,
    )
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
