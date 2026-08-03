"""Six-method comparison on the CASD problem (``casd_llm``).

"The CASD problem" is a **real-world** problem (backed by a live LLM +
judge-model server, see ``contexts/llm_application.md`` and
``itcas.pipeline.problems.ContextAwareSafeDecoding``) that -- like ``the FF
problem`` (``spacecraft_formation_flying_a1``, see
:mod:`itcas.reporting.ff_comparison`, whose structure this module mirrors) --
uses its own per-problem difficulty scale (``p1_00``..``p4_00``, four
hand-picked ``(tau_safety, tau_utility)`` levels, see
``configs/thresholds.json["casd_llm"]`` and
``itcas.pipeline.problems.ContextAwareSafeDecoding``'s "Difficulty levels"
section) instead of the shared ``p0_01``/``p0_05``/``p0_10``/``p0_20`` scale
every "standard" (synthetic, closed-form) problem uses. This report compares
six distinct algorithms on it, all at every one of its four difficulty
levels.

* ``itcas_ndig`` -- **the proposed method**. Full ITCAS (QD-DPP greedy batch
  selection, honors ``--batch_size``), quality=``ndig``. Parsed label for
  ``method="itcas", quality="ndig"`` (see ``visualize._load_run``); on disk
  under ``itcas/ndig``. This is the only **batch** method among the six: it
  logs one record per algorithmic step of up to ``batch_size``
  individually-evaluated points.
* ``bes_then_sample_lse10`` -- "BES-then-sample 10%", the Family-C
  Hybrid-Cartographer baseline (see
  ``itcas.reporting.lse_ratio_comparison``'s module docstring) with a 10%
  Stage-1/LSE budget split, forced sequential.
* ``straddle_then_sample_lse10`` -- same Family-C construction with the
  ``straddle`` Stage-1 acquisition instead of ``bes``, also forced sequential.
* ``random`` -- the trivial uniform-random baseline, forced sequential.
* ``cas_eci`` -- CAS/ECI (``contexts/cas.md``), forced sequential.
* ``moc_cas_hard`` -- MOC-CAS hard (geometric) acquisition
  (``contexts/moccas.md``), forced sequential.

**Axis.** All five baselines are forced-sequential (one record per
individual evaluation); only ``itcas_ndig`` is batch. Exactly as in
:mod:`itcas.reporting.ff_comparison` (whose rationale this mirrors), that
makes **total individual evaluations** (``RunSeries.x_evals``) the only fair
shared x-axis between a batch method and five sequential ones -- ``x_steps``
would silently rescale the sequential methods' x-axis by a factor the batch
method doesn't share. This report therefore only ever uses ``_AXIS =
"evals"``; there is no steps variant.

**Layout.** Unlike the "standard" problems (which get one PDF per shared
difficulty level, rows = problems), the CASD problem already has all four of
its own difficulty levels as natural rows -- exactly the
``spacecraft_formation_flying_a1`` layout
:func:`itcas.reporting.batch_vs_sequential._rows_by_difficulty` already
builds for other reports. Mirroring :mod:`itcas.reporting.ndig_comparison`'s
layout (one PDF per row, not one combined grid) and
:mod:`itcas.reporting.ff_comparison` (whose structure this mirrors
end-to-end), this module writes **one single-row PDF per difficulty level**
(``casd_comparison_pN_00_vs_evaluations.pdf``) plus three standalone
cross-row summary figures -- an average-rank boxplot figure, a relative-AUC
("relative ranking") boxplot figure, and a relative-AUC-by-difficulty
line+band figure, see below -- instead of a single combined grid with an
appended bottom row.

Unlike ``ff_comparison``, there is no curated row subset here: CASD only has
four difficulty levels to begin with (not ten), so all of ``_SELECTED_LEVELS
= (1, 2, 3, 4)`` get a PDF, in that (hardest-to-easiest) order -- no
selection or reordering needed. Each row's PDF still drops the trailing
**product-rank column** (``include_product_rank_column=False``, same as
``ff_comparison``): just the metric columns plus the raw-product column. No
figure in this module ever renders a ``suptitle`` -- ``plot_group_grid``
never renders one regardless of what's passed, and the three standalone
summary figures (:func:`itcas.reporting.summary._plot_avg_rank_box_figure` /
:func:`itcas.reporting.summary._plot_relative_auc_box_figure` /
:func:`itcas.reporting.summary._plot_relative_auc_by_difficulty_figure`) only
ever draw per-column titles (the metric name, plus a ``↑``/``↓`` arrow where
the column has a direction). Method names in the legend and in every summary
figure are shown via their short display labels from
:mod:`itcas.reporting.method_labels` instead of their real (long) method
names.

Each row's label is **not** the raw difficulty tag (``p2_00`` etc.) but the
level's bare ``(tau_safety, tau_utility)`` threshold pair plus the joint
feasible fraction, e.g. ``(tau_safety>=0.999, tau_utility>=-2.00)\\n~6.2%
feasible`` for level 2, read from ``configs/thresholds.json``'s ``casd_llm``
entry for that level -- see :func:`_level_row_label`. Unlike
``ff_comparison``'s ``#initials`` (a count of scanned initial points, read
from a separate ``results/ff_initial_data/ff_initial_data_scan.json`` scan
file that has no CASD equivalent -- CASD's context space is a fixed pool of
750 real prompts, not something scanned for feasible initial conditions the
way FF's Basilisk initial states are), the feasible fraction here is parsed
straight out of that level's own ``description`` string in
``configs/thresholds.json`` (the ``~X%`` figure already recorded there from
the 1000-sample real calibration run -- see
``itcas.pipeline.problems.ContextAwareSafeDecoding``'s docstring). This is
purely cosmetic (row *content* -- which runs/seeds feed each row -- is
unchanged, only the row label changes); the statistics below are unaffected
and still cover all four difficulty levels.

**Average-rank figure.** A standalone PDF (``casd_comparison_avg_rank_vs_
evaluations.pdf``) via :mod:`itcas.reporting.ranking`
(:func:`~itcas.reporting.ranking.rank_lists_over_rows`) and
:func:`itcas.reporting.summary._plot_avg_rank_box_figure`, which are both
deliberately row-agnostic so they work unchanged whether a "row" is a
difficulty level (as here) or, in some future report, a problem at a fixed
difficulty. Per the user's framing (mirrored from ``ff_comparison``): for
each metric, on each of the four CASD difficulty levels, compute the area
under that metric's curve for every run, average those areas within each
method (across its seeds present at that difficulty), then rank the six
methods against each other by that mean area (tie-aware, rank 1 = best;
direction follows the metric's ``higher_is_better``, so a smaller mean area
is rank 1 for FCFD). That gives each method one rank per level; instead of
collapsing those four ranks straight to a mean, this figure draws a
horizontal **boxplot** per method (one box per method, best-median first)
summarizing the method's rank *distribution* across the four difficulty
levels. The same pipeline runs once more for the raw point-wise **product**
of the metrics (always higher-is-better by construction), drawn under the
Product column the same way.

**Relative-ranking (relative-AUC) figure.** A second standalone PDF
(``casd_comparison_relative_auc_vs_evaluations.pdf``) via
:func:`itcas.reporting.ranking.relative_auc_ratio_lists_over_rows` and
:func:`itcas.reporting.summary._plot_relative_auc_box_figure`, over the same
four difficulty levels. Instead of a rank, each column reports every
(method, seed) AUC divided by the best per-seed AUC seen anywhere in that
difficulty level (any method, any seed), averaged over seeds to get one
ratio per (level, method) -- a ratio of 1.0 (dashed guide line) means
"matched the best seed-level AUC seen anywhere at that difficulty"; away
from 1.0 moves in that column's *worse* direction (below 1 for a
higher-is-better column, above 1 for FCFD, the one lower-is-better metric).
As with the average-rank figure above, those four per-level ratios are drawn
as a **boxplot** per method (spread across the four difficulty levels)
rather than collapsed to a single averaged bar.

**Relative-ranking-by-difficulty figure.** A third standalone PDF
(``casd_comparison_relative_auc_by_difficulty.pdf``) via
:func:`itcas.reporting.ranking.relative_auc_seed_ratios_for_row` (the raw
per-seed ratios, one call per level) and a sibling of the figure above,
:func:`itcas.reporting.summary._plot_relative_auc_by_difficulty_figure`
(mirrors ``ff_comparison``'s figure of the same name). The boxplot above
collapses the four difficulty levels into one per-level-averaged
distribution per method per column; this figure instead draws one line per
method per column, plotted across difficulty level on the x-axis, so a
method's trend as the problem gets harder/easier stays visible -- each line
point is that level's mean ratio across seeds, exactly as the boxplot above
averages within a level. Unlike the boxplot, the *within-level* seed spread
is not thrown away either: each line is wrapped in a shaded band spanning
the 25th-75th percentile (interquartile range) of that level's own per-seed
ratios, so both the across-level trend (the line) and the within-level seed
spread (the band) are visible at once. Each level's ratio list comes from
its own single-level call to ``relative_auc_seed_ratios_for_row`` (no
cross-level averaging); x-tick labels are the bare level number
(``"1"``..``"4"``), not the verbose threshold/feasible-fraction row labels
used by the grid PDFs. All six methods' lines share one legend for the whole
figure rather than per-panel labels.

**Normalized average curve figure.** A fourth standalone PDF
(``casd_comparison_normalized_avg_curve_vs_pct_budget.pdf``) via
:func:`itcas.reporting.summary.normalized_avg_curve_figure_over_rows`, a
different kind of cross-row aggregate from the three above: instead of
collapsing each level's curve to one scalar (an AUC, a rank, a ratio) before
combining across levels, this one keeps each level's full iteration-by-
iteration curve and averages those curves directly, so a reader can see
*where in the search* (early vs. late) a method's advantage shows up rather
than only its endpoint summary. Two problems specific to averaging raw
curves across rows -- which the three scalar-based figures above never have
to solve -- are handled by that function: the x-axis is **% of each run's
own evaluation budget** (``run.config["budget"]``, via
``summary._run_budget``) rather than raw evaluation count (a no-op rescale
here, since all four CASD levels already share one fixed budget -- this
matters for the synthetic pipeline's mismatched 100-/200-budget problems,
not for CASD, but the same function serves both); and each level's curves
are normalized by that level's own best-ever value anywhere (any method, any
seed, any timestep) before averaging, the same ratio-to-row-best convention
the relative-AUC figures above already use, just applied point-wise to the
whole curve instead of once to its AUC. One line + shaded IQR band per
method per column, spread shown **across the four difficulty levels**
(mirroring the by-difficulty figure's own across-level band, not an
across-seed one).

**Statistics ("dominance of NDIG").** Per CASD difficulty level -- all four:
a Friedman omnibus test over all six methods gates a Holm-Bonferroni
corrected one-sided paired Wilcoxon signed-rank test (``H1: itcas_ndig >
baseline``) against each of the five baselines individually. The per-seed
scalar under test is the **area under the point-wise product curve**
(:func:`itcas.reporting.summary._compute_seed_product_curve` integrated via
:func:`itcas.reporting.summary._curve_area`), exactly the convention
``ff_comparison``/:mod:`itcas.reporting.ndig_comparison`/
:mod:`itcas.reporting.batch_vs_sequential` use for the same
batch-vs-sequential axis mismatch -- higher is better throughout (FCFD
contributes as a reciprocal). This module writes its own Markdown renderer
(mirroring ``ff_comparison``'s five-baseline-column table) rather than
``stats.report_to_markdown``, whose prose assumes the opposite ("lower is
better") direction; the underlying Wilcoxon computation itself (``H1:
proposed > baseline``) is direction-agnostic and reused as-is via
:func:`itcas.reporting.stats.run_stats`.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

from . import ranking
from .batch_vs_sequential import (
    _collect_family_runs,
    _rows_by_difficulty,
    build_stats_report,
    plot_group_grid,
)
from .method_labels import METHOD_ABBREVIATIONS
from .metrics import RunSeries
from .stats import StatsReport, _fmt, _sig_marker
from .stats import write_stats_report as _write_stats_report
from .summary import (
    CurveCache,
    _metrics_present_in_rows,
    _plot_avg_rank_box_figure,
    _plot_relative_auc_box_figure,
    _plot_relative_auc_by_difficulty_figure,
    _precompute_cached,
    normalized_avg_curve_figure_over_rows,
)

_PROBLEM = "casd_llm"
_AXIS = "evals"  # the only axis this report ever plots/tests on -- see module docstring.
_DEFAULT_OUTPUT_DIR = "results/casd_comparison"

# All four of CASD's own difficulty levels, hardest-to-easiest -- unlike
# ff_comparison's curated 6-of-10 subset, there is no selection/reordering
# here (see module docstring's "Layout" section).
_SELECTED_LEVELS: tuple[int, ...] = (1, 2, 3, 4)

# Repo-root-relative config path, same construction as
# itcas.reporting.metrics._EXPERIMENTS_PATH.
_THRESHOLDS_CONFIG = Path(__file__).parent.parent.parent / "configs" / "thresholds.json"

# Matches the "~X%" (or "~X.Y%") joint-feasible-fraction figure embedded in
# each level's `description` string in configs/thresholds.json["casd_llm"],
# e.g. "...(~6.2% jointly feasible)" -> "6.2". See _level_row_label.
_FEASIBLE_PCT_RE = re.compile(r"~([\d.]+)%")

PROPOSED_METHOD = "itcas_ndig"
BASELINE_METHODS: tuple[str, ...] = (
    "bes_then_sample_lse10",
    "straddle_then_sample_lse10",
    "random",
    "cas_eci",
    "moc_cas_hard",
)
METHODS: tuple[str, ...] = (PROPOSED_METHOD,) + BASELINE_METHODS

# Stable hue per method (tab10), all solid -- identical roster/coloring to
# ff_comparison.py: six distinct algorithms, itcas_ndig (proposed) gets blue,
# random (the trivial baseline) gets grey.
_METHOD_STYLES: dict[str, dict] = {
    PROPOSED_METHOD: {"color": "#1f77b4", "linestyle": "-"},
    "bes_then_sample_lse10": {"color": "#ff7f0e", "linestyle": "-"},
    "straddle_then_sample_lse10": {"color": "#2ca02c", "linestyle": "-"},
    "random": {"color": "#7f7f7f", "linestyle": "-"},
    "cas_eci": {"color": "#9467bd", "linestyle": "-"},
    "moc_cas_hard": {"color": "#d62728", "linestyle": "-"},
}


# ---------------------------------------------------------------------------
# Row labels: threshold values + feasible fraction per CASD difficulty level
# ---------------------------------------------------------------------------
def _load_casd_thresholds(config_path: str | Path = _THRESHOLDS_CONFIG) -> dict[str, dict]:
    """Return ``{level_str: cfg}`` for ``_PROBLEM``, ``cfg`` being the raw
    ``configs/thresholds.json["casd_llm"][level_str]`` entry (``tau_safety``,
    ``tau_utility``, ``thresholds``, ``description``).

    ``level_str`` matches the string keys of ``configs/thresholds.json``'s
    ``casd_llm`` entry (``"1"``..``"4"``).
    """
    with Path(config_path).open() as f:
        data = json.load(f)
    return data[_PROBLEM]


def _feasible_pct(description: str) -> Optional[float]:
    """Parse the ``~X%`` joint-feasible-fraction figure out of a level's
    ``description`` string. Returns ``None`` (rather than raising) if the
    string doesn't match, so a row label can still be rendered -- with a "?"
    placeholder -- even if a future description stops following this format.
    """
    m = _FEASIBLE_PCT_RE.search(description)
    return float(m.group(1)) if m else None


def _level_row_label(level: int, thresholds_by_level: dict[str, dict]) -> str:
    """Return this level's row label: bare ``(tau_safety, tau_utility)`` plus
    the joint feasible fraction parsed from its ``description``.

    Replaces the raw difficulty tag (``p2_00``) as the row label -- see
    module docstring's "Layout" section. Mirrors
    ``ff_comparison._level_row_label``'s ``#initials`` line, but CASD has no
    scanned-initial-points equivalent (see module docstring), so the second
    line is the calibration run's own feasible-fraction figure instead.
    """
    cfg = thresholds_by_level[str(level)]
    tau_safety, tau_utility = cfg["thresholds"]
    pct = _feasible_pct(cfg.get("description", ""))
    pct_str = f"~{pct:g}% feasible" if pct is not None else "? feasible"
    return f"(τ_safety≥{tau_safety:g}, τ_utility≥{tau_utility:g})\n{pct_str}"


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def summarize_plots(
    input_dir: str | Path,
    output_dir: str | Path,
    auc_cache_dir: str | Path | None = None,
) -> tuple[list[str], list[RunSeries], CurveCache]:
    """Produce one single-row PDF per difficulty level plus the three standalone
    summary figures (see module docstring). Returns ``(paths, runs, cache)``.

    The latter two are returned so the stats step below can reuse the same
    discovered runs / precomputed metric curves instead of re-reading logs.

    When ``auc_cache_dir`` is given (see :func:`summarize_casd_comparison`),
    the same directory holds two side-by-side caches: the raw per-iteration
    curves (``<problem>_curve_cache.json``, via ``summary._precompute_cached``
    -- this is the one that matters, since ``compute_metric``/``_precompute``,
    not the trapezoidal integration, is the actual bottleneck), and the
    per-seed AUCs (``<problem>_auc_cache.json``, via
    :mod:`itcas.reporting.auc_cache`) fed into every ``ranking.*`` call below
    so it can skip recomputing ``summary._curve_area`` for any
    ``(run, axis, metric)`` already on disk. Both caches are merged back with
    anything newly computed before returning. This never changes any figure's
    content -- a cached curve/AUC is numerically identical to what
    ``compute_metric``/``_curve_area`` would compute fresh (see
    ``summary._precompute_cached`` / ``ranking._lookup_or_compute_auc``).
    """
    from .auc_cache import compute_auc_table, load_auc_cache_for_problem, save_auc_cache

    input_path = Path(input_dir)
    out_dir = Path(output_dir)

    runs_by_problem = _collect_family_runs(input_path, [_PROBLEM], list(METHODS))
    runs = runs_by_problem.get(_PROBLEM, [])
    cache = _precompute_cached(runs, auc_cache_dir, _PROBLEM) if runs else {}
    existing_auc_cache = load_auc_cache_for_problem(auc_cache_dir, _PROBLEM, runs=runs) if (runs and auc_cache_dir is not None) else {}

    paths: list[str] = []
    all_rows = _rows_by_difficulty(runs, cache)
    if not all_rows:
        return paths, runs, cache

    # Per-level (not averaged across levels) raw per-seed relative-AUC ratios
    # feeding the by-difficulty line+band plot below -- see module
    # docstring's "Relative-ranking-by-difficulty figure" section. CASD's
    # `_SELECTED_LEVELS` already covers all four levels (unlike
    # ff_comparison's curated 6-of-10 subset), so `all_rows` here already is
    # the full sweep; still computed independently of the curated-subset rows
    # below since each level's ratio list must come from its own single-row
    # `relative_auc_seed_ratios_for_row` call, with no cross-level averaging
    # (and no within-level seed averaging either -- that happens only inside
    # `_draw_metric_line_panel` when it plots the mean point / IQR band).
    relative_auc_by_level: list[tuple[str, dict[str, dict[str, list[float]]]]] = []
    for diff_label, row_runs, row_cache in all_rows:
        level_row = ranking.relative_auc_seed_ratios_for_row(
            (diff_label, row_runs, row_cache), list(METHODS), _AXIS,
            auc_cache=existing_auc_cache,
        )
        tick_label = diff_label.removeprefix("p").split("_")[0].lstrip("0") or "0"
        relative_auc_by_level.append((tick_label, level_row))
    metrics_present_all = _metrics_present_in_rows(all_rows)

    # Select + relabel to _SELECTED_LEVELS (see module docstring's "Layout"
    # section) -- row content (runs/cache) is untouched, only the label
    # changes. `diff_tags` keeps each row's raw difficulty tag (e.g.
    # "p2_00") alongside it, purely for building clean filenames below -- the
    # pretty multi-line row_label isn't filesystem-safe.
    rows_by_diff_label = {label: (runs_, cache_) for label, runs_, cache_ in all_rows}
    thresholds_by_level = _load_casd_thresholds()
    rows = []
    diff_tags = []
    for level in _SELECTED_LEVELS:
        diff_label = f"p{level}_00"
        entry = rows_by_diff_label.get(diff_label)
        if entry is None:
            continue
        row_runs, row_cache = entry
        row_label = _level_row_label(level, thresholds_by_level)
        rows.append((row_label, row_runs, row_cache))
        diff_tags.append(diff_label)
    if not rows:
        return paths, runs, cache

    # One single-row PDF per difficulty level (see module docstring's
    # "Layout" section) -- mirrors itcas.reporting.ndig_comparison's
    # one-PDF-per-row split rather than one combined grid.
    for diff_tag, row in zip(diff_tags, rows):
        out_path = out_dir / f"casd_comparison_{diff_tag}_vs_evaluations.pdf"
        ok = plot_group_grid(
            list(METHODS), _METHOD_STYLES, [row], None, out_path,
            include_product_rank_column=False, method_labels=METHOD_ABBREVIATIONS,
        )
        if ok is not None:
            paths.append(str(ok))

    # Standalone average-rank + relative-AUC ("relative ranking") boxplot
    # figures across the four difficulty levels (see module docstring) --
    # each one box per method showing that method's spread across the four
    # levels rather than just its mean.
    metrics_present = _metrics_present_in_rows(rows)

    avg_rank_lists = ranking.rank_lists_over_rows(
        rows, list(METHODS), _AXIS, auc_cache=existing_auc_cache
    )
    avg_rank_path = out_dir / "casd_comparison_avg_rank_vs_evaluations.pdf"
    ok = _plot_avg_rank_box_figure(
        avg_rank_lists, list(METHODS), _METHOD_STYLES, metrics_present, len(rows),
        avg_rank_path, method_labels=METHOD_ABBREVIATIONS,
    )
    if ok is not None:
        paths.append(str(ok))

    relative_auc_lists = ranking.relative_auc_ratio_lists_over_rows(
        rows, list(METHODS), _AXIS, auc_cache=existing_auc_cache
    )
    relative_auc_path = out_dir / "casd_comparison_relative_auc_vs_evaluations.pdf"
    ok = _plot_relative_auc_box_figure(
        relative_auc_lists, list(METHODS), _METHOD_STYLES, metrics_present, len(rows),
        relative_auc_path, method_labels=METHOD_ABBREVIATIONS,
    )
    if ok is not None:
        paths.append(str(ok))

    # Normalized-average-curve figure: one line + shaded IQR band per method
    # per column, spread across all four CASD difficulty levels -- see module
    # docstring's "Normalized average curve figure" section.
    normalized_curve_path = out_dir / "casd_comparison_normalized_avg_curve_vs_pct_budget.pdf"
    ok = normalized_avg_curve_figure_over_rows(
        rows, list(METHODS), _METHOD_STYLES, metrics_present, normalized_curve_path,
        method_labels=METHOD_ABBREVIATIONS,
    )
    if ok is not None:
        paths.append(str(ok))

    # Line-plot sibling of the relative-AUC boxplot above: one line + shaded
    # IQR band per method across all four CASD difficulty levels -- see
    # module docstring.
    by_difficulty_path = out_dir / "casd_comparison_relative_auc_by_difficulty.pdf"
    ok = _plot_relative_auc_by_difficulty_figure(
        relative_auc_by_level, list(METHODS), _METHOD_STYLES, metrics_present_all,
        by_difficulty_path, method_labels=METHOD_ABBREVIATIONS,
    )
    if ok is not None:
        paths.append(str(ok))

    if auc_cache_dir is not None and runs:
        save_auc_cache(auc_cache_dir, _PROBLEM, runs, compute_auc_table(runs, cache))

    return paths, runs, cache


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
# build_stats_report itself now lives in batch_vs_sequential.py (shared,
# byte-identical to ff_comparison.py's former copy save for which module
# constants it closed over) -- see the imported `build_stats_report`.


def _report_to_markdown(report: StatsReport) -> str:
    """Custom renderer: higher product-curve AUC is better (see module docstring).

    Mirrors ``ff_comparison._report_to_markdown``'s five-baseline-column table.
    """
    lines: list[str] = []
    lines.append(f"# CASD comparison — {PROPOSED_METHOD} dominance vs 5 baselines ({_PROBLEM})")
    lines.append("")
    lines.append(
        f"**Proposed:** `{report.proposed_method}` &nbsp;|&nbsp; "
        f"**Baselines:** {', '.join(f'`{b}`' for b in BASELINE_METHODS)} "
        f"&nbsp;|&nbsp; **α =** {report.alpha}"
    )
    lines.append("")
    lines.append(
        "Per CASD difficulty level: a **Friedman omnibus test** over all six methods "
        f"gates a **one-sided paired Wilcoxon signed-rank test** (`H1: {PROPOSED_METHOD} > "
        "baseline`) against each of the five baselines individually, Holm-Bonferroni "
        "corrected over those five baselines. The per-seed scalar is the **area under the "
        "point-wise product curve** (`summary._compute_seed_product_curve` integrated via "
        "`summary._curve_area`) on the total-individual-evaluations axis -- higher is better "
        "(higher-is-better metrics multiply directly into the product; FCFD, the one "
        "lower-is-better metric, contributes as a reciprocal; see `contexts/metrics.md`)."
    )
    lines.append("")
    lines.append(
        "Significance markers: `***` p_adj < 0.001, `**` p_adj < 0.01, `*` p_adj < 0.05, `ns` not "
        "significant. If the Friedman omnibus test does not reach significance, pairwise tests "
        "are skipped for that difficulty level (noted below)."
    )
    lines.append("")

    baseline_headers = " | ".join(f"vs `{b}`" for b in BASELINE_METHODS)
    baseline_sep = "".join(":--------------------:|" for _ in BASELINE_METHODS)
    lines.append(f"| Difficulty | Seeds | Friedman p | Friedman sig | {baseline_headers} |")
    lines.append(f"|:-----------|------:|-----------:|:------------:|{baseline_sep}")

    for g in sorted(report.groups, key=lambda x: x.difficulty):
        friedman_sig = "Yes" if g.friedman_significant else "No"
        if g.note or not g.pairwise:
            note = g.note or "no pairwise result"
            blanks = " | ".join(f"_{note}_" for _ in BASELINE_METHODS)
            lines.append(
                f"| `{g.difficulty}` | {g.n_seeds} | {_fmt(g.friedman_p)} "
                f"| {friedman_sig} | {blanks} |"
            )
            continue
        by_baseline = {pw.baseline: pw for pw in g.pairwise}
        cells = []
        for baseline in BASELINE_METHODS:
            pw = by_baseline.get(baseline)
            if pw is None:
                cells.append("—")
                continue
            sig_str = _sig_marker(pw.significant, pw.p_adj)
            cells.append(f"p_adj={_fmt(pw.p_adj)} {sig_str} (Δ={_fmt(pw.effect_median_diff)})")
        cells_str = " | ".join(cells)
        lines.append(
            f"| `{g.difficulty}` | {g.n_seeds} | {_fmt(g.friedman_p)} "
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
        f"`{b}` in **{_count_sig(b)} / {n_tested}**" for b in BASELINE_METHODS
    )
    lines.append(
        f"**Summary:** {n_tested} / {n_total} CASD difficulty levels had a significant Friedman "
        f"omnibus test (α={report.alpha}). Among those, `{PROPOSED_METHOD}` significantly "
        f"outperforms {per_baseline_summary} groups."
    )
    lines.append("")
    return "\n".join(lines)


def write_stats_report(report: StatsReport, out_dir: Path) -> tuple[Path, Path]:
    """Write ``casd_comparison_stats_report.{json,md}`` to ``out_dir``.

    Thin wrapper around :func:`itcas.reporting.stats.write_stats_report`
    (JSON write + directory handling), passing this module's own
    :func:`_report_to_markdown` since ``stats.report_to_markdown``'s prose
    assumes the opposite ("lower is better") direction -- see module
    docstring's "Statistics" section.
    """
    return _write_stats_report(
        report, out_dir, stem="casd_comparison_stats_report", markdown_fn=_report_to_markdown
    )


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
def summarize_casd_comparison(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    alpha: float = 0.05,
    auc_cache_dir: str | Path | None = None,
) -> list[str]:
    """See module docstring. ``auc_cache_dir`` defaults to
    ``Path(input_dir).parent / "auc_cache"`` when ``None`` -- a stable
    location shared across every report/problem regardless of
    ``output_dir`` (which varies per run, including throwaway smoke-test
    dirs) -- see :mod:`itcas.reporting.auc_cache`.
    """
    out_dir = Path(output_dir) if output_dir is not None else Path(_DEFAULT_OUTPUT_DIR)
    resolved_auc_cache_dir = (
        Path(auc_cache_dir) if auc_cache_dir is not None else Path(input_dir).parent / "auc_cache"
    )

    paths, runs, cache = summarize_plots(input_dir, out_dir, auc_cache_dir=resolved_auc_cache_dir)

    report = build_stats_report(
        runs, cache, problem=_PROBLEM, proposed_method=PROPOSED_METHOD, axis=_AXIS, alpha=alpha
    )
    json_p, md_p = write_stats_report(report, out_dir)
    paths.extend([str(json_p), str(md_p)])

    return paths


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser("itcas.casd_comparison")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--auc-cache-dir", type=str, default=None, dest="auc_cache_dir",
        help=(
            "Directory for the disk-backed per-seed metric-AUC cache (see "
            "itcas.reporting.auc_cache). Defaults to <input-dir's parent>/auc_cache."
        ),
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance level for the Friedman gate and Holm-Bonferroni corrected Wilcoxon tests.",
    )
    args = parser.parse_args(argv)

    paths = summarize_casd_comparison(
        args.input_dir,
        output_dir=args.output_dir,
        alpha=args.alpha,
        auc_cache_dir=args.auc_cache_dir,
    )
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
