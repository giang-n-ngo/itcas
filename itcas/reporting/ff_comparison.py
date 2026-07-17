"""Six-method comparison on the FF problem (``spacecraft_formation_flying_a1``).

"The FF problem" is the only problem in this codebase that uses its own
10-level difficulty scale (``p1_00``..``p10_00``, see
``itcas.pipeline.problems``), instead of the shared ``p0_01``/``p0_05``/
``p0_10``/``p0_20`` scale every other problem uses. This report compares six
distinct algorithms on it, all at every difficulty level:

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
:mod:`itcas.reporting.ndig_comparison` (whose rationale this mirrors), that
makes **total individual evaluations** (``RunSeries.x_evals``) the only fair
shared x-axis between a batch method and five sequential ones -- ``x_steps``
would silently rescale the sequential methods' x-axis by a factor the batch
method doesn't share. This report therefore only ever uses ``_AXIS =
"evals"``; there is no steps variant.

**Layout.** Unlike the "standard" problems (which get one PDF per shared
difficulty level, rows = problems), the FF problem already has all ten of its
own difficulty levels as natural rows for a *single* figure -- rows = FF
difficulty level, exactly the ``spacecraft_formation_flying_a1`` layout
:func:`itcas.reporting.batch_vs_sequential._rows_by_difficulty` already
builds for other reports. So this module writes exactly **one** grid PDF
(``ff_comparison_vs_evaluations.pdf``), not one per difficulty.

The plotted rows are a **curated subset and order**, not all ten: only
``_SELECTED_LEVELS`` (currently levels 2, 9, 3, 5, 6, 4, in that exact order)
are shown, one row per level in that order -- not the full ten, and not
sorted by level number. The trailing **product-rank column** is also dropped
from this plot (``include_product_rank_column=False``): each row keeps just
the metric columns plus the raw-product column. The figure has no
``suptitle`` (``title=None``); method names in the legend and the
average-rank row are shown via their short display labels from
:mod:`itcas.reporting.method_labels` instead of their real (long) method
names.

Each row's label is **not** the raw difficulty tag (``p2_00`` etc.) but the
level's bare feasibility-threshold values, e.g. ``(100m, 100g, 5N)`` for
``(rmse, fuel, peak_thrust)``, read from ``configs/thresholds.json``'s
``spacecraft_formation_flying_a1`` entry for that level, plus
``#initials=N`` -- the count of scanned initial points that jointly satisfy
all three thresholds, looked up from the matching ``threshold_scan`` entry
(matched by exact ``(rmse, fuel, peak_thrust)`` threshold triple, the JSON's
own ``B_all_constraints`` field) in
``results/ff_initial_data/ff_initial_data_scan.json`` -- see
:func:`_level_row_label`. This is purely cosmetic (row *content* --
which runs/seeds feed each row -- is unchanged, only the row label and which
rows/columns are shown); the statistics below are unaffected and still cover
all ten difficulty levels.

**Average-rank bottom row.** The grid also gets one extra row at the bottom
via :mod:`itcas.reporting.ranking` (:func:`~itcas.reporting.ranking.average_ranks_over_rows`),
which is deliberately row-agnostic so it works unchanged whether a "row" is a
difficulty level (as here) or, in some future report, a problem at a fixed
difficulty. Per the user's framing: for each metric, on each of the *plotted*
FF difficulty levels (the six selected above, not all ten -- the average
should reflect what's actually shown), compute the area under that metric's
curve for every run, average those areas within each method (across its
seeds present at that difficulty), then rank the six methods against each
other by that mean area (tie-aware, rank 1 = best; direction follows the
metric's ``higher_is_better``, so a smaller mean area is rank 1 for FCFD).
Averaging those per-difficulty ranks across the six plotted levels gives each
method's final average rank per metric, drawn as a horizontal bar chart (one
bar per method, shortest = best) under each metric column. The same pipeline
runs once more for the raw point-wise **product** of the metrics (always
higher-is-better by construction), drawn under the Product column the same
way.

**Statistics ("dominance of NDIG").** Per FF difficulty level -- all ten,
independent of the plot's six-level subset above: a
Friedman omnibus test over all six methods gates a Holm-Bonferroni corrected
one-sided paired Wilcoxon signed-rank test (``H1: itcas_ndig > baseline``)
against each of the five baselines individually. The per-seed scalar under
test is the **area under the point-wise product curve**
(:func:`itcas.reporting.summary._compute_seed_product_curve` integrated via
:func:`itcas.reporting.summary._curve_area`), exactly the convention
:mod:`itcas.reporting.ndig_comparison` and
:mod:`itcas.reporting.batch_vs_sequential` use for the same
batch-vs-sequential axis mismatch -- higher is better throughout (FCFD
contributes as a reciprocal). This module writes its own Markdown renderer
(generalizing :mod:`itcas.reporting.ndig_comparison`'s two-baseline table to
five baseline columns) rather than ``stats.report_to_markdown``, whose prose
assumes the opposite ("lower is better") direction; the underlying Wilcoxon
computation itself (``H1: proposed > baseline``) is direction-agnostic and
reused as-is via :func:`itcas.reporting.stats.run_stats`.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

from . import ranking
from .batch_vs_sequential import (
    _collect_family_runs,
    _collect_product_auc_for_stats,
    _rows_by_difficulty,
    plot_group_grid,
)
from .method_labels import METHOD_ABBREVIATIONS
from .metrics import RunSeries
from .stats import StatsReport, _fmt, _sig_marker, report_to_json, run_stats
from .summary import CurveCache, _precompute

_PROBLEM = "spacecraft_formation_flying_a1"
_AXIS = "evals"  # the only axis this report ever plots/tests on -- see module docstring.
_DEFAULT_OUTPUT_DIR = "results/ff_comparison"

# Which FF difficulty levels the plot shows, and in this exact order (not
# sorted by level number, not the full ten) -- see module docstring's
# "Layout" section.
_SELECTED_LEVELS: tuple[int, ...] = (2, 9, 3, 5, 6, 4)

# Repo-root-relative config paths, same construction as
# itcas.reporting.metrics._EXPERIMENTS_PATH.
_THRESHOLDS_CONFIG = Path(__file__).parent.parent.parent / "configs" / "thresholds.json"
_INITIAL_DATA_SCAN = (
    Path(__file__).parent.parent.parent / "results" / "ff_initial_data" / "ff_initial_data_scan.json"
)

PROPOSED_METHOD = "itcas_ndig"
BASELINE_METHODS: tuple[str, ...] = (
    "bes_then_sample_lse10",
    "straddle_then_sample_lse10",
    "random",
    "cas_eci",
    "moc_cas_hard",
)
METHODS: tuple[str, ...] = (PROPOSED_METHOD,) + BASELINE_METHODS

# Stable hue per method (tab10), all solid -- these are six distinct
# algorithms (no sequential/batch pair sharing one family here, unlike
# batch_vs_sequential's paired scheme), so there's no "same color different
# linestyle" pairing to preserve. itcas_ndig (proposed) gets blue, matching
# the "proposed = blue" convention already used in ndig_comparison.py; random
# (the trivial baseline) gets grey, the conventional trivial-baseline color.
_METHOD_STYLES: dict[str, dict] = {
    PROPOSED_METHOD: {"color": "#1f77b4", "linestyle": "-"},
    "bes_then_sample_lse10": {"color": "#ff7f0e", "linestyle": "-"},
    "straddle_then_sample_lse10": {"color": "#2ca02c", "linestyle": "-"},
    "random": {"color": "#7f7f7f", "linestyle": "-"},
    "cas_eci": {"color": "#9467bd", "linestyle": "-"},
    "moc_cas_hard": {"color": "#d62728", "linestyle": "-"},
}


# ---------------------------------------------------------------------------
# Row labels: threshold values + #initials per FF difficulty level
# ---------------------------------------------------------------------------
def _load_ff_thresholds(config_path: str | Path = _THRESHOLDS_CONFIG) -> dict[str, list[float]]:
    """Return ``{level_str: [rmse_m, fuel_g, thrust_n]}`` for ``_PROBLEM``.

    ``level_str`` matches the string keys of ``configs/thresholds.json``'s
    ``spacecraft_formation_flying_a1`` entry (``"1"``..``"10"``).
    """
    with Path(config_path).open() as f:
        data = json.load(f)
    problem_cfg = data[_PROBLEM]
    return {level: cfg["thresholds"] for level, cfg in problem_cfg.items()}


def _load_threshold_scan(scan_path: str | Path = _INITIAL_DATA_SCAN) -> list[dict]:
    """Return the ``threshold_scan`` list from ``results/ff_initial_data/ff_initial_data_scan.json``."""
    with Path(scan_path).open() as f:
        data = json.load(f)
    return data["threshold_scan"]


def _lookup_b_all_constraints(
    scan: list[dict], rmse: float, fuel: float, thrust: float
) -> Optional[int]:
    """Return ``B_all_constraints`` for the scan entry matching this exact threshold triple.

    Matches by float equality (``math.isclose``) on ``rmse_threshold_m``/
    ``fuel_threshold_g``/``peak_thrust_threshold_n``. Returns ``None`` if no
    entry matches (rather than raising), so a row label can still be rendered
    -- with a "?" placeholder -- even if the scan file's threshold grid ever
    stops covering one of ``_SELECTED_LEVELS``' exact triples.
    """
    for entry in scan:
        if (
            math.isclose(entry["rmse_threshold_m"], rmse)
            and math.isclose(entry["fuel_threshold_g"], fuel)
            and math.isclose(entry["peak_thrust_threshold_n"], thrust)
        ):
            return int(entry["B_all_constraints"])
    return None


def _level_row_label(
    level: int,
    thresholds_by_level: dict[str, list[float]],
    scan: list[dict],
) -> str:
    """Return this level's row label: bare threshold values + scanned ``#initials`` count.

    Replaces the raw difficulty tag (``p2_00``) as the row label -- see module
    docstring's "Layout" section. ``#initials`` is the scan's own
    ``B_all_constraints`` field (count of scanned initial points jointly
    satisfying all three thresholds), just displayed under a clearer name.
    """
    rmse, fuel, thrust = thresholds_by_level[str(level)]
    b_all = _lookup_b_all_constraints(scan, rmse, fuel, thrust)
    b_all_str = str(b_all) if b_all is not None else "?"
    return f"({rmse:.0f}m, {fuel:.0f}g, {thrust:.0f}N)\n#initials={b_all_str}"


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def summarize_plots(
    input_dir: str | Path,
    output_dir: str | Path,
) -> tuple[list[str], list[RunSeries], CurveCache]:
    """Produce the single grid PDF (see module docstring). Returns ``(paths, runs, cache)``.

    The latter two are returned so the stats step below can reuse the same
    discovered runs / precomputed metric curves instead of re-reading logs.
    """
    input_path = Path(input_dir)
    out_dir = Path(output_dir)

    runs_by_problem = _collect_family_runs(input_path, [_PROBLEM], list(METHODS))
    runs = runs_by_problem.get(_PROBLEM, [])
    cache = _precompute(runs) if runs else {}

    paths: list[str] = []
    all_rows = _rows_by_difficulty(runs, cache)
    if not all_rows:
        return paths, runs, cache

    # Select + reorder + relabel to _SELECTED_LEVELS (see module docstring's
    # "Layout" section) -- row content (runs/cache) is untouched, only which
    # rows are kept, their order, and their label change.
    rows_by_diff_label = {label: (runs_, cache_) for label, runs_, cache_ in all_rows}
    thresholds_by_level = _load_ff_thresholds()
    scan = _load_threshold_scan()
    rows = []
    for level in _SELECTED_LEVELS:
        diff_label = f"p{level}_00"
        entry = rows_by_diff_label.get(diff_label)
        if entry is None:
            continue
        row_runs, row_cache = entry
        row_label = _level_row_label(level, thresholds_by_level, scan)
        rows.append((row_label, row_runs, row_cache))
    if not rows:
        return paths, runs, cache

    avg_rank_row = ranking.average_ranks_over_rows(rows, list(METHODS), _AXIS)

    out_path = out_dir / "ff_comparison_vs_evaluations.pdf"
    ok = plot_group_grid(
        list(METHODS), _METHOD_STYLES, rows, None, out_path,
        avg_rank_row=avg_rank_row, include_product_rank_column=False,
        method_labels=METHOD_ABBREVIATIONS,
    )
    if ok is not None:
        paths.append(str(ok))

    return paths, runs, cache


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def build_stats_report(
    runs: list[RunSeries],
    cache: CurveCache,
    alpha: float = 0.05,
) -> StatsReport:
    """Friedman-gated, Holm-corrected pairwise Wilcoxon: ``itcas_ndig`` vs each of 5 baselines.

    Reuses :func:`itcas.reporting.batch_vs_sequential._collect_product_auc_for_stats`
    (generic over any method set) to get per-seed product-curve AUCs, then
    :func:`itcas.reporting.stats.run_stats` with ``proposed_method=PROPOSED_METHOD``
    -- already generic over any number of baselines, pairwise tests are
    automatically restricted to proposed-vs-baseline (never
    baseline-vs-baseline), Holm-Bonferroni corrected over the five baselines.
    """
    auc_by_problem = _collect_product_auc_for_stats({_PROBLEM: runs}, {_PROBLEM: cache})
    # run_stats expects {problem: {axis: {difficulty: {method: [...]}}}};
    # _collect_product_auc_for_stats already restricts to the evals axis (see
    # its docstring) but doesn't nest an axis key, so add it here.
    data = {problem: {_AXIS: diffs} for problem, diffs in auc_by_problem.items()}
    return run_stats(data, proposed_method=PROPOSED_METHOD, alpha=alpha)


def _report_to_markdown(report: StatsReport) -> str:
    """Custom renderer: higher product-curve AUC is better (see module docstring).

    Generalizes ``ndig_comparison._report_to_markdown``'s hardcoded
    two-baseline-column table to a dynamic ``len(BASELINE_METHODS)``-column
    layout (five columns here).
    """
    lines: list[str] = []
    lines.append(f"# FF comparison — {PROPOSED_METHOD} dominance vs 5 baselines ({_PROBLEM})")
    lines.append("")
    lines.append(
        f"**Proposed:** `{report.proposed_method}` &nbsp;|&nbsp; "
        f"**Baselines:** {', '.join(f'`{b}`' for b in BASELINE_METHODS)} "
        f"&nbsp;|&nbsp; **α =** {report.alpha}"
    )
    lines.append("")
    lines.append(
        "Per FF difficulty level: a **Friedman omnibus test** over all six methods "
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
        f"**Summary:** {n_tested} / {n_total} FF difficulty levels had a significant Friedman "
        f"omnibus test (α={report.alpha}). Among those, `{PROPOSED_METHOD}` significantly "
        f"outperforms {per_baseline_summary} groups."
    )
    lines.append("")
    return "\n".join(lines)


def write_stats_report(report: StatsReport, out_dir: Path) -> tuple[Path, Path]:
    """Write ``ff_comparison_stats_report.{json,md}`` to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "ff_comparison_stats_report.json"
    md_path = out_dir / "ff_comparison_stats_report.md"
    json_path.write_text(report_to_json(report), encoding="utf-8")
    md_path.write_text(_report_to_markdown(report), encoding="utf-8")
    return json_path, md_path


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
def summarize_ff_comparison(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    alpha: float = 0.05,
) -> list[str]:
    out_dir = Path(output_dir) if output_dir is not None else Path(_DEFAULT_OUTPUT_DIR)

    paths, runs, cache = summarize_plots(input_dir, out_dir)

    report = build_stats_report(runs, cache, alpha=alpha)
    json_p, md_p = write_stats_report(report, out_dir)
    paths.extend([str(json_p), str(md_p)])

    return paths


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser("itcas.ff_comparison")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance level for the Friedman gate and Holm-Bonferroni corrected Wilcoxon tests.",
    )
    args = parser.parse_args(argv)

    paths = summarize_ff_comparison(
        args.input_dir,
        output_dir=args.output_dir,
        alpha=args.alpha,
    )
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
