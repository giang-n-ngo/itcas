"""Full ITCAS/NDIG (batch) vs its two sequential NDIG siblings.

Compares three methods that all optimize the same NDIG quality signal but
differ in how points are selected:

* ``itcas_ndig`` -- the proposed method: full ITCAS (QD-DPP greedy batch
  selection, honors ``--batch_size``), quality=``ndig``. Parsed label for
  ``method="itcas", quality="ndig"`` (see ``visualize._load_run``); on disk
  under ``itcas/ndig``.
* ``itcas_seq_ndig`` -- ``itcas_seq``, the forced-sequential sibling: same
  NDIG quality machinery and multistart optimizer as ``itcas``, but always
  ``batch_size=1`` regardless of the config's ``batch_size``. On disk under
  ``itcas_seq/ndig``.
* ``cr_ndig`` -- Context-Repulsive NDIG (``contexts/sequential_ndig.md``): a
  purely-sequential variant with its own fixed quality (repulsion penalty
  against acquisition history) instead of QD-DPP batch diversity. On disk as
  ``cr_ndig`` (no quality suffix -- ``cr_ndig`` is not ``itcas``/``itcas_seq``
  so ``visualize._load_run`` never appends one).

Like :mod:`itcas.reporting.batch_vs_sequential` (whose module docstring this
mirrors), ``itcas_ndig`` logs one record per algorithmic step of up to
``batch_size`` individually-evaluated points, while its two sequential
siblings log one record per individual evaluation -- so **total individual
evaluations** (``RunSeries.x_evals``) is the only meaningful shared x-axis,
for both the plots and the statistical testing below (``x_steps`` would
silently rescale the sequential methods' x-axis by a factor the batch method
doesn't share).

Plots reuse :mod:`itcas.reporting.batch_vs_sequential`'s generic (non-family-
specific) discovery/grid-layout helpers directly, mirroring the layout
:func:`itcas.reporting.summary.summarize_synthetic_comparison` settled on --
one PDF per row rather than one combined grid, plus two standalone
cross-row summary figures:

* One single-row PDF per problem (for the "standard" difficulties) or per
  difficulty level (for each real-world problem, e.g.
  ``spacecraft_formation_flying_a1``, ``casd_llm`` -- see
  ``batch_vs_sequential._REAL_WORLD_PROBLEMS``): metric curves +
  raw product, no product-rank column (``include_product_rank_column=
  False`` -- that per-iteration "who's ahead right now" column only made
  sense across a shared grid; split one-row-per-PDF it adds nothing beyond
  the curves already shown).
* One standalone **average-rank** figure per difficulty (standard) or for
  the whole spacecraft problem: each method's average rank (1 = best) per
  metric + product, averaged across that difficulty's rows, via
  :func:`itcas.reporting.ranking.average_ranks_over_rows` and
  :func:`itcas.reporting.summary._plot_avg_rank_figure`.
* One standalone **relative-AUC ("relative ranking")** figure alongside it:
  for each metric (+ product), every (method, seed) AUC divided by the best
  AUC seen anywhere in that row, averaged over seeds then over rows, via
  :func:`itcas.reporting.ranking.relative_auc_ratios_over_rows` and
  :func:`itcas.reporting.summary._plot_relative_auc_figure`.

Statistics: unlike ``batch_vs_sequential`` (which only ever has two
conditions per family and therefore skips the Friedman gate), this report has
three methods, so it goes through the full pipeline in
:mod:`itcas.reporting.stats` (:func:`~itcas.reporting.stats.run_stats`):
per (problem, difficulty), a Friedman omnibus test over all three methods
gates a Holm-Bonferroni corrected one-sided paired Wilcoxon test
(``H1: itcas_ndig > baseline``) against each of ``itcas_seq_ndig`` and
``cr_ndig`` individually. The per-seed scalar under test is the **area under
the point-wise product curve** (:func:`itcas.reporting.summary._compute_seed_product_curve`
integrated via :func:`itcas.reporting.summary._curve_area`), exactly the
convention ``batch_vs_sequential`` uses for the same batch-vs-sequential axis
mismatch -- higher is better throughout (FCFD contributes as a reciprocal).
This module writes its own Markdown renderer rather than
``stats.report_to_markdown`` because that renderer's prose assumes the
opposite ("lower is better") direction; the underlying Wilcoxon computation
itself (``H1: proposed > baseline``) is direction-agnostic and reused as-is.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from . import ranking
from .batch_vs_sequential import (
    _AXIS,
    _REAL_WORLD_PROBLEMS,
    _collect_family_runs,
    _collect_product_auc_for_stats,
    _difficulties_present,
    _rows_by_difficulty,
    _rows_by_problem,
    load_problems,
    plot_group_grid,
)
from .stats import GroupResult, StatsReport, _fmt, _sig_marker, report_to_json, run_stats
from .summary import (
    CurveCache,
    _metrics_present_in_rows,
    _plot_avg_rank_figure,
    _plot_relative_auc_figure,
    _precompute,
)

_DEFAULT_PROBLEMS_CONFIG = "configs/final_problems.json"
_DEFAULT_OUTPUT_DIR = "results/ndig_comparison"

PROPOSED_METHOD = "itcas_ndig"
BASELINE_METHODS: tuple[str, ...] = ("itcas_seq_ndig", "cr_ndig")
METHODS: tuple[str, ...] = (PROPOSED_METHOD,) + BASELINE_METHODS

# Stable hue per method (tab10), all solid -- these are three distinct
# algorithms (not a sequential/batch pair sharing one family), so there is no
# "same color different linestyle" pairing to preserve here.
_METHOD_STYLES: dict[str, dict] = {
    PROPOSED_METHOD: {"color": "#1f77b4", "linestyle": "-"},
    "itcas_seq_ndig": {"color": "#ff7f0e", "linestyle": "-"},
    "cr_ndig": {"color": "#2ca02c", "linestyle": "-"},
}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _emit_row_pdfs(
    rows: list,
    row_kind: str,
    out_dir: Path,
    file_prefix: str,
    title_prefix: str,
) -> list[str]:
    """One single-row PDF per entry in ``rows`` (see module docstring)."""
    paths: list[str] = []
    for row_label, row_runs, row_cache in rows:
        title = f"{title_prefix} -- {row_label} ({row_kind}) vs total individual evaluations"
        out_path = out_dir / f"{file_prefix}_{row_label}_vs_evaluations.pdf"
        ok = plot_group_grid(
            list(METHODS), _METHOD_STYLES, [(row_label, row_runs, row_cache)], title, out_path,
            include_product_rank_column=False,
        )
        if ok is not None:
            paths.append(str(ok))
    return paths


def _emit_rank_summary_pdfs(
    rows: list,
    out_dir: Path,
    file_prefix: str,
) -> list[str]:
    """The standalone average-rank + relative-AUC ("relative ranking") PDFs across ``rows``."""
    paths: list[str] = []
    metrics_present = _metrics_present_in_rows(rows)

    avg_rank_row = ranking.average_ranks_over_rows(rows, list(METHODS), _AXIS)
    avg_rank_path = out_dir / f"{file_prefix}_avg_rank_vs_evaluations.pdf"
    ok = _plot_avg_rank_figure(
        avg_rank_row, list(METHODS), _METHOD_STYLES, metrics_present, len(rows),
        avg_rank_path,
    )
    if ok is not None:
        paths.append(str(ok))

    relative_auc_row = ranking.relative_auc_ratios_over_rows(rows, list(METHODS), _AXIS)
    relative_auc_path = out_dir / f"{file_prefix}_relative_auc_vs_evaluations.pdf"
    ok = _plot_relative_auc_figure(
        relative_auc_row, list(METHODS), _METHOD_STYLES, metrics_present, len(rows),
        relative_auc_path,
    )
    if ok is not None:
        paths.append(str(ok))

    return paths


def summarize_plots(
    input_dir: str | Path,
    problems: list[str],
    output_dir: str | Path,
) -> tuple[list[str], dict[str, list], dict[str, CurveCache]]:
    """Produce the PDFs (see module docstring). Returns ``(paths, runs_by_problem, caches_by_problem)``.

    The latter two are returned so the stats step below can reuse the same
    discovered runs / precomputed metric curves instead of re-reading logs.
    """
    input_path = Path(input_dir)
    out_dir = Path(output_dir)

    runs_by_problem = _collect_family_runs(input_path, problems, list(METHODS))
    caches_by_problem = {p: _precompute(runs) for p, runs in runs_by_problem.items() if runs}

    paths: list[str] = []
    title_prefix = "itcas_ndig vs itcas_seq_ndig vs cr_ndig"

    standard_problems = [p for p in problems if p not in _REAL_WORLD_PROBLEMS]
    standard_runs = {p: runs_by_problem.get(p, []) for p in standard_problems}
    for diff in _difficulties_present(standard_runs):
        rows = _rows_by_problem(standard_problems, standard_runs, caches_by_problem, diff)
        if not rows:
            continue
        paths.extend(
            _emit_row_pdfs(rows, diff, out_dir, f"ndig_comparison_{diff}", title_prefix)
        )
        paths.extend(
            _emit_rank_summary_pdfs(rows, out_dir, f"ndig_comparison_{diff}")
        )

    for rw_problem in _REAL_WORLD_PROBLEMS:
        if rw_problem not in problems:
            continue
        rw_runs = runs_by_problem.get(rw_problem, [])
        rw_cache = caches_by_problem.get(rw_problem, {})
        rows = _rows_by_difficulty(rw_runs, rw_cache)
        if rows:
            file_prefix = f"ndig_comparison_{rw_problem}"
            paths.extend(
                _emit_row_pdfs(rows, rw_problem, out_dir, file_prefix, title_prefix)
            )
            paths.extend(
                _emit_rank_summary_pdfs(rows, out_dir, file_prefix)
            )

    return paths, runs_by_problem, caches_by_problem


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def build_stats_report(
    runs_by_problem: dict,
    caches_by_problem: dict[str, CurveCache],
    alpha: float = 0.05,
) -> StatsReport:
    """Friedman-gated, Holm-corrected pairwise Wilcoxon: ``itcas_ndig`` vs each baseline.

    Reuses :func:`itcas.reporting.batch_vs_sequential._collect_product_auc_for_stats`
    (generic over any method set) to get per-seed product-curve AUCs, then
    :func:`itcas.reporting.stats.run_stats` with ``proposed_method=PROPOSED_METHOD``
    -- pairwise tests are automatically restricted to proposed-vs-baseline
    (never baseline-vs-baseline), Holm-Bonferroni corrected over the two
    baselines, exactly matching the comparison asked for.
    """
    auc_by_problem = _collect_product_auc_for_stats(runs_by_problem, caches_by_problem)
    # run_stats expects {problem: {axis: {difficulty: {method: [...]}}}};
    # _collect_product_auc_for_stats already restricts to the evals axis (see
    # its docstring) but doesn't nest an axis key, so add it here.
    data = {problem: {_AXIS: diffs} for problem, diffs in auc_by_problem.items()}
    return run_stats(data, proposed_method=PROPOSED_METHOD, alpha=alpha)


def _report_to_markdown(report: StatsReport) -> str:
    """Custom renderer: higher product-curve AUC is better (see module docstring)."""
    lines: list[str] = []
    lines.append("# ITCAS/NDIG (batch) vs itcas_seq/ndig vs cr_ndig")
    lines.append("")
    lines.append(
        f"**Proposed:** `{report.proposed_method}` &nbsp;|&nbsp; "
        f"**Baselines:** {', '.join(f'`{b}`' for b in BASELINE_METHODS)} "
        f"&nbsp;|&nbsp; **α =** {report.alpha}"
    )
    lines.append("")
    lines.append(
        "Per (problem, difficulty): a **Friedman omnibus test** over all three methods "
        "gates a **one-sided paired Wilcoxon signed-rank test** (`H1: itcas_ndig > baseline`) "
        "against each baseline individually, Holm-Bonferroni corrected over the two baselines. "
        "The per-seed scalar is the **area under the point-wise product curve** "
        "(`summary._compute_seed_product_curve` integrated via `summary._curve_area`) on the "
        "total-individual-evaluations axis -- higher is better (higher-is-better metrics "
        "multiply directly into the product; FCFD, the one lower-is-better metric, "
        "contributes as a reciprocal; see `contexts/metrics.md`)."
    )
    lines.append("")
    lines.append(
        "Significance markers: `***` p_adj < 0.001, `**` p_adj < 0.01, `*` p_adj < 0.05, `ns` not "
        "significant. If the Friedman omnibus test does not reach significance, pairwise tests "
        "are skipped for that group (noted below)."
    )
    lines.append("")
    lines.append(
        "| Problem | Difficulty | Seeds | Friedman p | Friedman sig | vs `itcas_seq_ndig` | "
        "vs `cr_ndig` |"
    )
    lines.append(
        "|:--------|:-----------|------:|-----------:|:------------:|:--------------------:|"
        ":------------:|"
    )
    for g in sorted(report.groups, key=lambda x: (x.problem, x.difficulty)):
        friedman_sig = "Yes" if g.friedman_significant else "No"
        if g.note or not g.pairwise:
            note = g.note or "no pairwise result"
            lines.append(
                f"| `{g.problem}` | `{g.difficulty}` | {g.n_seeds} | {_fmt(g.friedman_p)} "
                f"| {friedman_sig} | _{note}_ | _{note}_ |"
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
        lines.append(
            f"| `{g.problem}` | `{g.difficulty}` | {g.n_seeds} | {_fmt(g.friedman_p)} "
            f"| {friedman_sig} | {cells[0]} | {cells[1]} |"
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

    lines.append(
        f"**Summary:** {n_tested} / {n_total} (problem, difficulty) groups had a significant "
        f"Friedman omnibus test (α={report.alpha}). Among those, `itcas_ndig` significantly "
        f"outperforms `itcas_seq_ndig` in **{_count_sig('itcas_seq_ndig')} / {n_tested}** and "
        f"`cr_ndig` in **{_count_sig('cr_ndig')} / {n_tested}** groups."
    )
    lines.append("")
    return "\n".join(lines)


def write_stats_report(report: StatsReport, out_dir: Path) -> tuple[Path, Path]:
    """Write ``ndig_comparison_stats_report.{json,md}`` to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "ndig_comparison_stats_report.json"
    md_path = out_dir / "ndig_comparison_stats_report.md"
    json_path.write_text(report_to_json(report), encoding="utf-8")
    md_path.write_text(_report_to_markdown(report), encoding="utf-8")
    return json_path, md_path


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
def summarize_ndig_comparison(
    input_dir: str | Path,
    problems_config: str | Path = _DEFAULT_PROBLEMS_CONFIG,
    output_dir: str | Path | None = None,
    alpha: float = 0.05,
) -> list[str]:
    problems = load_problems(problems_config)
    out_dir = Path(output_dir) if output_dir is not None else Path(_DEFAULT_OUTPUT_DIR)

    paths, runs_by_problem, caches_by_problem = summarize_plots(input_dir, problems, out_dir)

    report = build_stats_report(runs_by_problem, caches_by_problem, alpha=alpha)
    json_p, md_p = write_stats_report(report, out_dir)
    paths.extend([str(json_p), str(md_p)])

    return paths


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser("itcas.ndig_comparison")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument(
        "--problems-config", type=str, default=_DEFAULT_PROBLEMS_CONFIG,
        help="Path to the final-problems config listing the problem suite.",
    )
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance level for the Friedman gate and Holm-Bonferroni corrected Wilcoxon tests.",
    )
    args = parser.parse_args(argv)

    paths = summarize_ndig_comparison(
        args.input_dir,
        problems_config=args.problems_config,
        output_dir=args.output_dir,
        alpha=args.alpha,
    )
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
