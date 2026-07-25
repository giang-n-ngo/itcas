"""NDIG (sequential) vs NDIG-B (batch): relative-AUC-vs-synthetic-benchmarks comparison.

Narrow, single-purpose report comparing exactly two methods -- both use the
NDIG quality signal, differing only in whether points are selected in a
batch or one at a time:

* ``itcas_ndig`` -- **"NDIG-B"**, the proposed method: full ITCAS (QD-DPP
  greedy batch selection, honors ``--batch_size``), quality=``ndig``. Parsed
  label for ``method="itcas", quality="ndig"`` (see
  ``visualize._load_run``); on disk under ``itcas/ndig``.
* ``itcas_seq_ndig`` -- **"NDIG"**, the forced-sequential sibling: same NDIG
  quality machinery and multistart optimizer as ``itcas_ndig``, but always
  ``batch_size=1`` regardless of the config's ``batch_size``. On disk under
  ``itcas_seq/ndig``.

``cr_ndig`` (Context-Repulsive NDIG, a third method the previous version of
this report also compared against) is deliberately dropped -- this report
only ever shows the two methods above (see :data:`METHODS`, and
:data:`itcas.reporting.method_labels.METHOD_ABBREVIATIONS` for both methods'
short display labels, both already registered there).

**The one output: a 2x2 relative-AUC boxplot.** A single PDF
(``ndig_comparison_relative_auc_vs_synthetic_benchmarks.pdf``) via
:func:`itcas.reporting.ranking.relative_auc_ratio_lists_over_rows` and
:func:`itcas.reporting.summary._plot_relative_auc_box_grid_figure`: a 2x2
grid, one panel per metric (**no product panel** -- exactly the four metrics
:func:`itcas.reporting.summary._ordered_metrics` registers, never five), each
panel a horizontal boxplot with one box per method (``itcas_ndig``,
``itcas_seq_ndig``) showing that method's relative-AUC ratio spread. This
replaces the previous version's curve-grid PDFs, avg-rank figure, and
Friedman/Wilcoxon stats report entirely -- this module now produces exactly
one figure, mirroring :mod:`itcas.reporting.batch_improvement_comparison`'s
single-purpose-report convention rather than the full per-problem/
per-difficulty pipeline ``batch_vs_sequential``-derived reports use.

Every panel draws NDIG (``itcas_seq_ndig``) above NDIG-B (``itcas_ndig``) in
that fixed order (:data:`PLOT_ORDER`, ``sort_by_mean=False`` -- not sorted by
which one scores higher in that particular panel), and has no per-panel
y-tick method labels (``show_labels=False``): since the same two methods
repeat in all four panels, their names/colors are shown once, via a single
legend shared across the whole figure, rather than four times. The x-axis
has no numeric label either -- the whole figure gets one shared caption,
``"Avg relative AUC across synthetic problems and difficulty (1.0 = best)"``,
via ``fig.text`` below the legend.

**"Relative to all methods shown in synthetic comparison."** Per (problem,
difficulty) row, the ratio's denominator -- the "best AUC seen anywhere in
this row" -- is taken not just over the two plotted methods but over every
method :mod:`itcas.reporting.summary`'s synthetic-comparison pipeline shows
(:data:`itcas.reporting.summary.SYNTHETIC_METHODS`: ``itcas_ndig``,
``random``, ``straddle_then_sample_lse10``, ``bes_then_sample_lse10``,
``cas_eci``, ``moc_cas_hard`` -- six methods, not including
``itcas_seq_ndig``, which is not part of that comparison). This is exactly
what :func:`itcas.reporting.ranking.relative_auc_seed_ratios_for_row`'s
``pool_methods`` parameter (added alongside this report) is for: the
denominator pool (:data:`POOL_METHODS`) and the reported/plotted methods
(:data:`METHODS`) are passed separately, so a ratio of 1.0 means "matched the
best AUC any of the six synthetic-comparison methods achieved in that row" --
not merely "matched the better of NDIG/NDIG-B". Run discovery therefore
covers the union of both sets (see :data:`_ALL_NEEDED_METHODS`), even though
only :data:`METHODS` ever appears in the output.

**Rows: every synthetic problem, every shared difficulty, pooled into one
figure.** :func:`itcas.reporting.summary._synthetic_problems` (default
``configs/final_problems.json``, excludes the two real-world problems) times
the four shared standard-problem difficulty levels
(``p0_01``/``p0_05``/``p0_10``/``p0_20``, via
``batch_vs_sequential._rows_by_problem`` -- the "standard layout": one row
per problem at a fixed difficulty). Unlike
:mod:`itcas.reporting.summary`'s own synthetic-comparison pipeline (which
renders one relative-AUC figure per difficulty), this report pools every
difficulty's rows into a single list before computing ratios, so each box's
per-row ratio list has one entry per (problem, difficulty) pair -- one
overall summary figure rather than four.

**Axis.** ``itcas_ndig`` is the only batch method between the pair (one
record per algorithmic step of up to ``batch_size`` individually-evaluated
points); ``itcas_seq_ndig`` logs one record per individual evaluation. As in
:mod:`itcas.reporting.batch_vs_sequential` (whose module docstring this
mirrors), that makes **total individual evaluations** (``RunSeries.x_evals``)
the only fair shared x-axis -- this report only ever uses ``_AXIS = "evals"``.

**No statistics.** Like :mod:`itcas.reporting.batch_improvement_comparison`,
this is a single descriptive figure; it does not run the Friedman/Wilcoxon
pipeline the previous three-method version of this report used.

**AUC caching.** Curves are cached via
:func:`itcas.reporting.summary._precompute_cached` (same
``auc_cache_dir``/default-location convention as
``ff_comparison.py``/``casd_comparison.py``/``batch_improvement_comparison.py``).
The per-seed AUC scalar cache (:mod:`itcas.reporting.auc_cache`) is
deliberately **not** wired into the ``ranking.py`` call here, for the same
reason :mod:`itcas.reporting.batch_improvement_comparison` omits it: this
report's rows span multiple problems at once, and that cache's ``run_name``
keys are only unique *within* one problem's own cache file -- merging them
across problems risks a same-named run from a different problem silently
returning the wrong cached AUC (see that module's docstring, "AUC caching"
section, for the full argument).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from . import ranking
from .batch_vs_sequential import _collect_family_runs, _rows_by_problem
from .method_labels import METHOD_ABBREVIATIONS
from .summary import (
    SYNTHETIC_METHODS,
    _metrics_present_in_rows,
    _precompute_cached,
    _plot_relative_auc_box_grid_figure,
    _synthetic_problems,
)

_DEFAULT_PROBLEMS_CONFIG = "configs/final_problems.json"
_DEFAULT_OUTPUT_DIR = "results/ndig_comparison"
_AXIS = "evals"  # the only axis this report ever plots -- see module docstring.

# The shared "standard problem" difficulty scale -- see module docstring's
# "Rows" section. NOT the FF/CASD reports' own per-problem scales.
DIFFICULTIES: tuple[str, ...] = ("p0_01", "p0_05", "p0_10", "p0_20")

PROPOSED_METHOD = "itcas_ndig"       # "NDIG-B"
OTHER_METHOD = "itcas_seq_ndig"      # "NDIG"
METHODS: tuple[str, ...] = (PROPOSED_METHOD, OTHER_METHOD)

# Denominator pool for the relative-AUC ratio: every method
# summary.py's synthetic-comparison pipeline shows -- see module docstring's
# "Relative to all methods shown in synthetic comparison" section.
POOL_METHODS: tuple[str, ...] = SYNTHETIC_METHODS

# Runs actually discovered on disk: the union of what's plotted and what's
# needed for the denominator pool (itcas_seq_ndig is in METHODS but not
# POOL_METHODS; itcas_ndig is in both).
_ALL_NEEDED_METHODS: list[str] = sorted(set(METHODS) | set(POOL_METHODS))

# Fixed top-to-bottom box order for the 2x2 figure -- NDIG always above
# NDIG-B, regardless of which one scores higher in a given panel (see module
# docstring's "The one output" section). Only the plot call uses this; every
# other use of the method set (run discovery, the ratio computation itself)
# is order-independent and keeps using METHODS/PROPOSED_METHOD-first.
PLOT_ORDER: tuple[str, ...] = (OTHER_METHOD, PROPOSED_METHOD)

_METHOD_STYLES: dict[str, dict] = {
    PROPOSED_METHOD: {"color": "#1f77b4", "linestyle": "-"},
    OTHER_METHOD: {"color": "#ff7f0e", "linestyle": "-"},
}


def summarize_ndig_comparison(
    input_dir: str | Path,
    problems_config: str | Path = _DEFAULT_PROBLEMS_CONFIG,
    output_dir: str | Path | None = None,
    auc_cache_dir: str | Path | None = None,
) -> list[str]:
    """Write ``ndig_comparison_relative_auc_vs_synthetic_benchmarks.pdf`` under ``output_dir``.

    ``auc_cache_dir`` defaults to ``Path(input_dir).parent / "auc_cache"``
    when ``None`` -- same convention as
    ``ff_comparison.py``/``casd_comparison.py``/``batch_improvement_comparison.py``.
    """
    input_path = Path(input_dir)
    out_dir = Path(output_dir) if output_dir is not None else Path(_DEFAULT_OUTPUT_DIR)
    resolved_auc_cache_dir = (
        Path(auc_cache_dir) if auc_cache_dir is not None else Path(input_dir).parent / "auc_cache"
    )

    problems = _synthetic_problems(problems_config)

    runs_by_problem = _collect_family_runs(input_path, problems, _ALL_NEEDED_METHODS)
    caches_by_problem = {
        p: _precompute_cached(runs, resolved_auc_cache_dir, p)
        for p, runs in runs_by_problem.items() if runs
    }

    rows = []
    for diff in DIFFICULTIES:
        rows.extend(_rows_by_problem(problems, runs_by_problem, caches_by_problem, diff))
    if not rows:
        return []

    metrics_present = _metrics_present_in_rows(rows)

    # auc_cache intentionally omitted -- rows span multiple problems at once,
    # see module docstring's "AUC caching" section.
    relative_auc_lists = ranking.relative_auc_ratio_lists_over_rows(
        rows, list(METHODS), _AXIS, pool_methods=list(POOL_METHODS),
    )

    out_path = out_dir / "ndig_comparison_relative_auc_vs_synthetic_benchmarks.pdf"
    ok = _plot_relative_auc_box_grid_figure(
        relative_auc_lists, list(PLOT_ORDER), _METHOD_STYLES, metrics_present,
        out_path, method_labels=METHOD_ABBREVIATIONS,
    )
    return [str(ok)] if ok is not None else []


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser("itcas.ndig_comparison")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument(
        "--problems-config", type=str, default=_DEFAULT_PROBLEMS_CONFIG,
        help="Path to the final-problems config listing the synthetic problem suite.",
    )
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--auc-cache-dir", type=str, default=None, dest="auc_cache_dir",
        help=(
            "Directory for the disk-backed curve cache (see "
            "itcas.reporting.auc_cache). Defaults to <input-dir's parent>/auc_cache."
        ),
    )
    args = parser.parse_args(argv)

    paths = summarize_ndig_comparison(
        args.input_dir,
        problems_config=args.problems_config,
        output_dir=args.output_dir,
        auc_cache_dir=args.auc_cache_dir,
    )
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
