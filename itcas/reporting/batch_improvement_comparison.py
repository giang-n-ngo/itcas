"""Batch-improvement heatmap: how much does DPP batch selection help over
sequential sampling, per method family, per metric, per difficulty?

Answers a narrower, more direct question than
:mod:`itcas.reporting.batch_vs_sequential` (now trimmed to shared row/grid
infrastructure, see that module's docstring) used to: not "is batch
significantly better" (a per-(problem, difficulty) hypothesis test) but "by
how much, on average across problems, does switching sequential -> batch
change each method's mean AUC" -- a single compact heatmap rather than a
per-family grid of curve PDFs plus a Wilcoxon report.

**The 5 method families** (sequential, batch pair; ``random`` excluded --
no batch/sequential pair meaningfully compares the trivial baseline), in the
row order this report always uses (proposed method first):

    itcas       itcas_seq_ndig              itcas_ndig                  "NDIG"
    bes         bes_then_sample_lse10       bes_then_sample_lse10_batch "BES-TS-10"
    straddle    straddle_then_sample_lse10  straddle_then_sample_lse10_batch "STR-TS-10"
    eci         cas_eci                     cas_eci_batch               "ECI"
    moc_cas_hard moc_cas_hard               moc_cas_hard_batch          "MOC-CAS"

``itcas_seq_ndig``/``itcas_ndig`` are the parsed labels for
``method="itcas_seq", quality="ndig"`` / ``method="itcas", quality="ndig"``
(see ``visualize._load_run``, same convention :mod:`itcas.reporting.summary`'s
synthetic-comparison pipeline and the now-deleted ``ndig_comparison.py`` used).
The other four pairs are read verbatim, not derived from
``configs/final_methods.json``'s ``"families"`` mapping (that mapping's
``bes``/``straddle`` entries pair with the ``lse50`` Stage-1 split, not
``lse10`` -- ``lse10`` is the split every other headline report in this
package, ``ff_comparison.py``/``casd_comparison.py``, already uses as its
baseline, and is what the user asked for here); both variants are confirmed
present in that config's flat ``"methods"`` list, so the names are real,
just not reachable through ``variant_pair``.

**Scope: standard synthetic problems, 4 shared difficulty levels only.** Uses
:func:`itcas.reporting.summary._synthetic_problems` (default
``configs/final_problems.json``, already excludes the two real-world
problems -- ``spacecraft_formation_flying_a1``, ``casd_llm``, see
``batch_vs_sequential._REAL_WORLD_PROBLEMS``) and the shared
``p0_01``/``p0_05``/``p0_10``/``p0_20`` difficulty scale -- *not* the FF/CASD
reports' own per-problem difficulty scales (10 and 4 levels respectively).
This is a deliberate reading of "4 difficulty levels" (the only one that
already exists cleanly as a first-class concept in this codebase, mirroring
``summary.py``'s synthetic-comparison pipeline) rather than an oversight; if
a different scope was intended (e.g. including the two real-world problems'
own scales), that would need a follow-up.

**Axis.** Evaluations only (``RunSeries.x_evals``), never ``x_steps`` -- see
``batch_vs_sequential.py``'s module docstring for the detailed argument
(``x_evals`` counts individually-evaluated points regardless of
``batch_size``; ``x_steps`` counts algorithmic iterations, which is not a
fair shared unit between a method's own sequential and batch variant).

**Aggregate-then-ratio, not mean-of-ratios.** For each (family, difficulty,
column) cell: every problem's own mean-AUC-across-seeds is first averaged
*across problems* separately for the sequential side and the batch side
(``agg_seq``, ``agg_batch``), and only then is the percentage change taken
(``100 * (agg_batch - agg_seq) / agg_seq``) -- not a mean of each problem's
own ``(batch - seq) / seq`` ratio. Averaging ratios first would let one
problem with a tiny (near-zero) sequential AUC dominate the mean with an
enormous, non-representative ratio; aggregating the raw AUCs first is robust
to that. A cell is ``None`` (rendered as a fixed grey, not a 0%) when either
side has no data at all, or when ``agg_seq == 0`` (the ratio is undefined).

**FCFD sign flip.** Every column except FCFD is higher-is-better, so its raw
percentage change already means "positive = batch improved". FCFD is the one
lower-is-better metric (:class:`MetricSpec` with
``higher_is_better=False``), so its raw percentage change is negated before
plotting -- exactly the sign-flip convention every other figure in this
package already applies for FCFD (see
``summary._draw_metric_bar_panel``'s ``ascending_is_better=not
spec.higher_is_better``, and the relative-AUC figures' ``↑``/``↓`` column
arrows) -- so that, uniformly across every panel of the heatmap, positive
(blue) always means "batch improved" and negative (red) always means "batch
regressed", regardless of that metric's own better-direction.

**No statistics.** This report is a single descriptive heatmap; it
deliberately does *not* add a Friedman/Wilcoxon significance report the way
``batch_vs_sequential.py`` used to (that reinstated hypothesis-testing
machinery answered a different question -- "is batch significantly better"
-- than the one this report answers -- "how much better, on average"). This
is a deliberate scope decision, not an oversight; a stats report could be
added later as a companion output if wanted.

**AUC caching.** Per problem, the raw per-iteration curves are computed via
:func:`itcas.reporting.summary._precompute_cached` (the disk-backed curve
cache -- reused verbatim from the ``ff_comparison.py``/``casd_comparison.py``
convention, same ``auc_cache_dir`` parameter/CLI flag, same default
``Path(input_dir).parent / "auc_cache"``), which is genuinely the expensive
step this report would otherwise repeat on every invocation. The per-seed
AUC scalar cache (:mod:`itcas.reporting.auc_cache`'s ``*_auc_cache.json``,
fed into ``ranking.py`` via ``auc_cache=``) is deliberately **not** wired
into :func:`itcas.reporting.ranking.mean_auc_lists_over_rows` here, unlike
every other report that uses it: that cache's ``run_name`` keys are only
guaranteed unique *within* one problem's own cache file (``run_name`` is
just ``jsonl_path.stem``, see ``visualize._load_run``, with no problem/method
prefix) -- exactly why ``save_auc_cache``/``save_curve_cache`` are always
called per-problem, one file per problem, never merged. This report's own
``rows`` (see :func:`summarize_batch_improvement_comparison`) span *multiple*
problems at once per difficulty (``batch_vs_sequential._rows_by_problem``'s
"standard layout"), so merging every present problem's own ``*_auc_cache.json``
into one flat dict before passing it to ``mean_auc_lists_over_rows`` would
risk a same-named run from a different problem silently returning the wrong
cached AUC. The curve cache above already provides the real speedup (the
per-seed AUC integration ``ranking.py`` still does on every call is cheap,
per :mod:`itcas.reporting.auc_cache`'s own docstring), so this omission
costs negligible performance for a real correctness guarantee.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .ranking import mean_auc_lists_over_rows

_DEFAULT_PROBLEMS_CONFIG = "configs/final_problems.json"
_DEFAULT_OUTPUT_DIR = "results/batch_improvement_comparison"
_AXIS = "evals"  # the only axis this report ever plots -- see module docstring.

# (family_key, sequential_method, batch_method, display_label) -- see module
# docstring. Row order top-to-bottom in the heatmap; proposed method first.
FAMILIES: list[tuple[str, str, str, str]] = [
    ("itcas", "itcas_seq_ndig", "itcas_ndig", "NDIG"),
    ("bes", "bes_then_sample_lse10", "bes_then_sample_lse10_batch", "BES-TS-10"),
    ("straddle", "straddle_then_sample_lse10", "straddle_then_sample_lse10_batch", "STR-TS-10"),
    ("eci", "cas_eci", "cas_eci_batch", "ECI"),
    ("moc_cas_hard", "moc_cas_hard", "moc_cas_hard_batch", "MOC-CAS"),
]

ALL_METHODS: list[str] = [m for fam in FAMILIES for m in (fam[1], fam[2])]

# The shared "standard problem" difficulty scale (see module docstring's
# "Scope" section) -- NOT the FF/CASD reports' own per-problem scales.
DIFFICULTIES: tuple[str, ...] = ("p0_01", "p0_05", "p0_10", "p0_20")


# ---------------------------------------------------------------------------
# Aggregation: per-row mean AUCs -> {column: {family: {difficulty: pct|None}}}
# ---------------------------------------------------------------------------
def _pct_change_by_column(
    mean_auc_by_diff: dict[str, dict[str, dict[str, list[float]]]],
    metric_higher_is_better: dict[str, bool],
) -> dict[str, dict[str, dict[str, Optional[float]]]]:
    """Reduce per-(difficulty, column, method) mean-AUC lists to per-(column,
    family, difficulty) percentage-change cells (see module docstring's
    "Aggregate-then-ratio" and "FCFD sign flip" sections for the exact
    definition).

    ``mean_auc_by_diff`` is ``{difficulty: {column_key: {method:
    [per_problem_mean_auc, ...]}}}`` -- one
    :func:`itcas.reporting.ranking.mean_auc_lists_over_rows` call's output
    per difficulty. ``metric_higher_is_better`` maps every column key
    (including ``"product"``) to its direction.
    """
    out: dict[str, dict[str, dict[str, Optional[float]]]] = {}
    for column_key, higher_is_better in metric_higher_is_better.items():
        out[column_key] = {}
        for family_key, seq_method, batch_method, _label in FAMILIES:
            out[column_key][family_key] = {}
            for diff in DIFFICULTIES:
                col_data = mean_auc_by_diff.get(diff, {}).get(column_key, {})
                seq_vals = col_data.get(seq_method) or []
                batch_vals = col_data.get(batch_method) or []
                if not seq_vals or not batch_vals:
                    out[column_key][family_key][diff] = None
                    continue
                agg_seq = sum(seq_vals) / len(seq_vals)
                agg_batch = sum(batch_vals) / len(batch_vals)
                if agg_seq == 0:
                    out[column_key][family_key][diff] = None
                    continue
                raw_pct = 100.0 * (agg_batch - agg_seq) / agg_seq
                out[column_key][family_key][diff] = raw_pct if higher_is_better else -raw_pct
    return out


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
def summarize_batch_improvement_comparison(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    problems_config: str | Path = _DEFAULT_PROBLEMS_CONFIG,
    auc_cache_dir: str | Path | None = None,
) -> list[str]:
    """Write ``batch_improvement_comparison_heatmap.pdf`` under ``output_dir``.

    ``auc_cache_dir`` defaults to ``Path(input_dir).parent / "auc_cache"``
    when ``None`` -- same convention as ``ff_comparison.py``/
    ``casd_comparison.py`` (see module docstring's "AUC caching" section for
    exactly what is and isn't reused from that infrastructure).
    """
    from .batch_vs_sequential import _collect_family_runs, _rows_by_problem
    from .summary import _metrics_present_in_rows, _ordered_metrics, _plot_batch_improvement_heatmap, _precompute_cached, _synthetic_problems

    input_path = Path(input_dir)
    out_dir = Path(output_dir) if output_dir is not None else Path(_DEFAULT_OUTPUT_DIR)
    resolved_auc_cache_dir = (
        Path(auc_cache_dir) if auc_cache_dir is not None else Path(input_dir).parent / "auc_cache"
    )

    problems = _synthetic_problems(problems_config)

    runs_by_problem = _collect_family_runs(input_path, problems, ALL_METHODS)
    caches_by_problem = {
        p: _precompute_cached(runs, resolved_auc_cache_dir, p)
        for p, runs in runs_by_problem.items() if runs
    }

    mean_auc_by_diff: dict[str, dict[str, dict[str, list[float]]]] = {}
    metric_keys_present: set[str] = set()
    for diff in DIFFICULTIES:
        rows = _rows_by_problem(problems, runs_by_problem, caches_by_problem, diff)
        if not rows:
            continue
        for spec in _metrics_present_in_rows(rows):
            metric_keys_present.add(spec.key)
        # auc_cache intentionally omitted -- see module docstring's "AUC
        # caching" section for why merging per-problem AUC caches here would
        # be unsafe.
        mean_auc_by_diff[diff] = mean_auc_lists_over_rows(rows, ALL_METHODS, _AXIS)

    metrics_present = [s for s in _ordered_metrics() if s.key in metric_keys_present]
    if not metrics_present:
        return []

    # No "product" column here -- the heatmap is a 2x2 grid of the four
    # registered metrics only, see _plot_batch_improvement_heatmap.
    metric_higher_is_better = {s.key: s.higher_is_better for s in metrics_present}

    pct_by_column = _pct_change_by_column(mean_auc_by_diff, metric_higher_is_better)

    out_path = out_dir / "batch_improvement_comparison_heatmap.pdf"
    ok = _plot_batch_improvement_heatmap(
        pct_by_column, FAMILIES, list(DIFFICULTIES), metrics_present, out_path
    )
    return [str(ok)] if ok is not None else []


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser("itcas.batch_improvement_comparison")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--problems-config", type=str, default=_DEFAULT_PROBLEMS_CONFIG,
        help="Path to the final-problems config listing the synthetic problem suite.",
    )
    parser.add_argument(
        "--auc-cache-dir", type=str, default=None, dest="auc_cache_dir",
        help=(
            "Directory for the disk-backed curve cache (see "
            "itcas.reporting.auc_cache). Defaults to <input-dir's parent>/auc_cache."
        ),
    )
    args = parser.parse_args(argv)

    paths = summarize_batch_improvement_comparison(
        args.input_dir,
        output_dir=args.output_dir,
        problems_config=args.problems_config,
        auc_cache_dir=args.auc_cache_dir,
    )
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
