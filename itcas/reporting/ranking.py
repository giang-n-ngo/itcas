"""Row-agnostic average-rank-of-methods (and relative-AUC) utilities.

Several reports in this package (``batch_vs_sequential``, ``ndig_comparison``,
``lse_ratio_comparison``, and now ``ff_comparison``) already share the same
``_Row = tuple[str, list[RunSeries], CurveCache]`` triple: a row label, the
slice of runs that belong to that row (mixed methods, single problem +
difficulty slice), and a metric-curve cache restricted to those runs'
``run_name``\\ s. What a "row" actually *means* varies by report -- one
difficulty level of one problem (the spacecraft layout used by
``batch_vs_sequential``/``ndig_comparison``), one problem at a fixed
difficulty (the "standard" layout used by those same modules), or -- as used
by :mod:`itcas.reporting.ff_comparison` -- one of the ten FF-specific
difficulty levels of the single ``spacecraft_formation_flying_a1`` problem.

This module deliberately does not know or care which of those a row
represents. It only consumes the row triple's shape, so any current or
future report that already builds ``_Row``\\ s (via
``batch_vs_sequential._rows_by_problem`` / ``_rows_by_difficulty``, or any
future "rows = problems at a fixed difficulty" variant) can reuse it as-is --
including a hypothetical future report ranking methods across *problems* at
one fixed difficulty, which would call this same function with a completely
different notion of "row" and get a correct answer without any code change
here.

Ranking pipeline (per the user's framing, reproduced verbatim): "For each
metric on each problem [here: each row], the ranks are determined by
calculating the area under the curves for each run, averaging across all
runs for the same method, then comparing between methods to rank. Then we
average the ranks across all difficulty levels [here: across rows]." Spelled
out:

1. For each row and each metric, compute the per-seed area under that
   metric's curve (:func:`itcas.reporting.summary._curve_area`, trapezoidal,
   on the run's own ``x_evals``/``x_steps`` depending on ``axis``).
2. Average those per-seed areas within each method (across that method's
   seeds present in this row) to get one scalar "mean AUC" per method.
3. Rank the methods within the row by that mean AUC, tie-aware
   (``scipy.stats.rankdata(..., method="average")``), rank 1 = best. Ranking
   direction follows ``MetricSpec.higher_is_better``: for a higher-is-better
   metric, the largest mean AUC gets rank 1; for a lower-is-better metric
   (FCFD), the smallest mean AUC gets rank 1. A row contributes no rank for a
   metric where fewer than two methods have any data (nothing to compare).
4. Average each method's per-row ranks across every row where it was ranked
   for that metric, giving the final ``{metric_key: {method: avg_rank}}``.

The same pipeline additionally runs once more for the **raw product of the
metrics** (:func:`itcas.reporting.summary._compute_seed_product_curve` --
the point-wise product across ``metrics`` at each iteration, already
direction-corrected there: FCFD contributes as a reciprocal, so a larger
product is always better), whose per-seed area is ranked and averaged across
rows exactly like any other metric and reported under the synthetic key
``"product"`` in the returned dict. This is not one of ``metrics`` and has no
``MetricSpec`` -- it is always higher-is-better by construction.

Only ``methods`` passed in are ever ranked; any other method present in the
rows' runs (e.g. an LSE-ratio sibling not part of this particular comparison)
is ignored.

:func:`relative_auc_ratios_over_rows` is a sibling entry point over the same
``Row`` triples and the same per-row AUC pipeline, reporting each method's
AUC *relative to the best AUC seen anywhere in that row* (any method, any
seed) instead of a rank -- see its own docstring for the exact definition.
"""
from __future__ import annotations

from .metrics import MetricSpec, RunSeries
from .summary import CurveCache, _compute_seed_product_curve, _curve_area, _ordered_metrics

# Same shape as batch_vs_sequential._Row: (row_label, runs for this row
# (mixed methods, single slice), cache restricted to those runs' run_names).
Row = tuple[str, list[RunSeries], CurveCache]


def average_ranks_over_rows(
    rows: list[Row],
    methods: list[str],
    axis: str,
    metrics: list[MetricSpec] | None = None,
) -> dict[str, dict[str, float]]:
    """Return ``{column_key: {method: avg_rank}}`` (rank 1 = best; see module docstring).

    ``column_key`` is either a ``MetricSpec.key`` from ``metrics`` or the
    synthetic key ``"product"`` (the raw point-wise product of those metrics,
    always higher-is-better -- see module docstring). ``rows`` is
    intentionally opaque -- see the module docstring for why. Any method that
    never has data for a given column across every row is simply absent from
    that column's inner dict (never assigned a fabricated rank).
    """
    from scipy.stats import rankdata  # lazy import, mirrors itcas.reporting.stats

    if metrics is None:
        metrics = _ordered_metrics()
    method_set = set(methods)

    # per_row_ranks[column_key][method] -> list of per-row ranks (1 = best)
    per_row_ranks: dict[str, dict[str, list[float]]] = {spec.key: {} for spec in metrics}
    per_row_ranks["product"] = {}

    def _rank_and_record(column_key: str, mean_by_method: dict[str, float], higher_is_better: bool) -> None:
        if len(mean_by_method) < 2:
            return  # nothing to rank against in this row for this column
        present_methods = list(mean_by_method.keys())
        values = [mean_by_method[m] for m in present_methods]
        # rankdata assigns rank 1 to the smallest value; negate for
        # higher-is-better columns so rank 1 lands on the largest value.
        ranked_input = [-v for v in values] if higher_is_better else values
        ranks = rankdata(ranked_input, method="average")
        for m, r in zip(present_methods, ranks):
            per_row_ranks[column_key].setdefault(m, []).append(float(r))

    for _row_label, row_runs, cache in rows:
        method_runs: dict[str, list[RunSeries]] = {}
        for run in row_runs:
            if run.method in method_set:
                method_runs.setdefault(run.method, []).append(run)

        for spec in metrics:
            mean_auc_by_method: dict[str, float] = {}
            for method in methods:
                aucs: list[float] = []
                for run in method_runs.get(method, []):
                    y = (cache.get(run.run_name) or {}).get(spec.key)
                    if y is None:
                        continue
                    x_axis = run.x_evals if axis == "evals" else run.x_steps
                    auc = _curve_area(x_axis, y)
                    if auc is not None:
                        aucs.append(auc)
                if aucs:
                    mean_auc_by_method[method] = sum(aucs) / len(aucs)
            _rank_and_record(spec.key, mean_auc_by_method, spec.higher_is_better)

        # Raw product of the metrics -- same per-row/per-method AUC pipeline,
        # but on `_compute_seed_product_curve`'s point-wise product instead of
        # a single metric's own curve; always higher-is-better by construction.
        mean_product_auc_by_method: dict[str, float] = {}
        for method in methods:
            aucs = []
            for run in method_runs.get(method, []):
                run_curves = cache.get(run.run_name) or {}
                prod = _compute_seed_product_curve(run, axis, run_curves, metrics)
                if prod is None:
                    continue
                x_axis = run.x_evals if axis == "evals" else run.x_steps
                auc = _curve_area(x_axis, prod)
                if auc is not None:
                    aucs.append(auc)
            if aucs:
                mean_product_auc_by_method[method] = sum(aucs) / len(aucs)
        _rank_and_record("product", mean_product_auc_by_method, higher_is_better=True)

    out: dict[str, dict[str, float]] = {}
    for column_key, method_ranks in per_row_ranks.items():
        if not method_ranks:
            continue
        out[column_key] = {m: sum(rs) / len(rs) for m, rs in method_ranks.items() if rs}
    return out


def relative_auc_ratios_over_rows(
    rows: list[Row],
    methods: list[str],
    axis: str,
    metrics: list[MetricSpec] | None = None,
) -> dict[str, dict[str, float]]:
    """Return ``{column_key: {method: avg_relative_ratio}}`` -- see below.

    A sibling of :func:`average_ranks_over_rows` over the same ``Row`` triples,
    reporting *relative AUC* instead of rank:

    1. For each row and each metric (plus the synthetic ``"product"`` key,
       exactly as in :func:`average_ranks_over_rows`), find that row's best
       per-seed AUC **regardless of method or seed** -- the maximum area for a
       higher-is-better column (a metric with ``higher_is_better=True``, or
       ``"product"``, always higher-is-better by construction), the minimum
       for a lower-is-better one (currently only FCFD).
    2. Divide every (method, seed) AUC in that row by that row's best AUC,
       then average across a method's own seeds to get one ratio per (row,
       method).
    3. Average those per-row ratios across every row where the method had
       data, giving the final ``{column_key: {method: avg_ratio}}``.

    A ratio of 1.0 means "matched the best seed-level AUC seen anywhere in
    that row"; away from 1.0 the ratio moves in the metric's *worse*
    direction -- below 1 for a higher-is-better column, above 1 for a
    lower-is-better one -- so, unlike rank, the "better" direction is not the
    same for every column (see
    :func:`itcas.reporting.summary._plot_relative_auc_figure`, which reads
    each column's own ``higher_is_better`` when sorting/drawing its bars). A
    row contributes no ratio for a column where its best AUC is exactly zero
    (division would be undefined) or where no method has any data.
    """
    if metrics is None:
        metrics = _ordered_metrics()
    method_set = set(methods)

    # per_row_ratios[column_key][method] -> list of per-row average ratios
    per_row_ratios: dict[str, dict[str, list[float]]] = {spec.key: {} for spec in metrics}
    per_row_ratios["product"] = {}

    def _ratio_and_record(
        column_key: str, aucs_by_method: dict[str, list[float]], higher_is_better: bool
    ) -> None:
        all_aucs = [a for aucs in aucs_by_method.values() for a in aucs]
        if not all_aucs:
            return
        best = max(all_aucs) if higher_is_better else min(all_aucs)
        if best == 0:
            return  # undefined ratio -- skip this row/column rather than divide by zero
        for method, aucs in aucs_by_method.items():
            if not aucs:
                continue
            ratio = sum(a / best for a in aucs) / len(aucs)
            per_row_ratios[column_key].setdefault(method, []).append(ratio)

    for _row_label, row_runs, cache in rows:
        method_runs: dict[str, list[RunSeries]] = {}
        for run in row_runs:
            if run.method in method_set:
                method_runs.setdefault(run.method, []).append(run)

        for spec in metrics:
            aucs_by_method: dict[str, list[float]] = {}
            for method in methods:
                aucs: list[float] = []
                for run in method_runs.get(method, []):
                    y = (cache.get(run.run_name) or {}).get(spec.key)
                    if y is None:
                        continue
                    x_axis = run.x_evals if axis == "evals" else run.x_steps
                    auc = _curve_area(x_axis, y)
                    if auc is not None:
                        aucs.append(auc)
                if aucs:
                    aucs_by_method[method] = aucs
            _ratio_and_record(spec.key, aucs_by_method, spec.higher_is_better)

        # Raw product of the metrics -- same per-row/per-method AUC pipeline as
        # average_ranks_over_rows' product column, always higher-is-better.
        aucs_by_method = {}
        for method in methods:
            aucs = []
            for run in method_runs.get(method, []):
                run_curves = cache.get(run.run_name) or {}
                prod = _compute_seed_product_curve(run, axis, run_curves, metrics)
                if prod is None:
                    continue
                x_axis = run.x_evals if axis == "evals" else run.x_steps
                auc = _curve_area(x_axis, prod)
                if auc is not None:
                    aucs.append(auc)
            if aucs:
                aucs_by_method[method] = aucs
        _ratio_and_record("product", aucs_by_method, higher_is_better=True)

    out: dict[str, dict[str, float]] = {}
    for column_key, method_ratios in per_row_ratios.items():
        if not method_ratios:
            continue
        out[column_key] = {m: sum(rs) / len(rs) for m, rs in method_ratios.items() if rs}
    return out
