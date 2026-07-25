"""Row-agnostic average-rank-of-methods (and relative-AUC) utilities.

Several reports in this package (``batch_vs_sequential``, ``ndig_comparison``,
``lse_ratio_comparison``, and now ``ff_comparison``) already share the same
``_Row = tuple[str, list[RunSeries], CurveCache]`` triple: a row label, the
slice of runs that belong to that row (mixed methods, single problem +
difficulty slice), and a metric-curve cache restricted to those runs'
``run_name``\\ s. What a "row" actually *means* varies by report -- one
difficulty level of one real-world problem (the layout used by
``batch_vs_sequential``/``ndig_comparison`` for each problem in
``batch_vs_sequential._REAL_WORLD_PROBLEMS``), one problem at a fixed
difficulty (the "standard" layout used by those same modules), or -- as used
by :mod:`itcas.reporting.ff_comparison` / :mod:`itcas.reporting.casd_comparison`
-- one of that single real-world problem's own difficulty levels (ten for
``spacecraft_formation_flying_a1``, four for ``casd_llm``).

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

from typing import Optional

from .metrics import MetricSpec, RunSeries
from .summary import CurveCache, _compute_seed_product_curve, _curve_area, _ordered_metrics

# Same shape as batch_vs_sequential._Row: (row_label, runs for this row
# (mixed methods, single slice), cache restricted to those runs' run_names).
Row = tuple[str, list[RunSeries], CurveCache]

# run_name -> axis -> metric_key (or "product") -> AUC. Optional fast path:
# see itcas.reporting.auc_cache.AUCTable (same shape, not imported here to
# avoid a hard dependency -- any dict of this shape works).
AUCCache = dict[str, dict[str, dict[str, float]]]


def _lookup_or_compute_auc(
    auc_cache: Optional[AUCCache],
    run: RunSeries,
    axis: str,
    column_key: str,
    compute_curve,
) -> Optional[float]:
    """Return this ``(run, axis, column_key)``'s AUC, preferring ``auc_cache``.

    ``compute_curve`` is a zero-arg callable producing the raw curve (a
    metric's own curve, or the point-wise product curve) only when the cache
    misses -- so the expensive curve lookup/computation is skipped entirely
    on a cache hit. Returns ``None`` when there is no data either way (mirrors
    every call site's previous ``if y is None: continue`` / ``if auc is
    None: continue`` behavior).
    """
    cached = (auc_cache or {}).get(run.run_name, {}).get(axis, {}).get(column_key)
    if cached is not None:
        return cached
    y = compute_curve()
    if y is None:
        return None
    x_axis = run.x_evals if axis == "evals" else run.x_steps
    return _curve_area(x_axis, y)


def rank_lists_over_rows(
    rows: list[Row],
    methods: list[str],
    axis: str,
    metrics: list[MetricSpec] | None = None,
    auc_cache: Optional[AUCCache] = None,
) -> dict[str, dict[str, list[float]]]:
    """Return ``{column_key: {method: [per_row_rank, ...]}}`` (rank 1 = best; see module docstring).

    Same per-row ranking pipeline as :func:`average_ranks_over_rows`, but
    returns each column's full list of per-row ranks per method instead of
    reducing it to one mean -- the right input for a boxplot showing a
    method's rank spread across rows. :func:`average_ranks_over_rows` is now
    a thin sum/len wrapper around this.

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
                    auc = _lookup_or_compute_auc(
                        auc_cache, run, axis, spec.key,
                        lambda run=run: (cache.get(run.run_name) or {}).get(spec.key),
                    )
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
                auc = _lookup_or_compute_auc(
                    auc_cache, run, axis, "product",
                    lambda run=run: _compute_seed_product_curve(
                        run, axis, cache.get(run.run_name) or {}, metrics
                    ),
                )
                if auc is not None:
                    aucs.append(auc)
            if aucs:
                mean_product_auc_by_method[method] = sum(aucs) / len(aucs)
        _rank_and_record("product", mean_product_auc_by_method, higher_is_better=True)

    out: dict[str, dict[str, list[float]]] = {}
    for column_key, method_ranks in per_row_ranks.items():
        if not method_ranks:
            continue
        out[column_key] = {m: rs for m, rs in method_ranks.items() if rs}
    return out


def mean_auc_lists_over_rows(
    rows: list[Row],
    methods: list[str],
    axis: str,
    metrics: list[MetricSpec] | None = None,
    auc_cache: Optional[AUCCache] = None,
) -> dict[str, dict[str, list[float]]]:
    """Return ``{column_key: {method: [per_row_mean_auc, ...]}}`` -- the raw mean AUC per row.

    Same shared primitive :func:`rank_lists_over_rows` builds internally for
    each row (its ``mean_auc_by_method``/``mean_product_auc_by_method``
    dicts: for each row and each metric, average that method's per-seed AUC
    across its seeds present in the row -- see that function's docstring,
    steps 1-2), now exposed standalone for a caller that wants the raw
    per-row mean value itself rather than a rank (:func:`rank_lists_over_rows`
    /:func:`average_ranks_over_rows`) or a relative ratio
    (:func:`relative_auc_ratios_over_rows`). There is no ``rankdata``/
    ``_rank_and_record`` step here at all -- no ranking is involved, and
    unlike that function's ``_rank_and_record`` this never requires at least
    two methods present in a row (a row contributes its mean AUC for every
    method that has any data, even if it's the only method present there).

    ``column_key`` is either a ``MetricSpec.key`` from ``metrics`` or the
    synthetic key ``"product"`` (the raw point-wise product of those metrics,
    always higher-is-better -- see module docstring), exactly mirroring
    :func:`rank_lists_over_rows`. A method with no data at all across every
    row for a given column is simply absent from that column's inner dict.
    """
    if metrics is None:
        metrics = _ordered_metrics()
    method_set = set(methods)

    # per_row_means[column_key][method] -> list of per-row mean AUCs
    per_row_means: dict[str, dict[str, list[float]]] = {spec.key: {} for spec in metrics}
    per_row_means["product"] = {}

    for _row_label, row_runs, cache in rows:
        method_runs: dict[str, list[RunSeries]] = {}
        for run in row_runs:
            if run.method in method_set:
                method_runs.setdefault(run.method, []).append(run)

        for spec in metrics:
            for method in methods:
                aucs: list[float] = []
                for run in method_runs.get(method, []):
                    auc = _lookup_or_compute_auc(
                        auc_cache, run, axis, spec.key,
                        lambda run=run: (cache.get(run.run_name) or {}).get(spec.key),
                    )
                    if auc is not None:
                        aucs.append(auc)
                if aucs:
                    per_row_means[spec.key].setdefault(method, []).append(sum(aucs) / len(aucs))

        # Raw product of the metrics -- same per-row/per-method AUC pipeline,
        # always higher-is-better by construction.
        for method in methods:
            aucs = []
            for run in method_runs.get(method, []):
                auc = _lookup_or_compute_auc(
                    auc_cache, run, axis, "product",
                    lambda run=run: _compute_seed_product_curve(
                        run, axis, cache.get(run.run_name) or {}, metrics
                    ),
                )
                if auc is not None:
                    aucs.append(auc)
            if aucs:
                per_row_means["product"].setdefault(method, []).append(sum(aucs) / len(aucs))

    out: dict[str, dict[str, list[float]]] = {}
    for column_key, method_lists in per_row_means.items():
        if not method_lists:
            continue
        out[column_key] = {m: vs for m, vs in method_lists.items() if vs}
    return out


def average_ranks_over_rows(
    rows: list[Row],
    methods: list[str],
    axis: str,
    metrics: list[MetricSpec] | None = None,
    auc_cache: Optional[AUCCache] = None,
) -> dict[str, dict[str, float]]:
    """Return ``{column_key: {method: avg_rank}}`` (rank 1 = best; see module docstring).

    A thin sum/len reduction over :func:`rank_lists_over_rows` -- see that
    function's docstring for the full per-row ranking pipeline.
    """
    lists = rank_lists_over_rows(rows, methods, axis, metrics, auc_cache=auc_cache)
    return {
        column_key: {m: sum(vs) / len(vs) for m, vs in method_lists.items() if vs}
        for column_key, method_lists in lists.items()
    }


def relative_auc_seed_ratios_for_row(
    row: Row,
    methods: list[str],
    axis: str,
    metrics: list[MetricSpec] | None = None,
    auc_cache: Optional[AUCCache] = None,
    pool_methods: Optional[list[str]] = None,
) -> dict[str, dict[str, list[float]]]:
    """Return ``{column_key: {method: [seed_ratio, ...]}}`` for exactly one row.

    For each (method, seed) AUC in this single row, divide by the best AUC
    seen anywhere in this row (any method, any seed) -- *not* averaged across
    seeds, unlike :func:`relative_auc_ratio_lists_over_rows`. This is the
    shared per-row primitive: :func:`relative_auc_ratio_lists_over_rows`
    averages each row's seeds down to one value and collects those across
    rows (for the boxplot); a caller wanting one row's own full per-seed
    spread (e.g. the by-difficulty line chart's shaded band) can use this
    directly.

    ``pool_methods``, when given, decouples the ratio's *denominator* (the
    "best AUC seen anywhere") from its *numerator* (the methods actually
    reported in the output): the best is taken over ``pool_methods`` instead
    of ``methods`` -- e.g. every method in a broader comparison, when only a
    subset of those methods should appear in the returned dict. Defaults to
    ``methods`` (byte-identical to every pre-existing caller) when omitted.

    Same ``best == 0`` / "no data anywhere in this row for this column" skip
    behavior as :func:`relative_auc_ratios_over_rows` (see its docstring): a
    column with no ratio at all for this row is simply absent from the
    returned dict.
    """
    if metrics is None:
        metrics = _ordered_metrics()
    pool = pool_methods if pool_methods is not None else methods
    pool_set = set(pool)
    method_set = set(methods)
    needed_set = pool_set | method_set
    _row_label, row_runs, cache = row

    out: dict[str, dict[str, list[float]]] = {}

    def _ratios_for_column(
        column_key: str,
        aucs_by_pool_method: dict[str, list[float]],
        aucs_by_method: dict[str, list[float]],
        higher_is_better: bool,
    ) -> None:
        all_aucs = [a for aucs in aucs_by_pool_method.values() for a in aucs]
        if not all_aucs:
            return
        best = max(all_aucs) if higher_is_better else min(all_aucs)
        if best == 0:
            return  # undefined ratio -- skip this row/column rather than divide by zero
        out[column_key] = {
            m: [a / best for a in aucs] for m, aucs in aucs_by_method.items() if aucs
        }

    method_runs: dict[str, list[RunSeries]] = {}
    for run in row_runs:
        if run.method in needed_set:
            method_runs.setdefault(run.method, []).append(run)

    for spec in metrics:
        aucs_by_any_method: dict[str, list[float]] = {}
        for method in needed_set:
            aucs: list[float] = []
            for run in method_runs.get(method, []):
                auc = _lookup_or_compute_auc(
                    auc_cache, run, axis, spec.key,
                    lambda run=run: (cache.get(run.run_name) or {}).get(spec.key),
                )
                if auc is not None:
                    aucs.append(auc)
            if aucs:
                aucs_by_any_method[method] = aucs
        aucs_by_pool_method = {m: v for m, v in aucs_by_any_method.items() if m in pool_set}
        aucs_by_method = {m: v for m, v in aucs_by_any_method.items() if m in method_set}
        _ratios_for_column(spec.key, aucs_by_pool_method, aucs_by_method, spec.higher_is_better)

    # Raw product of the metrics -- same per-row/per-method AUC pipeline as
    # average_ranks_over_rows' product column, always higher-is-better.
    aucs_by_any_method = {}
    for method in needed_set:
        aucs = []
        for run in method_runs.get(method, []):
            auc = _lookup_or_compute_auc(
                auc_cache, run, axis, "product",
                lambda run=run: _compute_seed_product_curve(
                    run, axis, cache.get(run.run_name) or {}, metrics
                ),
            )
            if auc is not None:
                aucs.append(auc)
        if aucs:
            aucs_by_any_method[method] = aucs
    aucs_by_pool_method = {m: v for m, v in aucs_by_any_method.items() if m in pool_set}
    aucs_by_method = {m: v for m, v in aucs_by_any_method.items() if m in method_set}
    _ratios_for_column("product", aucs_by_pool_method, aucs_by_method, higher_is_better=True)

    return out


def relative_auc_ratio_lists_over_rows(
    rows: list[Row],
    methods: list[str],
    axis: str,
    metrics: list[MetricSpec] | None = None,
    auc_cache: Optional[AUCCache] = None,
    pool_methods: Optional[list[str]] = None,
) -> dict[str, dict[str, list[float]]]:
    """Return ``{column_key: {method: [per_row_ratio, ...]}}`` -- see below.

    Each row's seeds are averaged to one ratio (via
    :func:`relative_auc_seed_ratios_for_row`) and collected across rows
    without the final across-rows reduction -- the right input for a boxplot
    showing a method's ratio spread across rows.
    :func:`relative_auc_ratios_over_rows` is now a thin sum/len wrapper
    around this. See that function's docstring for the exact per-row
    definition of "ratio". ``pool_methods`` is forwarded verbatim to
    :func:`relative_auc_seed_ratios_for_row` -- see its docstring.
    """
    if metrics is None:
        metrics = _ordered_metrics()

    per_row_ratios: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        seed_ratios = relative_auc_seed_ratios_for_row(
            row, methods, axis, metrics, auc_cache=auc_cache, pool_methods=pool_methods,
        )
        for column_key, method_ratios in seed_ratios.items():
            for method, ratios in method_ratios.items():
                if not ratios:
                    continue
                ratio = sum(ratios) / len(ratios)
                per_row_ratios.setdefault(column_key, {}).setdefault(method, []).append(ratio)
    return per_row_ratios


def relative_auc_ratios_over_rows(
    rows: list[Row],
    methods: list[str],
    axis: str,
    metrics: list[MetricSpec] | None = None,
    auc_cache: Optional[AUCCache] = None,
    pool_methods: Optional[list[str]] = None,
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
    (division would be undefined) or where no method has any data. This is a
    thin sum/len reduction over :func:`relative_auc_ratio_lists_over_rows`.
    """
    lists = relative_auc_ratio_lists_over_rows(
        rows, methods, axis, metrics, auc_cache=auc_cache, pool_methods=pool_methods,
    )
    return {
        column_key: {m: sum(vs) / len(vs) for m, vs in method_lists.items() if vs}
        for column_key, method_lists in lists.items()
    }
