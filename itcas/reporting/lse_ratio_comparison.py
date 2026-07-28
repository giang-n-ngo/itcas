"""Stage-1/LSE budget ratio comparison for the Family-C "LSE-then-sample" baselines.

Family-C "LSE-then-sample" ("Hybrid Cartographer") baselines
(``straddle_then_sample`` / ``bes_then_sample``, see
``itcas.pipeline.loop._parse_two_stage``/``TwoStageSpec``) run a Stage-1
level-set-estimation acquisition for the first ``stage1_fraction * n_iters``
iterations, then switch to Stage-2 penalized interior sampling for the rest.
``stage1_fraction`` is encoded as a mandatory ``_lseNN`` infix on the
``--method`` name (``NN`` = integer percent), e.g. ``straddle_then_sample_lse10``
= 10% Stage-1 / 90% Stage-2, with an optional trailing ``_batch`` for the
batch (QD-DPP) sibling. This module answers, for each of the ``sequential``
and ``batch`` settings, "how much does that split matter?" by plotting all
three ratios (10/25/50%, see ``configs/lse_ratio_comparison.json``) for both
bases together in one grid figure per (setting, difficulty).

Color/linestyle scheme (stable across every figure in this report, so "blue
always means 10%"):

* **Color = ratio.** Each Stage-1/LSE percentage gets one fixed tab10 hue,
  reused in every panel of every PDF this module writes: 10% and 25% and 50%
  are always plotted in the same three colors regardless of which figure
  you're looking at (see ``_KNOWN_RATIO_COLORS`` -- extended deterministically
  for any additional ratio a future config might add).
* **Linestyle = base.** ``straddle_then_sample`` is always solid,
  ``bes_then_sample`` always dashed (see ``_LINESTYLES`` -- extended
  deterministically for any additional base a future config might add).

This is the same "orthogonal color/linestyle" idea as
``itcas.reporting.batch_vs_sequential._group_method_styles`` (there:
color = family, linestyle = sequential-vs-batch variant), just with the two
axes swapped here: color = ratio, linestyle = base algorithm. With 3 ratios x
2 bases that's up to 6 lines per panel, and the color/linestyle split keeps
"same ratio" and "same base" both immediately visually identifiable.

**Axis per setting.** Exactly like
``itcas.reporting.school_comparison`` (see ``_AXIS_BY_SETTING`` there and its
comment): two-stage *sequential* mode burns a fixed Stage-1 budget of
individual evaluations that doesn't correspond to algorithmic steps the same
way batch mode does, so **total individual evaluations**
(``RunSeries.x_evals``) is the only fair shared x-axis across the three
ratios in sequential mode -- a lower LSE fraction reaches Stage-2 (and
therefore a comparable per-iteration acquisition) sooner in evaluation-count
terms, but "step" would conflate that with the (irrelevant, sequential-only)
notion of one evaluation per step. Batch mode instead advances one
algorithmic step per batch regardless of batch size, so **algorithmic step**
(``RunSeries.x_steps``) is the meaningful shared axis there.

Coverage: every difficulty level present on disk is plotted (unlike
``school_comparison``, which restricts to one "default" difficulty), using
the same rows-by-problem / rows-by-difficulty layout split as
``itcas.reporting.batch_vs_sequential`` -- one PDF per shared difficulty
level with rows = problems, plus one PDF per **real-world** problem (see
``batch_vs_sequential._REAL_WORLD_PROBLEMS``: ``spacecraft_formation_flying_a1``
and ``casd_llm``, each using its own per-problem difficulty scale instead of
the shared ``p0_01``/``p0_05``/``p0_10``/``p0_20`` scale) with rows = that
problem's own difficulty levels.

Each panel's columns are metric curves (see ``itcas.reporting.metrics``) plus
a raw point-wise product column and a product-rank column, exactly mirroring
``batch_vs_sequential.plot_group_grid`` / ``summary._plot_problem_curves``.

Like ``school_comparison`` (the closer analog here: also a cross-cutting axis
comparison rather than a proposed-vs-baselines comparison), this module ships
with **no** Friedman/Wilcoxon statistics section -- there is no natural
"proposed" method among three ratios, and the existing "Product"/"Product
rank" columns already do the descriptive "who's better" job.

The default config (``configs/lse_ratio_comparison.json``) only lists the two
bases that actually have run data on disk today (``straddle_then_sample``,
``bes_then_sample``); ``c2lse_then_sample`` has zero runs on disk and is
intentionally omitted so a first run of this report produces real figures
instead of a wall of "(no data)" panels. The method-list construction below
is generic over ``config["bases"]``, so adding ``c2lse_then_sample`` back in
once it has data is a one-line config change, no code change.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .batch_vs_sequential import (
    _REAL_WORLD_PROBLEMS,
    _Row,
    _collect_family_runs,
    _difficulties_present,
    _rows_by_difficulty,
    _rows_by_problem,
)
from .metrics import RunSeries
from .summary import (
    CurveCache,
    _SHORT_CURVE_LABELS,
    _min_med_max,
    _ordered_metrics,
    _per_method_product_curves,
    _precompute,
    _rank_curves_on_union_grid,
)

_DEFAULT_CONFIG = "configs/lse_ratio_comparison.json"
_DEFAULT_OUTPUT_DIR = "results/lse_ratio_comparison"

# Stable hue per ratio (tab10) -- reused verbatim from
# batch_vs_sequential._TAB10 so both reports draw from the same fixed palette
# (not that any single figure ever needs both reports' color keys to agree,
# but there's no reason to diverge).
_TAB10: tuple[str, ...] = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)

# Fixed color assignment for the three ratios this report is built around
# (see module docstring: "blue always means 10%"). Any *additional* ratio a
# future config adds falls back to the next unused tab10 hue, assigned
# deterministically in sorted-ratio order (see _ratio_color_map) rather than
# by config-list order, so re-ordering "lse_proportions" in the config can
# never silently reshuffle an existing ratio's color.
_KNOWN_RATIO_COLORS: dict[int, str] = {10: _TAB10[0], 25: _TAB10[1], 50: _TAB10[2]}

# Fixed linestyle assignment for the two known bases (see module docstring:
# straddle_then_sample solid, bes_then_sample dashed). Any additional base a
# future config adds falls back to the next unused linestyle, in the order
# listed under config["bases"].
_LINESTYLES: tuple[str, ...] = ("-", "--", "-.", ":")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def _load_json(path: str | Path) -> dict:
    with Path(path).open() as f:
        return json.load(f)


def load_config(config_path: str | Path = _DEFAULT_CONFIG) -> dict:
    """Return the raw ``lse_proportions``/``bases``/``problems`` config dict."""
    return _load_json(config_path)


def load_problems(config_path: str | Path = _DEFAULT_CONFIG) -> list[str]:
    return load_config(config_path)["problems"]


# ---------------------------------------------------------------------------
# Method list + style construction
# ---------------------------------------------------------------------------
def _ratio_color_map(proportions: list[int]) -> dict[int, str]:
    """Fixed color per ratio (see module docstring); extended deterministically."""
    colors: dict[int, str] = {}
    extra_idx = len(_KNOWN_RATIO_COLORS)
    for p in sorted(proportions):
        if p in _KNOWN_RATIO_COLORS:
            colors[p] = _KNOWN_RATIO_COLORS[p]
        else:
            colors[p] = _TAB10[extra_idx % len(_TAB10)]
            extra_idx += 1
    return colors


def _base_linestyle_map(bases: list[str]) -> dict[str, str]:
    """Fixed linestyle per base, in ``bases`` list order (see module docstring)."""
    return {b: _LINESTYLES[i % len(_LINESTYLES)] for i, b in enumerate(bases)}


def ratio_base_methods_and_styles(
    proportions: list[int],
    bases: list[str],
    setting: str,
) -> tuple[list[str], dict[str, dict]]:
    """Return ``(methods, styles)`` for one setting (mirrors ``_group_method_styles``).

    ``methods`` lists every ``(ratio, base)`` combination's concrete method
    name, e.g. ``straddle_then_sample_lse10`` (sequential) or
    ``straddle_then_sample_lse10_batch`` (batch) -- ``_batch`` is appended
    whenever ``setting == "batch"``, exactly like
    ``school_comparison._methods_for_proportion``, since these Family-C bases
    have genuine batch-mode run directories on disk distinct from their
    sequential counterparts. ``styles[method]`` is
    ``{"color": ..., "linestyle": ...}`` with color keyed by ratio and
    linestyle keyed by base (see module docstring).
    """
    suffix = "_batch" if setting == "batch" else ""
    ratio_colors = _ratio_color_map(proportions)
    base_linestyles = _base_linestyle_map(bases)
    methods: list[str] = []
    styles: dict[str, dict] = {}
    for proportion in proportions:
        for base in bases:
            method = f"{base}_lse{proportion}{suffix}"
            styles[method] = {
                "color": ratio_colors[proportion],
                "linestyle": base_linestyles[base],
            }
            methods.append(method)
    return methods, styles


# ---------------------------------------------------------------------------
# Discovery -- reuses batch_vs_sequential's generic (family-agnostic) helpers
# ---------------------------------------------------------------------------
# _collect_family_runs, _difficulties_present, _rows_by_problem and
# _rows_by_difficulty (imported above) take a plain `methods: list[str]` and
# carry no family-specific coupling, so they're reused as-is here rather than
# duplicated -- same "every difficulty, not just one default" coverage as
# batch_vs_sequential (unlike school_comparison's single-difficulty version).


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
_AXIS_LABEL = {"evals": "Total individual evaluations", "steps": "Algorithmic step"}


def plot_ratio_grid(
    methods: list[str],
    method_styles: dict[str, dict],
    rows: list[_Row],
    axis: str,
    out_path: str | Path,
) -> Optional[Path]:
    """Render one grid figure: rows = ``rows``, cols = metrics + product + rank.

    Adapted from ``batch_vs_sequential.plot_group_grid``, parametrized by
    ``axis`` (``"evals"`` or ``"steps"``, see module docstring for why this
    report can't hardcode one axis the way that module does) instead of a
    fixed module-level ``_AXIS`` constant.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if not rows:
        return None

    axis_label = _AXIS_LABEL[axis]

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

    n_rows = len(rows)
    n_cols = len(metrics_present) + 2  # + raw product + product rank

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
                    xs = run.x_evals if axis == "evals" else run.x_steps
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
                    linewidth=1.5, label=method,
                )
                if len(curves) > 1:
                    ax.fill_between(x_plot, lo[:n], hi[:n], color=style["color"], alpha=0.15)
                plotted = True
            if not plotted:
                ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12, color="grey")
            if r_idx == 0:
                ax.set_title(_SHORT_CURVE_LABELS.get(spec.key, spec.label), fontsize=12)
            if r_idx == n_rows - 1:
                ax.set_xlabel(axis_label, fontsize=12)
            if c_idx == 0:
                ax.set_ylabel(row_label, fontsize=12)

        method_x, method_med, method_lo, method_hi = _per_method_product_curves(
            method_runs, methods, axis, cache, metrics_present,
        )

        # Raw product column
        ax = axes[r_idx][-2]
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
                linewidth=1.5, label=method,
            )
            if len(method_runs.get(method, [])) > 1:
                ax.fill_between(x_plot, method_lo[method], method_hi[method],
                                color=style["color"], alpha=0.15)
            plotted = True
        if not plotted:
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="grey")
        if r_idx == 0:
            ax.set_title("Product\n(raw, higher is better)", fontsize=12)
        if r_idx == n_rows - 1:
            ax.set_xlabel(axis_label, fontsize=12)

        # Rank column
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
                        linestyle=style["linestyle"], linewidth=1.5, label=method)
                plotted = True
            n_ranked = len(method_med)
            ax.set_ylim(n_ranked + 0.5, 0.5)
            ax.set_yticks(list(range(1, n_ranked + 1)))
        if not plotted:
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="grey")
        if r_idx == 0:
            ax.set_title("Product rank\n(1 = best)", fontsize=12)
        if r_idx == n_rows - 1:
            ax.set_xlabel(axis_label, fontsize=12)

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
            fontsize=12, bbox_to_anchor=(0.5, -0.1),
        )
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    else:
        fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
# Sequential only, by request -- batch is intentionally not generated right
# now (drop this comment and re-add a "batch": ("steps", "steps") entry if
# it's needed again). Two-stage sequential mode burns a fixed Stage-1
# evaluation budget that doesn't line up step-for-step across ratios, so the
# evaluations axis is what's comparable here. See module docstring /
# school_comparison._AXIS_BY_SETTING for the full reasoning.
_AXIS_BY_SETTING: dict[str, tuple[str, str]] = {
    "sequential": ("evals", "evaluations"),
}


def summarize_lse_ratio_comparison(
    input_dir: str | Path,
    config_path: str | Path = _DEFAULT_CONFIG,
    output_dir: str | Path | None = None,
) -> list[str]:
    """Produce every (setting, difficulty) ratio-comparison grid PDF.

    One PDF per shared difficulty level (rows = problems) plus one PDF per
    real-world problem (rows = its own difficulty levels), per setting --
    see module docstring.
    """
    input_path = Path(input_dir)
    cfg = load_config(config_path)
    proportions = [int(p) for p in cfg["lse_proportions"]]
    bases = list(cfg["bases"])
    problems = list(cfg["problems"])
    out_dir = Path(output_dir) if output_dir is not None else Path(_DEFAULT_OUTPUT_DIR)

    paths: list[str] = []
    for setting, (axis, suffix) in _AXIS_BY_SETTING.items():
        methods, styles = ratio_base_methods_and_styles(proportions, bases, setting)
        runs_by_problem = _collect_family_runs(input_path, problems, methods)
        caches_by_problem: dict[str, CurveCache] = {
            p: _precompute(runs) for p, runs in runs_by_problem.items() if runs
        }

        # Standard layout: rows = problems, one PDF per shared difficulty level.
        standard_problems = [p for p in problems if p not in _REAL_WORLD_PROBLEMS]
        standard_runs = {p: runs_by_problem.get(p, []) for p in standard_problems}
        for diff in _difficulties_present(standard_runs):
            rows = _rows_by_problem(standard_problems, standard_runs, caches_by_problem, diff)
            if not rows:
                continue
            out_path = out_dir / f"ratio_comparison_{setting}_{diff}_vs_{suffix}.pdf"
            ok = plot_ratio_grid(methods, styles, rows, axis, out_path)
            if ok is not None:
                paths.append(str(ok))

        # Real-world layout: one PDF per real-world problem, rows = its own
        # difficulty levels (see batch_vs_sequential._REAL_WORLD_PROBLEMS).
        for rw_problem in _REAL_WORLD_PROBLEMS:
            if rw_problem not in problems:
                continue
            rw_runs = runs_by_problem.get(rw_problem, [])
            rw_cache = caches_by_problem.get(rw_problem, {})
            rows = _rows_by_difficulty(rw_runs, rw_cache)
            if rows:
                out_path = out_dir / f"ratio_comparison_{setting}_{rw_problem}_vs_{suffix}.pdf"
                ok = plot_ratio_grid(methods, styles, rows, axis, out_path)
                if ok is not None:
                    paths.append(str(ok))

    return paths


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser("itcas.lse_ratio_comparison")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument(
        "--config", type=str, default=_DEFAULT_CONFIG,
        help="Path to the LSE-ratio-comparison config (lse_proportions, bases, problems).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=_DEFAULT_OUTPUT_DIR,
        help=f"Directory to write grid PDFs to (default: {_DEFAULT_OUTPUT_DIR}).",
    )
    args = parser.parse_args(argv)

    paths = summarize_lse_ratio_comparison(
        args.input_dir, config_path=args.config, output_dir=args.output_dir
    )
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
