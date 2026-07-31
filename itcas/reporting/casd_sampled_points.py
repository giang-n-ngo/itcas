"""Sampled-points figures on the CASD problem (``casd_llm``) for the paper.

Unlike every other report in this package (whose figures aggregate metric
*curves* across seeds/methods), this module plots the raw points a single
representative run of each main algorithm actually **sampled** -- both in
objective space (Safety vs. Utility) and in context space (prompt_toxicity vs.
prompt_length) -- mirroring the per-run debug scatter
:func:`itcas.pipeline.loop._plot_run_scatter` produces during a live run, but
as a paper-grade multi-method comparison instead of a single-run PNG.

**Method roster.** For *each* of the sequential/batch settings, the SAME 5
"family" columns -- STR-TS-10, BES-TS-10, ECI, MOC-CAS, NDIG -- just using
that setting's own variant of each method, reusing the exact
``(family_key, sequential_method, batch_method, display_label)`` pairing
:data:`itcas.reporting.batch_improvement_comparison.FAMILIES` already defines
(imported from there, never re-typed, so this can't drift --
:mod:`itcas.reporting.method_group_comparison` is the precedent for exactly
this "batch methods among themselves / sequential methods among themselves"
split, see its module docstring):

    family        sequential                  batch                             label
    straddle      straddle_then_sample_lse10  straddle_then_sample_lse10_batch  STR-TS-10
    bes           bes_then_sample_lse10       bes_then_sample_lse10_batch       BES-TS-10
    eci           cas_eci                     cas_eci_batch                     ECI
    moc_cas_hard  moc_cas_hard                moc_cas_hard_batch                MOC-CAS
    itcas         itcas_seq_ndig              itcas_ndig                        NDIG

* :data:`SEQUENTIAL_METHODS` -- the five families' *sequential* variants.
* :data:`BATCH_METHODS` -- the five families' *batch* variants.

Both column tuples are built by indexing into ``FAMILIES`` in the explicit
column order above (``STR-TS-10, BES-TS-10, ECI, MOC-CAS, NDIG`` -- NDIG
*last*, unlike ``FAMILIES``' own itcas-first tuple order), not by hand-typing
method-name strings. ``random`` has no sequential/batch pair and is excluded
from both, same reasoning ``batch_improvement_comparison.py`` gives.

**Layout.** Per CASD difficulty level (``p1_00``..``p4_00``, hardest-to-
easiest, see ``configs/thresholds.json["casd_llm"]`` /
:func:`itcas.reporting.casd_comparison._load_casd_thresholds`), two PDFs are
written: one grid of :data:`SEQUENTIAL_METHODS` columns, one of
:data:`BATCH_METHODS` columns (4 levels x 2 groups = 8 files total). Each grid
is 2 rows x 5 columns:

* Row 1 (objective space): f1 (Safety, x) vs. f2 (Utility, y) for the run's
  full evaluated point set (:attr:`itcas.reporting.metrics.RunSeries.X_per_step`
  / ``Y_per_step``, index ``-1``). x-limits are ``(2*tau_safety - 1.0, 1.0)``
  for that level -- CASD's f1 values cluster near their natural ceiling of
  1.0, so this puts ``tau_safety`` exactly at the horizontal midpoint of
  every panel for every level (level 4's ``tau_safety=0.900`` gives
  ``(0.8, 1.0)``; level 1's tight ``tau_safety=0.998`` gives
  ``(0.996, 1.0)``) while keeping the upper bound at the metric's true max;
  points below the window (e.g. penalty-value fallback rows) simply clip
  off-screen. y-limits are shared across all columns *within one figure*,
  computed from the f2 values of in-window points across every column so the
  shared axis isn't dominated by off-screen-in-x outliers. Dashed lines mark
  that level's ``tau_safety`` (vertical, always at the panel's horizontal
  midpoint by construction) / ``tau_utility`` (horizontal), reused from
  ``casd_comparison._load_casd_thresholds`` rather than recomputed.
* Row 2 (context space): prompt_toxicity (x) vs. prompt_length (y). Limits are
  the *actual* ``casd_llm`` problem's bounds at its context dims
  (``REGISTRY["casd_llm"]().bounds[:, context_dims]``), not a hardcoded [0,1]
  literal, so this stays correct if bounds ever change; no threshold lines
  (thresholds are objective-space only).

Both rows plot only what the algorithm actually **acquired** -- the run's
init-dataset prefix (``X_per_step[0].shape[0]`` points) is dropped entirely
(not drawn, not in the legend), matching this figure's framing ("points
actually sampled by each algorithm"). Acquired points are drawn as an "x" in
the column's method color (``s=24, alpha=0.85``). Colors are reused from
:data:`itcas.reporting.method_group_comparison.METHOD_STYLES` (keyed
``"sequential"``/``"batch"``, one shared color per family so the same family
plots in the same color in both figures) rather than
``summary._SYNTHETIC_METHOD_STYLES`` (which only covers the sequential-
baseline + batch-NDIG mix used by the *different* main synthetic-comparison
table, not a real sequential/batch pair for every family here).

Row 1 additionally shades the **feasible corner** (``f1 >= tau_safety AND
f2 >= tau_utility``, see ``itcas/metrics/metrics.py::is_feasible``) as a
low-alpha gray rectangle from ``(tau_safety, tau_utility)`` to
``(obj_xlim[1], y_hi)``, under the scatter points and above the gridlines.

There is no per-panel title and no figure suptitle anywhere in this
module's output -- method identity lives entirely in one shared legend for
the whole figure (not per panel): one colored "x" entry per method/column,
labeled with that family's *bare* display label (``STR-TS-10``, ``BES-TS-10``,
``ECI``, ``MOC-CAS``, ``NDIG`` -- pulled straight from ``FAMILIES``' own
``label`` field via ``_FAMILY_BY_KEY``, never ``method_labels
.METHOD_ABBREVIATIONS``, so the batch figure's legend reads identically to
the sequential figure's -- no "-B" suffix anywhere here), plus one gray
"Feasible region" patch entry. Six legend entries total; no generic
"Acquired"/threshold-line entries (the dashed ``tau_safety``/``tau_utility``
lines are still drawn in every row-1 panel, just without their own legend
entry -- the feasible-region shading's edges already mark the same
boundary).

**Run selection.** One representative seed per (method, difficulty) -- default
``seed=0``, overridable. A missing (method, difficulty, seed) run on disk
skips just that column (an annotated empty panel), never the whole figure.

**Discovery.** Reuses
:func:`itcas.reporting.batch_vs_sequential._collect_family_runs` (no new
JSONL parsing) and :func:`itcas.reporting.summary._difficulty_of` to bucket
runs by CASD's own ``p1_00``..``p4_00`` difficulty scale, exactly as
:mod:`itcas.reporting.casd_comparison` already does.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from ..pipeline.problems import REGISTRY as PROBLEM_REGISTRY
from .batch_improvement_comparison import FAMILIES
from .batch_vs_sequential import _collect_family_runs
from .casd_comparison import _load_casd_thresholds
from .method_group_comparison import METHOD_STYLES as _FAMILY_METHOD_STYLES
from .metrics import RunSeries
from .summary import _difficulty_of

_PROBLEM = "casd_llm"
_DEFAULT_OUTPUT_DIR = "results/casd_sampled_points"
_SELECTED_LEVELS: tuple[int, ...] = (1, 2, 3, 4)

# {family_key: (sequential_method, batch_method, label)}, from
# batch_improvement_comparison.FAMILIES (never re-typed here, so this can't
# drift -- see module docstring). Column order is explicit here (NDIG last)
# and deliberately NOT `FAMILIES`' own itcas-first tuple order.
_FAMILY_BY_KEY: dict[str, tuple[str, str, str]] = {
    fam: (seq, batch, label) for fam, seq, batch, label in FAMILIES
}
_COLUMN_ORDER: tuple[str, ...] = ("straddle", "bes", "eci", "moc_cas_hard", "itcas")

SEQUENTIAL_METHODS: tuple[str, ...] = tuple(_FAMILY_BY_KEY[k][0] for k in _COLUMN_ORDER)
BATCH_METHODS: tuple[str, ...] = tuple(_FAMILY_BY_KEY[k][1] for k in _COLUMN_ORDER)
_ALL_METHODS: tuple[str, ...] = SEQUENTIAL_METHODS + BATCH_METHODS

# One color per family, shared across both figures (the sequential and batch
# variant of a family plot in the same color) -- reused verbatim from
# method_group_comparison.METHOD_STYLES, never invented fresh here. See
# module docstring for why this (not summary._SYNTHETIC_METHOD_STYLES) is the
# right source for this report.
_METHOD_STYLES_BY_GROUP: dict[str, dict[str, dict]] = _FAMILY_METHOD_STYLES

_TAU_SAFETY_COLOR = "black"
_TAU_UTILITY_COLOR = "dimgray"
_FEASIBLE_COLOR = "gray"
_FEASIBLE_ALPHA = 0.15


# ---------------------------------------------------------------------------
# Run selection
# ---------------------------------------------------------------------------
def _select_run(
    runs: list[RunSeries], diff_label: str, method: str, seed: int
) -> Optional[RunSeries]:
    for r in runs:
        if r.method == method and int(r.seed) == int(seed) and _difficulty_of(r) == diff_label:
            return r
    return None


def _run_points(run: RunSeries) -> dict:
    """Points actually *acquired* by the algorithm (post-init) from a
    ``RunSeries``' cumulative arrays.

    The init-dataset prefix (``X_per_step[0].shape[0]`` points) is sliced off
    here -- what's returned (``X``/``Y``) is only ``X_full[n_init:]`` /
    ``Y_full[n_init:]`` -- so nothing downstream ever sees or draws it (see
    module docstring). ``n_init``/``n_total`` are kept for bookkeeping only.
    """
    X_full = run.X_per_step[-1]
    Y_full = run.Y_per_step[-1]
    n_init = int(run.X_per_step[0].shape[0]) if run.X_per_step else 0
    n_total = int(X_full.shape[0])
    n_init = max(0, min(n_init, n_total))
    return {"X": X_full[n_init:], "Y": Y_full[n_init:], "n_init": n_init, "n_total": n_total}


# ---------------------------------------------------------------------------
# Shared y-limits for row 1 (objective space)
# ---------------------------------------------------------------------------
def _shared_objective_ylim(
    cols_points: list[Optional[dict]], obj_xlim: tuple[float, float]
) -> tuple[float, float]:
    """y-limits shared across a figure's columns, from f2 of in-x-window points.

    "In-window" means f1 (obj 0) in ``obj_xlim`` (that level's
    ``(2*tau_safety - 1.0, 1.0)`` window -- see module docstring). Falls back
    to every column's full f2 range if no point falls in-window, and to a
    fixed default if there is no data at all (e.g. every column missing its
    seed's run).
    """
    ys: list[float] = []
    for pts in cols_points:
        if pts is None:
            continue
        Y = pts["Y"]
        if Y.numel() == 0 or Y.shape[-1] < 2:
            continue
        f1 = Y[:, 0]
        f2 = Y[:, 1]
        mask = (f1 >= obj_xlim[0]) & (f1 <= obj_xlim[1])
        if bool(mask.any()):
            ys.extend(f2[mask].tolist())
    if not ys:
        for pts in cols_points:
            if pts is None:
                continue
            Y = pts["Y"]
            if Y.numel() == 0 or Y.shape[-1] < 2:
                continue
            ys.extend(Y[:, 1].tolist())
    if not ys:
        return (-1.0, 1.0)
    lo, hi = min(ys), max(ys)
    pad = 0.05 * max(1e-9, hi - lo)
    return (lo - pad, hi + pad)


# ---------------------------------------------------------------------------
# Per-column panel drawing
# ---------------------------------------------------------------------------
def _draw_objective_panel(
    ax, pts: Optional[dict], color: str, tau_safety: float, tau_utility: float,
    obj_xlim: tuple[float, float], obj_ylim: tuple[float, float], seed: int,
) -> None:
    ax.set_xlim(*obj_xlim)
    ax.set_ylim(*obj_ylim)
    ax.grid(True, alpha=0.25, zorder=0)

    # Feasible corner (f1 >= tau_safety AND f2 >= tau_utility -- see
    # itcas/metrics/metrics.py::is_feasible), shaded under the scatter points
    # but above the gridlines -- see module docstring.
    import matplotlib.patches as mpatches

    rect_x0 = max(tau_safety, obj_xlim[0])
    rect_y0 = max(tau_utility, obj_ylim[0])
    rect_w = obj_xlim[1] - rect_x0
    rect_h = obj_ylim[1] - rect_y0
    if rect_w > 0 and rect_h > 0:
        ax.add_patch(mpatches.Rectangle(
            (rect_x0, rect_y0), rect_w, rect_h,
            facecolor=_FEASIBLE_COLOR, alpha=_FEASIBLE_ALPHA, edgecolor="none", zorder=1,
        ))

    ax.axvline(tau_safety, color=_TAU_SAFETY_COLOR, linestyle="--", linewidth=1.5,
               alpha=0.9, zorder=10)
    ax.axhline(tau_utility, color=_TAU_UTILITY_COLOR, linestyle="--", linewidth=1.5,
               alpha=0.9, zorder=10)

    if pts is None:
        ax.text(0.5, 0.5, f"no seed {seed} run found", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="grey")
        return
    Y = pts["Y"]
    if Y.shape[0] > 0:
        ax.scatter(Y[:, 0].numpy(), Y[:, 1].numpy(), s=24, alpha=0.85,
                   color=color, marker="x", zorder=6)


def _draw_context_panel(
    ax, pts: Optional[dict], color: str, context_dims: tuple[int, ...],
    ctx_bounds: tuple[tuple[float, float], tuple[float, float]], seed: int,
) -> None:
    (x_lo, x_hi), (y_lo, y_hi) = ctx_bounds
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.grid(True, alpha=0.25)
    if pts is None:
        ax.text(0.5, 0.5, f"no seed {seed} run found", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="grey")
        return
    X = pts["X"]
    c0, c1 = context_dims[0], context_dims[1]
    if X.shape[0] > 0:
        ax.scatter(X[:, c0].numpy(), X[:, c1].numpy(), s=24, alpha=0.85,
                   color=color, marker="x", zorder=6)


# ---------------------------------------------------------------------------
# One (level, group) figure
# ---------------------------------------------------------------------------
def _plot_group_figure(
    level: int,
    diff_label: str,
    methods: tuple[str, ...],
    method_styles: dict[str, dict],
    runs: list[RunSeries],
    thresholds_by_level: dict[str, dict],
    context_dims: tuple[int, ...],
    ctx_bounds: tuple[tuple[float, float], tuple[float, float]],
    seed: int,
    out_path: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    cfg = thresholds_by_level[str(level)]
    tau_safety, tau_utility = float(cfg["thresholds"][0]), float(cfg["thresholds"][1])
    # tau_safety sits at the exact horizontal midpoint of every panel, for
    # every level -- see module docstring.
    obj_xlim = (2.0 * tau_safety - 1.0, 1.0)

    cols_run = [_select_run(runs, diff_label, m, seed) for m in methods]
    cols_points = [_run_points(r) if r is not None else None for r in cols_run]
    obj_ylim = _shared_objective_ylim(cols_points, obj_xlim)

    n_cols = len(methods)
    fig_w = max(3.6 * n_cols, 4.0)
    fig, axes = plt.subplots(2, n_cols, figsize=(fig_w, 7.2), squeeze=False)

    legend_handles = []
    for c_idx, (fam_key, method, pts) in enumerate(zip(_COLUMN_ORDER, methods, cols_points)):
        color = method_styles.get(method, {}).get("color", "tab:blue")
        legend_handles.append(mlines.Line2D(
            [], [], color=color, marker="x", linestyle="None", markersize=6,
            alpha=0.85, label=_FAMILY_BY_KEY[fam_key][2],
        ))

        ax0 = axes[0][c_idx]
        _draw_objective_panel(ax0, pts, color, tau_safety, tau_utility, obj_xlim, obj_ylim, seed)
        ax0.set_xlabel("Objective 0 (Safety)", fontsize=10.5)
        if c_idx == 0:
            ax0.set_ylabel("Objective 1 (Utility)", fontsize=10.5)
        ax0.tick_params(axis="both", labelsize=9.5)

        ax1 = axes[1][c_idx]
        _draw_context_panel(ax1, pts, color, context_dims, ctx_bounds, seed)
        ax1.set_xlabel("Context 0 (prompt_toxicity)", fontsize=10.5)
        if c_idx == 0:
            ax1.set_ylabel("Context 1 (prompt_length)", fontsize=10.5)
        ax1.tick_params(axis="both", labelsize=9.5)

    legend_handles.append(mpatches.Patch(
        facecolor=_FEASIBLE_COLOR, alpha=_FEASIBLE_ALPHA, label="Feasible region",
    ))
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles),
               fontsize=10.5, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
def summarize_casd_sampled_points(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    seed: int = 0,
) -> list[str]:
    """Write the 8 sampled-points PDFs (4 levels x {sequential, batch}).

    See module docstring for the full layout contract. Always attempts all 8
    figures regardless of what's discovered on disk -- a missing (method,
    difficulty, seed) run only skips that column (an annotated empty panel),
    never the whole figure.
    """
    input_path = Path(input_dir)
    out_dir = Path(output_dir) if output_dir is not None else Path(_DEFAULT_OUTPUT_DIR)

    runs_by_problem = _collect_family_runs(input_path, [_PROBLEM], list(_ALL_METHODS))
    runs = runs_by_problem.get(_PROBLEM, [])

    thresholds_by_level = _load_casd_thresholds()

    problem = PROBLEM_REGISTRY[_PROBLEM]()
    context_dims = tuple(int(d) for d in problem.context_dims)
    bounds = problem.bounds
    ctx_bounds = (
        (float(bounds[0, context_dims[0]]), float(bounds[1, context_dims[0]])),
        (float(bounds[0, context_dims[1]]), float(bounds[1, context_dims[1]])),
    )

    paths: list[str] = []
    for level in _SELECTED_LEVELS:
        diff_label = f"p{level}_00"
        for group_name, methods in (
            ("sequential", SEQUENTIAL_METHODS),
            ("batch", BATCH_METHODS),
        ):
            method_styles = _METHOD_STYLES_BY_GROUP[group_name]
            out_path = out_dir / f"casd_sampled_points_{diff_label}_{group_name}.pdf"
            ok = _plot_group_figure(
                level, diff_label, methods, method_styles, runs,
                thresholds_by_level, context_dims, ctx_bounds, seed, out_path,
            )
            paths.append(str(ok))

    return paths


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser("itcas.casd_sampled_points")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Representative seed to select per (method, difficulty) column.",
    )
    args = parser.parse_args(argv)

    paths = summarize_casd_sampled_points(
        args.input_dir, output_dir=args.output_dir, seed=args.seed,
    )
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
