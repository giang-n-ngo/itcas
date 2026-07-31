"""Two-schools-of-thought comparison: LSE-then-sample vs CAS methods.

Reads a config (default ``configs/two_schools_of_thought.json``) that lists,
for each of the ``sequential`` and ``batch`` settings, an ``lse`` method group
(base method names, e.g. ``straddle_then_sample``, without an ``_lseNN``
infix and without a ``_batch`` suffix -- both are appended programmatically,
see below) and a ``cas`` method group, plus the set of problems to compare,
the "default" difficulty to restrict to, and ``lse_proportions`` (the
Stage-1/LSE budget percentages, e.g. ``[10, 25, 50]``) to sweep over.

For each ``(setting, proportion)`` pair this produces one grid figure, where
the concrete LSE-then-sample method list is built by appending ``_lse{NN}``
to each base name in ``lse``, and additionally appending ``_batch`` when
``setting == "batch"`` (since ``straddle_then_sample``/``bes_then_sample``
have genuine batch-mode siblings on disk, e.g.
``straddle_then_sample_lse50_batch``); the ``cas`` group is used as-is,
since its batch/sequential variants are already spelled out explicitly in
the config (e.g. ``cas_eci`` vs ``cas_eci_batch``):

    rows    = problem (only ``default_difficulty``)
    columns = metric curves (see ``itcas.reporting.metrics.REGISTRY``),
              + raw product column, + product-rank column

mirroring the columns of :func:`itcas.reporting.summary._plot_problem_curves`
("the main summary"), but with the grouping axis transposed from
difficulty (one problem, many difficulties) to problem (one difficulty, many
problems) so LSE-then-sample and CAS methods can be compared side by side
across the whole problem suite.

The raw-product and rank columns reuse
:func:`itcas.reporting.summary._compute_seed_product_curve`, which already
flips lower-is-better metrics (currently only FCFD) via reciprocal before
multiplying, so a method that improves on every metric always has a larger
product and a better (lower) rank.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Optional

from .metrics import RunSeries
from .summary import (
    CurveCache,
    _SHORT_CURVE_LABELS,
    _difficulty_of,
    _ordered_metrics,
    _per_method_product_curves,
    _precompute,
    _rank_curves_on_union_grid,
)
from .visualize import _discover_runs

# Sequential-only for now (see _AXIS_BY_SETTING below) -- PDFs are written
# here rather than into --input-dir, mirroring batch_vs_sequential.py's and
# lse_ratio_comparison.py's own dedicated results/ subfolders.
_DEFAULT_OUTPUT_DIR = "results/school_comparison"


def _difficulty_label(threshold_pct: float) -> str:
    """Format a threshold fraction exactly like ``summary._difficulty_of``.

    Must stay byte-for-byte identical to that function's formatting so a
    difficulty read from the config lines up with the label derived from each
    run's ``cfg.extra.threshold_pct``.
    """
    v = float(threshold_pct)
    return f"p{int(v):d}_{int(round((v - int(v)) * 100)):02d}"


def _load_config(path: str | Path) -> dict:
    with Path(path).open() as f:
        return json.load(f)


def _collect_setting_runs(
    input_dir: Path,
    problems: list[str],
    methods: list[str],
    diff_label: str,
) -> dict[str, list[RunSeries]]:
    """Runs grouped by problem, filtered to ``methods`` at ``diff_label``.

    ``_discover_runs`` globs and parses every ``*.jsonl`` under the path it is
    given, applying its ``benchmark`` filter only *after* parsing. Under the
    standard ``<root>/<problem>/<difficulty>/<method>/<run>`` sweep layout,
    scoping each call to ``input_dir / problem`` (rather than calling it once
    on the shared root, benchmark-filtered) keeps each parse pass limited to
    that problem's own files instead of the whole multi-problem sweep tree.
    Falls back to the shared root (benchmark-filtered) for problems without
    their own subdirectory, e.g. a flat run layout.
    """
    out: dict[str, list[RunSeries]] = {p: [] for p in problems}
    for problem in problems:
        problem_dir = input_dir / problem
        runs = _discover_runs(problem_dir if problem_dir.is_dir() else input_dir, benchmark=problem)
        out[problem] = [
            r for r in runs if r.method in methods and _difficulty_of(r) == diff_label
        ]
    return out


def _min_med_max(
    curves: list[list[float]],
) -> tuple[list[float], list[float], list[float]]:
    n = min(len(c) for c in curves)
    trimmed = [c[:n] for c in curves]
    lo, med, hi = [], [], []
    for col in zip(*trimmed):
        finite = [v for v in col if v == v]
        if not finite:
            lo.append(float("nan"))
            med.append(float("nan"))
            hi.append(float("nan"))
        else:
            lo.append(min(finite))
            med.append(median(finite))
            hi.append(max(finite))
    return lo, med, hi


# ---------------------------------------------------------------------------
# Style scheme: color = school (LSE-then-sample vs CAS), marker = method
# within a school (see module docstring's "orthogonal color/style axis" note
# and the analogous schemes in ``lse_ratio_comparison`` (color=ratio,
# linestyle=base) / ``batch_vs_sequential`` (color=family, linestyle=variant)
# -- here the two axes are color=school and marker=method-within-school,
# since curves are still continuous lines (not scatter), so ``markevery`` is
# used below to keep markers legible instead of stamping one per data point.
#
# LSE-then-sample bases get cool hues, CAS methods get warm hues (tab10), so
# "cool = staged LSE-then-Stage-2, warm = pure CAS acquisition" reads
# consistently across every panel of every figure this module writes. Each
# base additionally gets one fixed marker, independent of color, so two
# methods sharing a school (e.g. straddle_then_sample vs bes_then_sample, or
# cas_eci vs moc_cas_hard) stay visually distinct even where their curves
# overlap.
#
# Both tables are keyed by *base* method name: for LSE-then-sample that is
# the raw ``methods_cfg["lse"]`` entry (before the ``_lseNN``/``_batch``
# suffixes ``_methods_for_proportion``/``_method_style_map`` append); for CAS
# it is the entry with any trailing ``_batch`` stripped (so ``cas_eci`` and
# ``cas_eci_batch`` share one color/marker). Any base not covered by these
# fixed tables falls back to the next unused color/marker in its school's
# pool, assigned deterministically in the order that base is encountered
# while building the style map (see ``_method_style_map``) -- so an unknown
# base's color/marker never depends on dict/config iteration order beyond
# that, and known bases (straddle/bes/eci/moc_cas_hard) never shift.
_LSE_SCHOOL_COLORS: dict[str, str] = {
    "straddle_then_sample": "#1f77b4",  # tab10 blue
    "bes_then_sample": "#17becf",  # tab10 cyan
}
_CAS_SCHOOL_COLORS: dict[str, str] = {
    "cas_eci": "#ff7f0e",  # tab10 orange
    "moc_cas_hard": "#d62728",  # tab10 red
}
# Fallback pools for bases not in the fixed tables above (kept within each
# school's warm/cool family so an unrecognized base still visually reads as
# "LSE-then-sample" or "CAS").
_LSE_COLOR_FALLBACK: tuple[str, ...] = ("#1f77b4", "#17becf", "#9467bd", "#7f7f7f")
_CAS_COLOR_FALLBACK: tuple[str, ...] = ("#ff7f0e", "#d62728", "#bcbd22", "#8c564b")

_KNOWN_MARKERS: dict[str, str] = {
    "straddle_then_sample": "o",
    "bes_then_sample": "s",
    "cas_eci": "^",
    "moc_cas_hard": "D",
}
_MARKER_FALLBACK: tuple[str, ...] = ("v", "P", "X", "*", "h", "p")


def _strip_batch_suffix(name: str) -> str:
    """Return ``name`` with a trailing ``"_batch"`` removed, if present."""
    return name[: -len("_batch")] if name.endswith("_batch") else name


def _method_style_map(methods_cfg: dict, proportion: int, setting: str) -> dict[str, dict]:
    """Return ``{concrete_method_name: {"color": ..., "marker": ...}}``.

    Mirrors :func:`_methods_for_proportion`'s construction of concrete method
    names (LSE bases get ``_lseNN`` + optional ``_batch`` appended; CAS
    methods are used as-is), but additionally resolves each concrete method's
    style from its *base* identifier via the color/marker tables above (see
    their comment for the "school = color, method = marker" scheme and the
    deterministic fallback for unknown bases).
    """
    suffix = "_batch" if setting == "batch" else ""
    styles: dict[str, dict] = {}

    lse_fallback_idx = 0
    for base in methods_cfg["lse"]:
        if base in _LSE_SCHOOL_COLORS:
            color = _LSE_SCHOOL_COLORS[base]
        else:
            color = _LSE_COLOR_FALLBACK[lse_fallback_idx % len(_LSE_COLOR_FALLBACK)]
            lse_fallback_idx += 1
        method = f"{base}_lse{proportion}{suffix}"
        styles[method] = {"color": color, "marker": _KNOWN_MARKERS.get(base)}

    cas_fallback_idx = 0
    for method in methods_cfg["cas"]:
        base = _strip_batch_suffix(method)
        if base in _CAS_SCHOOL_COLORS:
            color = _CAS_SCHOOL_COLORS[base]
        else:
            color = _CAS_COLOR_FALLBACK[cas_fallback_idx % len(_CAS_COLOR_FALLBACK)]
            cas_fallback_idx += 1
        styles[method] = {"color": color, "marker": _KNOWN_MARKERS.get(base)}

    # Resolve markers for any base absent from _KNOWN_MARKERS, deterministically,
    # in the order each such method was first inserted above (dicts preserve
    # insertion order).
    marker_fallback_idx = 0
    for style in styles.values():
        if style["marker"] is None:
            style["marker"] = _MARKER_FALLBACK[marker_fallback_idx % len(_MARKER_FALLBACK)]
            marker_fallback_idx += 1
    return styles


def _methods_for_proportion(methods_cfg: dict, proportion: int, setting: str) -> list[str]:
    """Build the concrete method list for one LSE proportion and setting.

    ``methods_cfg["lse"]`` holds *base* method names (no ``_lseNN`` infix, no
    ``_batch`` suffix); the per-figure proportion is appended here as
    ``_lse{NN}``, and ``_batch`` is further appended when
    ``setting == "batch"`` since ``straddle_then_sample``/``bes_then_sample``
    have real batch-mode run directories on disk
    (``{base}_lse{NN}_batch``) distinct from their sequential counterparts.
    ``methods_cfg["cas"]`` is used as-is since its batch/sequential variants
    are already spelled out explicitly per setting in the config (this group
    also holds any other already-final method name needing no suffix, e.g.
    ``cr_ndig`` (sequential-only) and ``itcas_ndig`` (batch-only, the parsed
    label for ``method="itcas", quality="ndig"`` -- see
    ``visualize._load_run``).
    """
    suffix = "_batch" if setting == "batch" else ""
    lse_methods = [f"{m}_lse{proportion}{suffix}" for m in methods_cfg["lse"]]
    return lse_methods + list(methods_cfg["cas"])


SettingCollection = tuple[
    list[str], str, dict, list[int], dict[str, list[RunSeries]], dict[str, CurveCache]
]


def _collect_setting(
    input_dir: str | Path,
    config_path: str | Path,
    setting: str,
) -> SettingCollection:
    """Discover + precompute once per setting, across every proportion's methods.

    ``_discover_runs`` parses every ``*.jsonl`` under each problem directory
    regardless of which methods are ultimately wanted, so calling it once per
    proportion (3 proportions -> 3x the parsing) would redundantly re-parse
    the same files. Instead this discovers the *union* of methods needed
    across every proportion in ``lse_proportions`` (plus the shared ``cas``
    methods) in one pass; :func:`plot_school_comparison` then cheaply
    projects the result down to one proportion's methods in memory (no I/O,
    no metric recomputation) via :func:`_filter_precomputed_to_methods`.

    Returns ``(problems, diff_label, methods_cfg, proportions, runs_by_problem, caches)``.
    """
    cfg = _load_config(config_path)
    problems: list[str] = cfg["problems"]
    diff_label = _difficulty_label(cfg["default_difficulty"])
    methods_cfg = cfg[f"{setting}_methods"]
    proportions: list[int] = [int(p) for p in cfg["lse_proportions"]]

    all_methods: set[str] = set(methods_cfg["cas"])
    for proportion in proportions:
        all_methods.update(_methods_for_proportion(methods_cfg, proportion, setting))

    input_path = Path(input_dir)
    runs_by_problem = _collect_setting_runs(input_path, problems, sorted(all_methods), diff_label)
    caches: dict[str, CurveCache] = {
        p: _precompute(runs) for p, runs in runs_by_problem.items() if runs
    }
    return problems, diff_label, methods_cfg, proportions, runs_by_problem, caches


def _filter_precomputed_to_methods(
    problems: list[str],
    runs_by_problem: dict[str, list[RunSeries]],
    caches: dict[str, CurveCache],
    methods: list[str],
) -> tuple[dict[str, list[RunSeries]], dict[str, CurveCache]]:
    """Project a shared (all-proportions) discovery down to one proportion's methods.

    Pure in-memory filtering over already-parsed ``RunSeries``/already-computed
    metric curves -- no I/O, no recomputation -- so this is cheap to call once
    per proportion even though the underlying discovery+precompute in
    :func:`_collect_setting` ran only once for the whole setting.
    """
    method_set = set(methods)
    filtered_runs: dict[str, list[RunSeries]] = {}
    filtered_caches: dict[str, CurveCache] = {}
    for p in problems:
        runs = [r for r in runs_by_problem.get(p, []) if r.method in method_set]
        filtered_runs[p] = runs
        if runs:
            full_cache = caches.get(p, {})
            filtered_caches[p] = {
                r.run_name: full_cache[r.run_name] for r in runs if r.run_name in full_cache
            }
    return filtered_runs, filtered_caches


def plot_school_comparison(
    input_dir: str | Path,
    config_path: str | Path,
    setting: str,
    axis: str,
    out_path: str | Path,
    *,
    proportion: int,
    precomputed: Optional[SettingCollection] = None,
) -> Optional[Path]:
    """Render the LSE-vs-CAS grid figure for one ``setting`` ("sequential"/"batch").

    ``axis`` is ``"evals"`` (total individual evaluations) or ``"steps"``
    (algorithmic step). ``proportion`` is the Stage-1/LSE budget percentage
    (e.g. ``10``, ``25``, ``50``) used to select the ``_lseNN``-suffixed
    LSE-then-sample methods and to label the figure. Returns ``None`` (and
    writes nothing) if no run under ``input_dir`` matches any listed
    method/problem/difficulty combination.

    Pass ``precomputed`` (the tuple returned by :func:`_collect_setting`,
    shared across every proportion for this setting) to skip re-discovering
    and re-computing every run's metric curves when rendering multiple
    proportions/axis variants for the same setting; this function projects
    it down to ``proportion``'s methods in memory via
    :func:`_filter_precomputed_to_methods`.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if precomputed is None:
        precomputed = _collect_setting(input_dir, config_path, setting)
    problems, _diff_label, methods_cfg, _proportions, all_runs_by_problem, all_caches = precomputed
    methods: list[str] = _methods_for_proportion(methods_cfg, proportion, setting)
    runs_by_problem, caches = _filter_precomputed_to_methods(
        problems, all_runs_by_problem, all_caches, methods
    )
    if not caches:
        return None

    metrics = _ordered_metrics()
    seen_keys: set[str] = set()
    for cache in caches.values():
        for run_curves in cache.values():
            for key, y in run_curves.items():
                if y is not None:
                    seen_keys.add(key)
    metrics_present = [s for s in metrics if s.key in seen_keys]
    if not metrics_present:
        return None

    problems_present = [p for p in problems if runs_by_problem.get(p)]
    if not problems_present:
        return None

    n_rows = len(problems_present)
    n_cols = len(metrics_present) + 2  # + raw product + product rank
    method_styles = _method_style_map(methods_cfg, proportion, setting)

    fig_w = max(4.0 * n_cols, 12.0)
    fig_h = max(2.5 * n_rows + 1.0, 5.0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    axis_label = "Total individual evaluations" if axis == "evals" else "Algorithmic step"

    for r_idx, problem in enumerate(problems_present):
        cache = caches[problem]
        method_runs: dict[str, list[RunSeries]] = {}
        for run in runs_by_problem[problem]:
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
                color = style["color"]
                ax.plot(
                    x_plot, med[:n], color=color, marker=style["marker"],
                    markevery=max(1, n // 8), linewidth=1.5, label=method,
                )
                if len(curves) > 1:
                    ax.fill_between(x_plot, lo[:n], hi[:n], color=color, alpha=0.15)
                plotted = True
            if not plotted:
                ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12, color="grey")
            if r_idx == 0:
                ax.set_title(_SHORT_CURVE_LABELS.get(spec.key, spec.label), fontsize=12)
            if r_idx == n_rows - 1:
                ax.set_xlabel(axis_label, fontsize=12)
            if c_idx == 0:
                ax.set_ylabel(problem, fontsize=12)

        # Pre-compute each method's own product curve (own x-values -- see
        # _per_method_product_curves; no cross-method truncation/alignment).
        method_x, method_med, method_lo, method_hi = _per_method_product_curves(
            method_runs, methods, axis, cache, metrics_present,
        )

        # Raw product column — each method plotted against its OWN x-values,
        # exactly like the metric columns above; no cross-method alignment
        # needed since it's N independent lines.
        ax = axes[r_idx][-2]
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=10.5)
        plotted = False
        for method in sorted(method_med.keys()):
            style = method_styles[method]
            color = style["color"]
            x_plot = method_x[method]
            n = len(method_med[method])
            ax.plot(
                x_plot, method_med[method], color=color, marker=style["marker"],
                markevery=max(1, n // 8), linewidth=1.5, label=method,
            )
            if len(method_runs.get(method, [])) > 1:
                ax.fill_between(x_plot, method_lo[method], method_hi[method],
                                color=color, alpha=0.15)
            plotted = True
        if not plotted:
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="grey")
        if r_idx == 0:
            ax.set_title("Product\n(raw, higher is better)", fontsize=12)
        if r_idx == n_rows - 1:
            ax.set_xlabel(axis_label, fontsize=12)

        # Rank column (1 = best) — aligned across methods on the union-of-
        # x-values grid via forward-fill (see _rank_curves_on_union_grid).
        ax = axes[r_idx][-1]
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="both", labelsize=10.5)
        plotted = False
        if method_med:
            grid, rank_curves = _rank_curves_on_union_grid(method_x, method_med)
            n = len(grid)
            for method in sorted(method_med.keys()):
                style = method_styles[method]
                ax.plot(
                    grid, rank_curves[method], color=style["color"], marker=style["marker"],
                    markevery=max(1, n // 8), linewidth=1.5, label=method,
                )
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

    # Shared legend drawn once below all subplots
    handles_seen: dict[str, object] = {}
    for row in axes:
        for a in row:
            for h, lbl in zip(*a.get_legend_handles_labels()):
                handles_seen.setdefault(lbl, h)
    if handles_seen:
        fig.legend(
            list(handles_seen.values()), list(handles_seen.keys()),
            loc="lower center", ncol=min(len(methods), 6),
            fontsize=12, bbox_to_anchor=(0.5, -0.02),
        )
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    else:
        fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


_AXIS_BY_SETTING: dict[str, tuple[str, str]] = {
    # Sequential only, by request -- batch is intentionally not generated
    # right now (drop this comment and re-add a "batch": ("steps", "vs_steps")
    # entry if it's needed again; sequential runs vary in step count per
    # method since LSE-then-sample burns a fixed Stage-1 budget CAS doesn't,
    # so the evaluations axis is what lines methods up on a comparable
    # x-value here).
    "sequential": ("evals", "vs_evaluations"),
}


def summarize_school_comparison(
    input_dir: str | Path,
    config_path: str | Path = "configs/two_schools_of_thought.json",
    output_dir: str | Path | None = None,
) -> list[str]:
    """Produce the sequential (vs evaluations) figures (see ``_AXIS_BY_SETTING``).

    One figure is generated per Stage-1/LSE proportion listed in the config's
    ``lse_proportions`` (e.g. ``[10, 25, 50]``), so the total PDF count is
    ``len(settings) * len(lse_proportions)``.
    """
    input_path = Path(input_dir)
    out_dir = Path(output_dir) if output_dir is not None else Path(_DEFAULT_OUTPUT_DIR)

    paths: list[str] = []
    for setting, (axis, suffix) in _AXIS_BY_SETTING.items():
        # Discovery + metric precompute happens ONCE per setting, across the
        # union of every proportion's methods (see _collect_setting) -- each
        # proportion's figure below just re-filters that shared result in
        # memory, rather than re-parsing the run logs per proportion.
        precomputed = _collect_setting(input_path, config_path, setting)
        proportions = precomputed[3]
        for proportion in proportions:
            out_path = out_dir / f"two_schools_{setting}_lse{proportion}_{suffix}.pdf"
            ok = plot_school_comparison(
                input_path, config_path, setting, axis, out_path,
                proportion=proportion, precomputed=precomputed,
            )
            if ok is not None:
                paths.append(str(ok))
    return paths


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser("itcas.school_comparison")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument(
        "--config", type=str, default="configs/two_schools_of_thought.json",
        help="Path to the two-schools config (problems, difficulty, method groups).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=_DEFAULT_OUTPUT_DIR,
        help=f"Directory to write grid PDFs to (default: {_DEFAULT_OUTPUT_DIR}).",
    )
    args = parser.parse_args(argv)

    paths = summarize_school_comparison(
        args.input_dir, config_path=args.config, output_dir=args.output_dir
    )
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
