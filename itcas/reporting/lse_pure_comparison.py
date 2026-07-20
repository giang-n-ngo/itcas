"""Pure LSE (Straddle/BES) vs its own LSE-then-sample variants, per problem.

Where :mod:`itcas.reporting.lse_ratio_comparison` asks "does the Stage-1/LSE
split *ratio* matter, among LSE-then-sample siblings" and
:mod:`itcas.reporting.school_comparison` asks "LSE-then-sample vs the
unrelated CAS acquisitions", this module asks a narrower question: **does
switching to Stage-2 (penalized interior sampling) at all help, relative to
just running the Stage-1 level-set-estimation acquisition for the whole
budget?** For each of the two LSE bases (``straddle``, ``bes``), it compares
that base's **pure** (single-stage, never switches to Stage-2) acquisition
against its three ``_lseNN`` "LSE-then-sample" siblings (10/25/50% Stage-1
split, see ``itcas.pipeline.loop._parse_two_stage``/``TwoStageSpec`` and
``itcas.reporting.lse_ratio_comparison``'s module docstring for the Family-C
construction) -- 4 lines per panel.

Method names (see ``configs/final_methods.json``'s ``"straddle"``/``"bes"``
families):

* Pure LSE, sequential: ``straddle`` / ``bes``. Batch: ``straddle_batch`` /
  ``bes_batch``.
* LSE-then-sample, sequential: ``{base}_then_sample_lse{10,25,50}``. Batch:
  ``{base}_then_sample_lse{10,25,50}_batch``.

**Grid: one PDF per (setting, base, problem).** Unlike
``lse_ratio_comparison``'s rows-by-problem grids, each PDF here holds a
*single* row (one problem, at the report's one fixed difficulty level -- see
below), mirroring the one-PDF-per-row split
:mod:`itcas.reporting.ndig_comparison` / :mod:`itcas.reporting.ff_comparison`
/ :mod:`itcas.reporting.casd_comparison` already use, just with "row" =
"problem" instead of "difficulty level" (this report has no difficulty axis
to sweep -- see below). Following that same precedent, each single-row PDF
drops the trailing **product-rank column**
(``include_product_rank_column=False``): a per-iteration "who's ahead right
now" rank column only earns its keep across a shared multi-row grid: with one
row it adds nothing the metric + product columns don't already show.

**Difficulty.** Fixed at the config's ``default_difficulty`` (0.05, i.e.
``p0_05``) -- unlike ``lse_ratio_comparison`` (which sweeps every difficulty
present), this report only ever needs the project's one "default" operating
point, exactly like ``school_comparison``'s own ``default_difficulty``.

**Problems.** Synthetic only: ``configs/lse_pure_comparison.json``'s
``"problems"`` list already excludes the two real-world problems
(``spacecraft_formation_flying_a1``, ``casd_llm`` -- see
``batch_vs_sequential._REAL_WORLD_PROBLEMS``), which have their own
per-problem difficulty scales that don't share a ``p0_05`` level with the
synthetic suite's shared scale.

**Axis per setting.** Exactly as in ``lse_ratio_comparison`` (see its module
docstring's "Axis per setting" section, reproduced here): two-stage
*sequential* mode burns a fixed Stage-1 budget of individual evaluations that
doesn't correspond to algorithmic steps, and the *pure* sequential LSE
acquisition also evaluates one point per iteration -- so **total individual
evaluations** (``RunSeries.x_evals``) is the shared axis for
``setting="sequential"``. Batch mode instead advances one algorithmic step
per batch regardless of batch size, so **algorithmic step**
(``RunSeries.x_steps``) is the shared axis for ``setting="batch"`` -- unlike
``lse_ratio_comparison`` (which only ships the sequential grids "by
request"), this report renders **both** settings, since the whole point here
is comparing pure-vs-staged *within* each setting on its own footing.

**Color = Stage-1 budget.** The pure method gets a fixed reserved hue (tab10
red, ``_PURE_COLOR``) and the three ``_lseNN`` siblings reuse
``lse_ratio_comparison._KNOWN_RATIO_COLORS`` (10%=blue, 25%=orange, 50%=green)
verbatim, so "blue always means 10%" stays true across every report in this
package that plots an LSE proportion. Every line in a given panel is solid
(``linestyle="-"``): a panel only ever holds one base's family, so there is no
second style axis to encode (unlike ``lse_ratio_comparison``, where
color=ratio/linestyle=base because two bases share one panel there).

**One summary-PDF pair per (setting, base) group.** Mirroring the avg-rank +
relative-AUC pair every other per-problem report in this package produces
(``ndig_comparison``, ``ff_comparison``, ``casd_comparison`` -- each via
:func:`itcas.reporting.ranking.average_ranks_over_rows` /
:func:`~itcas.reporting.ranking.relative_auc_ratios_over_rows` and
:func:`itcas.reporting.summary._plot_avg_rank_figure` /
:func:`~itcas.reporting.summary._plot_relative_auc_figure`), this report
produces one avg-rank PDF and one relative-AUC PDF **per (setting, base)
group** -- 2 settings x 2 bases = 8 summary PDFs total, each aggregating just
that group's own 4 methods across every problem present at the default
difficulty. Sequential and batch are never combined into one figure (they'd
need different x-axes, see above), and Straddle and BES are never combined
into one figure either -- each summary bar chart holds exactly the same 4
methods (pure + 3 ratios) as that group's own per-problem PDFs above, just
aggregated across problems instead of shown per problem. An earlier version
of this report merged everything into two combined 16-method summary
figures; that made "which of these bars is even comparable to which"
ambiguous (mixing two unrelated bases, and two axes, in one chart), so the
grouping was split to match the per-problem PDFs' own (setting, base)
grouping instead.

No figure in this module renders a ``suptitle``: ``plot_group_grid`` never
draws one regardless of what's passed (see its own docstring), and the
summary figures only ever draw per-column titles. This report ships with no
statistical-testing section -- like ``lse_ratio_comparison``/
``school_comparison`` (the closer analogs here: cross-cutting comparisons
among a family's own siblings, not a proposed-vs-baselines comparison),
there is no natural "proposed" method among "pure vs. 10%/25%/50% Stage-1
split", and the Product/Product-rank columns plus the summary PDFs already
answer "who's ahead" within each group.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from . import ranking
from .batch_vs_sequential import (
    _Row,
    _collect_family_runs,
    _filter_to_difficulty,
    plot_group_grid,
)
from .lse_ratio_comparison import _KNOWN_RATIO_COLORS
from .school_comparison import _difficulty_label
from .summary import (
    CurveCache,
    _metrics_present_in_rows,
    _plot_avg_rank_figure,
    _plot_relative_auc_figure,
    _precompute,
)

_DEFAULT_CONFIG = "configs/lse_pure_comparison.json"
_DEFAULT_OUTPUT_DIR = "results/lse_pure_comparison"

# One shared axis per setting (see module docstring's "Axis per setting").
# axis_suffix is used verbatim in output filenames.
_AXIS_BY_SETTING: dict[str, tuple[str, str]] = {
    "sequential": ("evals", "evaluations"),
    "batch": ("steps", "steps"),
}

# Reserved hue for the pure (single-stage) method -- the next unused tab10
# hue after lse_ratio_comparison's 10%/25%/50% (blue/orange/green), matching
# that module's own deterministic fallback ordering (see
# lse_ratio_comparison._ratio_color_map).
_PURE_COLOR = "#d62728"  # tab10 red

_BASE_ABBR: dict[str, str] = {"straddle": "STR", "bes": "BES"}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def _load_config(config_path: str | Path = _DEFAULT_CONFIG) -> dict:
    with Path(config_path).open() as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Method names / styles / labels for one (base, setting) group
# ---------------------------------------------------------------------------
def _pure_method(base: str, setting: str) -> str:
    """Return the pure (single-stage) method name for ``base`` at ``setting``."""
    return base if setting == "sequential" else f"{base}_batch"


def _sample_method(base: str, proportion: int, setting: str) -> str:
    """Return the ``_lseNN`` LSE-then-sample method name for ``base`` at ``setting``."""
    suffix = "_batch" if setting == "batch" else ""
    return f"{base}_then_sample_lse{proportion}{suffix}"


def group_methods(base: str, proportions: list[int], setting: str) -> list[str]:
    """Return ``[pure, lse10, lse25, ...]`` for one (base, setting) panel."""
    return [_pure_method(base, setting)] + [
        _sample_method(base, p, setting) for p in proportions
    ]


def _group_method_styles(base: str, proportions: list[int], setting: str) -> dict[str, dict]:
    """``{method: {"color", "linestyle"}}`` for one panel (see module docstring)."""
    styles = {_pure_method(base, setting): {"color": _PURE_COLOR, "linestyle": "-"}}
    for p in proportions:
        color = _KNOWN_RATIO_COLORS.get(p, _PURE_COLOR)
        styles[_sample_method(base, p, setting)] = {"color": color, "linestyle": "-"}
    return styles


def _method_label(base: str, proportion: Optional[int], setting: str) -> str:
    """Short legend/axis-tick label, e.g. ``"STR pure"``, ``"BES TS-25 (batch)"``."""
    abbr = _BASE_ABBR.get(base, base.upper())
    tag = "pure" if proportion is None else f"TS-{proportion}"
    suffix = " (batch)" if setting == "batch" else ""
    return f"{abbr} {tag}{suffix}"


def _group_method_labels(base: str, proportions: list[int], setting: str) -> dict[str, str]:
    labels = {_pure_method(base, setting): _method_label(base, None, setting)}
    for p in proportions:
        labels[_sample_method(base, p, setting)] = _method_label(base, p, setting)
    return labels


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
def summarize_lse_pure_comparison(
    input_dir: str | Path,
    config_path: str | Path = _DEFAULT_CONFIG,
    output_dir: str | Path | None = None,
) -> list[str]:
    """Produce every per-(setting, base, problem) PDF plus the per-group summary PDFs.

    See the module docstring for the full layout/axis/color rationale, and
    "One summary-PDF pair per (setting, base) group" specifically for why the
    summary PDFs are never combined across settings or across bases.
    """
    input_path = Path(input_dir)
    cfg = _load_config(config_path)
    problems = list(cfg["problems"])
    diff_label = _difficulty_label(cfg["default_difficulty"])
    proportions = [int(p) for p in cfg["lse_proportions"]]
    bases = list(cfg["bases"])
    out_dir = Path(output_dir) if output_dir is not None else Path(_DEFAULT_OUTPUT_DIR)

    paths: list[str] = []

    for setting, (axis, axis_suffix) in _AXIS_BY_SETTING.items():
        for base in bases:
            methods = group_methods(base, proportions, setting)
            styles = _group_method_styles(base, proportions, setting)
            labels = _group_method_labels(base, proportions, setting)

            runs_by_problem = _collect_family_runs(input_path, problems, methods)
            caches_by_problem: dict[str, CurveCache] = {
                p: _precompute(runs) for p, runs in runs_by_problem.items() if runs
            }
            problems_present, diff_runs, diff_caches = _filter_to_difficulty(
                problems, runs_by_problem, caches_by_problem, diff_label
            )

            group_rows: list[_Row] = []
            for problem in problems_present:
                row: _Row = (problem, diff_runs[problem], diff_caches.get(problem, {}))
                group_rows.append(row)
                out_path = out_dir / f"{setting}_{base}_{problem}_vs_{axis_suffix}.pdf"
                ok = plot_group_grid(
                    methods, styles, [row], None, out_path,
                    include_product_rank_column=False, method_labels=labels,
                )
                if ok is not None:
                    paths.append(str(ok))

            if not group_rows:
                continue

            # Per-(setting, base) summary PDFs (see module docstring): avg
            # rank + relative-AUC, aggregated across this group's own
            # problems and its own 4 methods only -- never mixed with the
            # other base or the other setting.
            metrics_present = _metrics_present_in_rows(group_rows)

            avg_rank_row = ranking.average_ranks_over_rows(group_rows, methods, axis)
            avg_rank_path = out_dir / f"{setting}_{base}_avg_rank_vs_{axis_suffix}.pdf"
            ok = _plot_avg_rank_figure(
                avg_rank_row, methods, styles, metrics_present, len(group_rows),
                avg_rank_path, method_labels=labels,
            )
            if ok is not None:
                paths.append(str(ok))

            relative_auc_row = ranking.relative_auc_ratios_over_rows(group_rows, methods, axis)
            relative_auc_path = out_dir / f"{setting}_{base}_relative_auc_vs_{axis_suffix}.pdf"
            ok = _plot_relative_auc_figure(
                relative_auc_row, methods, styles, metrics_present, len(group_rows),
                relative_auc_path, method_labels=labels,
            )
            if ok is not None:
                paths.append(str(ok))

    return paths


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser("itcas.lse_pure_comparison")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument(
        "--config", type=str, default=_DEFAULT_CONFIG,
        help="Path to the pure-vs-LSE-then-sample config (default_difficulty, "
        "lse_proportions, bases, problems).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=_DEFAULT_OUTPUT_DIR,
        help=f"Directory to write PDFs to (default: {_DEFAULT_OUTPUT_DIR}).",
    )
    args = parser.parse_args(argv)

    paths = summarize_lse_pure_comparison(
        args.input_dir, config_path=args.config, output_dir=args.output_dir
    )
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
