"""Pure LSE (Straddle/BES) vs its own LSE-then-sample variants -- combined summary.

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
construction). The pure acquisition is treated as a fourth, 100%-Stage-1
variant of that same family -- "100% of the budget spent in Stage-1/LSE" --
so every base's four variants (10%, 25%, 50%, 100%) sit on one shared scale.

Method names (see ``configs/final_methods.json``'s ``"straddle"``/``"bes"``
families):

* Pure LSE (the 100% variant), sequential: ``straddle`` / ``bes``. Batch:
  ``straddle_batch`` / ``bes_batch``.
* LSE-then-sample (10/25/50%), sequential: ``{base}_then_sample_lse{10,25,50}``.
  Batch: ``{base}_then_sample_lse{10,25,50}_batch``.

**Two output PDFs total: one per setting.** Unlike every other per-problem
report in this package, this module renders no per-problem breakdown and no
avg-rank/relative-AUC PDF *pair* -- just one combined figure per setting
(``sequential``, ``batch``), each aggregating the **product of the raw
metrics'** relative AUC (``ranking.relative_auc_ratios_over_rows``'s
``"product"`` column -- see that function's docstring) across every problem
present at the config's ``default_difficulty``, exactly the input the old
per-group relative-AUC summary PDFs used to plot.

**Layout.** Each figure has two groups on the y-axis, ``STR`` and ``BES``
(one per base, see ``cfg["bases"]``), each occupying its own band of up to
four horizontal boxplots placed next to each other (one per Stage-1/LSE
proportion: 10/25/50/100%). Each box summarizes that (base, proportion,
setting) method's relative-AUC-of-product ratio's spread **across problems**
-- one data point per problem present at ``default_difficulty`` (see
:func:`ranking.relative_auc_ratio_lists_over_rows`) -- rather than collapsing
straight to a single averaged bar. The y-axis carries one tick per group,
centered on that group's band of boxes, labeled ``STR`` / ``BES``.

**Color = Stage-1/LSE proportion**, fixed per proportion
(:func:`itcas.reporting.lse_ratio_comparison._ratio_color_map`: 10%=blue,
25%=orange, 50%=green, and -- by that function's own deterministic
fallback-hue ordering -- 100%=red) and reused verbatim between the STR and
BES bands, so "blue always means 10%" stays true both within this figure and
against every other report in this package that plots an LSE proportion
(``lse_ratio_comparison``). The plot's one legend maps these colors to their
proportions; there is no color for base -- the y-axis tick labels (``STR`` /
``BES``) are what tell a reader which band is which.

**Axis per setting.** Exactly as in ``lse_ratio_comparison``/``school_comparison``
(see either module's own docstring's "Axis per setting" section): two-stage
*sequential* mode burns a fixed Stage-1 budget of individual evaluations that
doesn't correspond to algorithmic steps, and the *pure* sequential LSE
acquisition also evaluates one point per iteration -- so **total individual
evaluations** (``RunSeries.x_evals``) is the shared axis for
``setting="sequential"``. Batch mode instead advances one algorithmic step per
batch regardless of batch size, so **algorithmic step** (``RunSeries.x_steps``)
is the shared axis for ``setting="batch"``. Both settings are rendered, since
the whole point here is comparing pure-vs-staged *within* each setting on its
own footing.

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

**The "best" a relative-AUC ratio is measured against is scoped to one base's
own four variants**, exactly like the old per-group summary PDFs:
``relative_auc_ratio_lists_over_rows`` is called once per (setting, base) with
only that base's own four methods, and the run/cache data for one base is
discarded before the next base is loaded. This is a deliberate memory bound,
not an oversight -- ``lse_pure_comparison.sbatch``'s own header documents an
OOM at 32GB from an earlier version of this module that loaded a whole
setting's worth of *both* bases (8 methods x 15 problems) into memory at
once; loading one base's 4 methods at a time keeps peak memory to what a
single (setting, base) group always required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from . import ranking
from .batch_vs_sequential import _collect_family_runs, _filter_to_difficulty
from .lse_ratio_comparison import _ratio_color_map
from .school_comparison import _difficulty_label
from .summary import CurveCache, _precompute

_DEFAULT_CONFIG = "configs/lse_pure_comparison.json"
_DEFAULT_OUTPUT_DIR = "results/lse_pure_comparison"

# One shared axis per setting (see module docstring's "Axis per setting").
# axis_suffix is used verbatim in output filenames.
_AXIS_BY_SETTING: dict[str, tuple[str, str]] = {
    "sequential": ("evals", "evaluations"),
    "batch": ("steps", "steps"),
}

_BASE_ABBR: dict[str, str] = {"straddle": "STR", "bes": "BES"}

# The pure (single-stage) acquisition, folded in as the 100%-Stage-1 variant
# (see module docstring). Colors are keyed by proportion via
# lse_ratio_comparison._ratio_color_map (see module docstring's "Color =
# Stage-1/LSE proportion") -- that function's own deterministic fallback-hue
# ordering happens to land proportion 100 on tab10 red, the same reserved hue
# an earlier version of this module used for the pure method directly.
_PURE_PROPORTION = 100


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def _load_config(config_path: str | Path = _DEFAULT_CONFIG) -> dict:
    with Path(config_path).open() as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Method names for one (base, proportion, setting)
# ---------------------------------------------------------------------------
def _variant_method(base: str, proportion: int, setting: str) -> str:
    """Method name for one (base, proportion, setting).

    ``proportion == 100`` is the pure (single-stage) method; any other
    proportion is the corresponding ``_lseNN`` LSE-then-sample sibling.
    """
    if proportion == _PURE_PROPORTION:
        return base if setting == "sequential" else f"{base}_batch"
    suffix = "_batch" if setting == "batch" else ""
    return f"{base}_then_sample_lse{proportion}{suffix}"


def _proportion_label(proportion: int) -> str:
    return "100% (pure)" if proportion == _PURE_PROPORTION else f"{proportion}%"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_pure_vs_ts_figure(
    product_lists: dict[str, list[float]],
    bases: list[str],
    proportions: list[int],
    setting: str,
    out_path: str | Path,
) -> Optional[Path]:
    """Grouped-boxplot figure: one y-axis band per base, one box per Stage-1/LSE proportion.

    See the module docstring's "Layout"/"Color = Stage-1/LSE proportion"
    sections for the full rationale. ``product_lists`` is
    ``ranking.relative_auc_ratio_lists_over_rows(...)["product"]``, keyed by
    method name (already merged across bases -- see
    :func:`summarize_lse_pure_comparison`).
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if not product_lists:
        return None

    proportion_colors = _ratio_color_map(proportions)

    n_var = len(proportions)
    box_width = 0.7
    group_gap = 1.0  # extra vertical space between one base's band and the next

    fig, ax = plt.subplots(figsize=(8.0, 0.6 * n_var * len(bases) + 1.5))

    group_centers: dict[str, float] = {}
    any_box = False
    for g, base in enumerate(bases):
        base_y0 = g * (n_var + group_gap)
        group_centers[base] = base_y0 + (n_var - 1) / 2.0
        for j, proportion in enumerate(proportions):
            method = _variant_method(base, proportion, setting)
            values = product_lists.get(method)
            if not values:
                continue
            y = base_y0 + j
            color = proportion_colors[proportion]
            ax.boxplot(
                [values], positions=[y], vert=False, widths=box_width,
                showfliers=True, patch_artist=True,
                boxprops=dict(facecolor=color, edgecolor="black", linewidth=0.8),
                medianprops=dict(color="black", linewidth=1.3),
                whiskerprops=dict(color="black"),
                capprops=dict(color="black"),
                flierprops=dict(marker=".", markersize=3, alpha=0.6),
            )
            any_box = True

    if not any_box:
        plt.close(fig)
        return None

    last_y = (len(bases) - 1) * (n_var + group_gap) + (n_var - 1)
    ax.set_yticks([group_centers[b] for b in bases])
    ax.set_yticklabels([_BASE_ABBR.get(b, b.upper()) for b in bases], fontsize=15)
    ax.set_ylim(-0.5, last_y + 0.5)
    ax.invert_yaxis()  # first base (STR) on top, matching the module docstring's layout
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_xlabel("Average relative AUC of the product of the raw metrics", fontsize=13.5)

    legend_handles = [
        Patch(facecolor=proportion_colors[p], edgecolor="black", label=_proportion_label(p))
        for p in proportions
    ]
    ax.legend(
        handles=legend_handles, title="Stage-1/LSE proportion",
        loc="lower right", ncol=1, fontsize=12, title_fontsize=12, framealpha=0.9,
    )

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
def summarize_lse_pure_comparison(
    input_dir: str | Path,
    config_path: str | Path = _DEFAULT_CONFIG,
    output_dir: str | Path | None = None,
) -> list[str]:
    """Produce the two (one per setting) pure-vs-LSE-then-sample summary PDFs.

    See the module docstring for the full layout/axis/color rationale.
    """
    input_path = Path(input_dir)
    cfg = _load_config(config_path)
    problems = list(cfg["problems"])
    diff_label = _difficulty_label(cfg["default_difficulty"])
    proportions = [int(p) for p in cfg["lse_proportions"]] + [_PURE_PROPORTION]
    bases = list(cfg["bases"])
    out_dir = Path(output_dir) if output_dir is not None else Path(_DEFAULT_OUTPUT_DIR)

    paths: list[str] = []

    for setting, (axis, axis_suffix) in _AXIS_BY_SETTING.items():
        # One base at a time -- see module docstring's "best" section: this
        # bounds peak memory to a single (setting, base) group's run/cache
        # data, discarded before the next base is loaded, instead of holding
        # both bases' worth (8 methods x every problem) at once.
        product_lists: dict[str, list[float]] = {}
        for base in bases:
            methods = [_variant_method(base, p, setting) for p in proportions]

            runs_by_problem = _collect_family_runs(input_path, problems, methods)
            caches_by_problem: dict[str, CurveCache] = {
                p: _precompute(runs) for p, runs in runs_by_problem.items() if runs
            }
            problems_present, diff_runs, diff_caches = _filter_to_difficulty(
                problems, runs_by_problem, caches_by_problem, diff_label
            )
            if problems_present:
                rows = [(p, diff_runs[p], diff_caches.get(p, {})) for p in problems_present]
                relative_auc = ranking.relative_auc_ratio_lists_over_rows(rows, methods, axis)
                product_lists.update(relative_auc.get("product", {}))
            del runs_by_problem, caches_by_problem, diff_runs, diff_caches

        if not product_lists:
            continue

        out_path = out_dir / f"pure_vs_ts_{setting}_vs_{axis_suffix}.pdf"
        ok = _plot_pure_vs_ts_figure(product_lists, bases, proportions, setting, out_path)
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
