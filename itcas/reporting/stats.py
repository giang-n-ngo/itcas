"""Non-parametric statistical comparison of methods (see ``contexts/metrics.md``).

Pipeline (per problem × difficulty):

1. **Friedman test** (omnibus): tests whether *any* method differs across all
   ``K`` methods using the paired-seed hypervolume matrix (N_seeds × K).
   Proceed to step 2 only if ``p < alpha``.
2. **Wilcoxon signed-rank test** (pairwise, one-sided "greater"): tests
   whether the *proposed method* has strictly **higher** hypervolume than each
   baseline (higher hypervolume = better performance).
3. **Holm-Bonferroni correction**: controls FWER over the ``K-1`` pairwise
   comparisons.

The main entry-point is :func:`run_stats`, which returns a structured
:class:`StatsReport` dataframe that can be serialised to JSON and rendered as
a Markdown/PDF table via :func:`write_stats_report`.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class PairwiseResult:
    """Wilcoxon signed-rank test result for one (proposed vs baseline) comparison."""
    baseline: str
    stat: float
    p_raw: float
    p_adj: float          # Holm-Bonferroni corrected
    significant: bool     # p_adj < alpha
    effect_median_diff: float   # median(proposed - baseline) — effect size proxy


@dataclass
class GroupResult:
    """Full statistical result for one (problem, difficulty, axis) group."""
    problem: str
    difficulty: str
    axis: str             # "evals" | "steps"
    n_seeds: int
    methods: list[str]
    proposed_method: str
    friedman_stat: float
    friedman_p: float
    friedman_significant: bool
    pairwise: list[PairwiseResult] = field(default_factory=list)
    note: str = ""        # e.g. "skipped: omnibus not significant"


@dataclass
class StatsReport:
    alpha: float
    proposed_method: str
    groups: list[GroupResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Holm-Bonferroni correction
# ---------------------------------------------------------------------------
def _holm_bonferroni(p_values: list[float]) -> list[float]:
    """Return Holm-Bonferroni adjusted p-values (same length as input)."""
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    max_adj = 0.0
    for rank, idx in enumerate(order):
        adj = p_values[idx] * (n - rank)
        adj = min(adj, 1.0)
        # Enforce monotonicity: adjusted p can only increase along the sorted order.
        max_adj = max(max_adj, adj)
        adjusted[idx] = max_adj
    return adjusted


# ---------------------------------------------------------------------------
# Core statistical tests
# ---------------------------------------------------------------------------
def _friedman(matrix: np.ndarray) -> tuple[float, float]:
    """Friedman test on an (n_seeds, n_methods) matrix.

    Returns ``(statistic, p_value)``.  Delegates to
    ``scipy.stats.friedmanchisquare``.
    """
    from scipy.stats import friedmanchisquare  # lazy import

    n, k = matrix.shape
    if k < 2:
        return float("nan"), 1.0
    if n < 3:
        # Not enough observations for a reliable Friedman test.
        return float("nan"), 1.0
    # scipy wants each group as a separate positional argument.
    stat, p = friedmanchisquare(*[matrix[:, j] for j in range(k)])
    return float(stat), float(p)


def _wilcoxon_greater(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """One-sided Wilcoxon signed-rank test: H1: x > y (x has higher hypervolume).

    Higher hypervolume is better, so this tests whether the proposed method
    significantly outperforms the baseline. Returns ``(statistic, p_value)``.
    Falls back to ``nan`` when all differences are zero (ties) or the sample
    is too small.
    """
    from scipy.stats import wilcoxon  # lazy import

    diff = x - y
    if len(diff) < 5 or np.all(diff == 0):
        return float("nan"), 1.0
    try:
        result = wilcoxon(diff, alternative="greater", zero_method="zsplit")
        return float(result.statistic), float(result.pvalue)
    except ValueError:
        return float("nan"), 1.0


# ---------------------------------------------------------------------------
# Main entry: run the full pipeline for one (problem, difficulty) group
# ---------------------------------------------------------------------------
def _run_group(
    problem: str,
    difficulty: str,
    axis: str,
    hv_by_method: dict[str, list[float]],
    proposed_method: str,
    alpha: float,
) -> GroupResult:
    """Run the three-step pipeline for one (problem, difficulty, axis) group."""
    methods = sorted(hv_by_method.keys())
    n_seeds = max(len(v) for v in hv_by_method.values()) if hv_by_method else 0

    # Build a common seed-aligned matrix, padding shorter series with NaN.
    # Seeds are implicitly aligned by position (same index = same seed).
    max_len = max((len(v) for v in hv_by_method.values()), default=0)
    matrix = np.full((max_len, len(methods)), np.nan)
    for j, m in enumerate(methods):
        vals = hv_by_method.get(m, [])
        matrix[: len(vals), j] = vals

    # Drop rows (seeds) that have any NaN.
    complete_rows = ~np.isnan(matrix).any(axis=1)
    matrix_clean = matrix[complete_rows]
    n_complete = int(matrix_clean.shape[0])

    proposed_idx = methods.index(proposed_method) if proposed_method in methods else -1

    result = GroupResult(
        problem=problem,
        difficulty=difficulty,
        axis=axis,
        n_seeds=n_complete,
        methods=methods,
        proposed_method=proposed_method,
        friedman_stat=float("nan"),
        friedman_p=1.0,
        friedman_significant=False,
    )

    if n_complete < 3 or len(methods) < 3:
        result.note = f"skipped: only {n_complete} complete seeds or {len(methods)} methods"
        return result

    # Step 1 — Friedman
    f_stat, f_p = _friedman(matrix_clean)
    result.friedman_stat = f_stat
    result.friedman_p = f_p
    result.friedman_significant = f_p < alpha

    if not result.friedman_significant:
        result.note = f"omnibus not significant (p={f_p:.4f}); pairwise tests skipped"
        return result

    if proposed_idx < 0:
        result.note = f"proposed method '{proposed_method}' not in group; pairwise skipped"
        return result

    # Step 2 — pairwise Wilcoxon (proposed vs each baseline)
    baselines = [m for m in methods if m != proposed_method]
    proposed_scores = matrix_clean[:, proposed_idx]
    raw_results: list[tuple[str, float, float, float]] = []
    for bl in baselines:
        bl_idx = methods.index(bl)
        bl_scores = matrix_clean[:, bl_idx]
        stat, p_raw = _wilcoxon_greater(proposed_scores, bl_scores)
        med_diff = float(np.nanmedian(proposed_scores - bl_scores))
        raw_results.append((bl, stat, p_raw, med_diff))

    # Step 3 — Holm-Bonferroni correction
    p_raws = [r[2] for r in raw_results]
    p_adjs = _holm_bonferroni(p_raws)

    result.pairwise = [
        PairwiseResult(
            baseline=bl,
            stat=stat,
            p_raw=p_raw,
            p_adj=p_adj,
            significant=p_adj < alpha,
            effect_median_diff=med_diff,
        )
        for (bl, stat, p_raw, med_diff), p_adj in zip(raw_results, p_adjs)
    ]
    return result


# ---------------------------------------------------------------------------
# Public entry-point: run stats over the full hypervolume collection
# ---------------------------------------------------------------------------
def _is_itcas_variant(method: str) -> bool:
    """Return True when *method* is any itcas quality variant.

    Matches ``"itcas"`` (bare), ``"itcas_efig"``, ``"itcas_edig"``,
    ``"itcas_ndig"``, ``"itcas_roi_mi"``, etc.
    """
    return method == "itcas" or method.startswith("itcas_")


def run_stats(
    hyper_by_problem: dict[str, dict[str, dict[str, dict[str, list[float]]]]],
    proposed_method: str = "itcas",
    alpha: float = 0.05,
) -> "StatsReport":
    """Run the full statistical pipeline for a *single* proposed method.

    Parameters
    ----------
    hyper_by_problem:
        Nested dict ``{problem: {axis: {difficulty: {method: [hv_seed_0, …]}}}}``.  
        Produced by :func:`itcas.reporting.summary._collect_hv_for_stats`.
    proposed_method:
        The focal method name tested against every other method in the data.
    alpha:
        Significance level.

    Returns
    -------
    StatsReport
    """
    report = StatsReport(alpha=alpha, proposed_method=proposed_method)
    for problem in sorted(hyper_by_problem):
        for axis in sorted(hyper_by_problem[problem]):
            for diff in sorted(hyper_by_problem[problem][axis]):
                hv_by_method = hyper_by_problem[problem][axis][diff]
                group = _run_group(
                    problem=problem,
                    difficulty=diff,
                    axis=axis,
                    hv_by_method=hv_by_method,
                    proposed_method=proposed_method,
                    alpha=alpha,
                )
                report.groups.append(group)
    return report


def run_stats_per_variant(
    hyper_by_problem: dict[str, dict[str, dict[str, dict[str, list[float]]]]],
    alpha: float = 0.05,
    itcas_prefix: str = "itcas",
) -> dict[str, "StatsReport"]:
    """Run the statistical pipeline once per itcas quality variant found in the data.

    Each itcas variant (methods matching ``itcas`` or ``itcas_*``) is treated as
    the *proposed* method in its own :class:`StatsReport`.  The baselines for
    that report are every method in the data that is **not** an itcas variant
    (i.e. does not start with ``"itcas"``).

    Parameters
    ----------
    hyper_by_problem:
        Nested dict ``{problem: {axis: {difficulty: {method: [hv_seed_0, …]}}}}``.  
        Produced by :func:`itcas.reporting.summary._collect_hv_for_stats`.
    alpha:
        Significance level for Friedman gate and Holm-Bonferroni correction.
    itcas_prefix:
        Prefix used to recognise itcas variants (default ``"itcas"``).

    Returns
    -------
    dict mapping variant name -> StatsReport (one entry per detected itcas variant).
    Returns an empty dict if no itcas variants are found.
    """
    # Collect all unique method names across the entire dataset.
    all_methods: set[str] = set()
    for axes in hyper_by_problem.values():
        for diffs in axes.values():
            for methods in diffs.values():
                all_methods.update(methods.keys())

    itcas_variants = sorted(m for m in all_methods if _is_itcas_variant(m))
    baselines = sorted(m for m in all_methods if not _is_itcas_variant(m))

    if not itcas_variants:
        return {}

    reports: dict[str, StatsReport] = {}
    for variant in itcas_variants:
        # Build a view of hv_data that contains only this variant + baselines,
        # so _run_group sees a clean method set.
        keep = {variant} | set(baselines)
        filtered: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}
        for problem, axes in hyper_by_problem.items():
            filtered[problem] = {}
            for axis, diffs in axes.items():
                filtered[problem][axis] = {}
                for diff, methods in diffs.items():
                    filtered[problem][axis][diff] = {
                        m: v for m, v in methods.items() if m in keep
                    }

        report = StatsReport(alpha=alpha, proposed_method=variant)
        for problem in sorted(filtered):
            for axis in sorted(filtered[problem]):
                for diff in sorted(filtered[problem][axis]):
                    hv_by_method = filtered[problem][axis][diff]
                    group = _run_group(
                        problem=problem,
                        difficulty=diff,
                        axis=axis,
                        hv_by_method=hv_by_method,
                        proposed_method=variant,
                        alpha=alpha,
                    )
                    report.groups.append(group)
        reports[variant] = report
    return reports
# Serialisation helpers
# ---------------------------------------------------------------------------
def report_to_dict(report: StatsReport) -> dict:
    return asdict(report)


def report_to_json(report: StatsReport, *, indent: int = 2) -> str:
    return json.dumps(report_to_dict(report), indent=indent)


# ---------------------------------------------------------------------------
# Markdown / text table rendering
# ---------------------------------------------------------------------------
_STAR = {True: "✓", False: "✗"}


def _sig_marker(sig: bool, p_adj: float) -> str:
    """Return a significance star string (**, *, ns)."""
    if not sig:
        return "ns"
    if p_adj < 0.001:
        return "***"
    if p_adj < 0.01:
        return "**"
    return "*"


def _fmt(x: float, precision: int = 4) -> str:
    if x != x:  # nan
        return "—"
    return f"{x:.{precision}f}"


def report_to_markdown(report: StatsReport) -> str:
    """Render the :class:`StatsReport` as a Markdown document."""
    lines: list[str] = []
    lines.append("# Statistical Comparison Report")
    lines.append("")
    lines.append(
        f"**Proposed method:** `{report.proposed_method}` &nbsp;|&nbsp; "
        f"**α =** {report.alpha}"
    )
    lines.append("")
    lines.append(
        "Significance markers: `***` p_adj < 0.001, `**` p_adj < 0.01, "
        "`*` p_adj < 0.05, `ns` not significant after Holm-Bonferroni."
    )
    lines.append("")

    # Group by (problem, axis) for readability.
    from itertools import groupby

    def _key(g: GroupResult) -> tuple[str, str]:
        return (g.problem, g.axis)

    sorted_groups = sorted(report.groups, key=_key)
    for (problem, axis), grp_iter in groupby(sorted_groups, key=_key):
        grp = list(grp_iter)
        axis_label = "Evaluations" if axis == "evals" else "Steps"
        lines.append(f"## {problem} — x-axis: {axis_label}")
        lines.append("")

        for g in sorted(grp, key=lambda x: x.difficulty):
            lines.append(f"### Difficulty: `{g.difficulty}`")
            lines.append(
                f"Seeds (complete pairs): {g.n_seeds} &nbsp;|&nbsp; "
                f"Methods: {', '.join(f'`{m}`' for m in g.methods)}"
            )
            lines.append("")

            # Friedman row
            lines.append("**Step 1 — Friedman omnibus test**")
            lines.append("")
            lines.append(f"| Statistic | p-value | Significant (α={report.alpha}) |")
            lines.append("|----------:|--------:|:-------------------------------|")
            lines.append(
                f"| {_fmt(g.friedman_stat)} | {_fmt(g.friedman_p)} "
                f"| {'Yes ✓' if g.friedman_significant else 'No ✗'} |"
            )
            lines.append("")

            if g.note:
                lines.append(f"> _{g.note}_")
                lines.append("")
                continue

            # Pairwise table
            lines.append(
                f"**Step 2 & 3 — Wilcoxon (one-sided: `{g.proposed_method}` < baseline, "
                f"lower HV = better) with Holm-Bonferroni**"
            )
            lines.append("")
            lines.append(
                "| Baseline | W statistic | p (raw) | p (adj) | Sig | "
                "Median Δ (proposed − baseline; negative = proposed better) |"
            )
            lines.append(
                "|:---------|------------:|--------:|--------:|:---:|"
                "------------------------------------------------------:|"
            )
            for pw in sorted(g.pairwise, key=lambda x: x.p_adj):
                sig_str = _sig_marker(pw.significant, pw.p_adj)
                lines.append(
                    f"| `{pw.baseline}` "
                    f"| {_fmt(pw.stat, 2)} "
                    f"| {_fmt(pw.p_raw)} "
                    f"| {_fmt(pw.p_adj)} "
                    f"| {sig_str} "
                    f"| {_fmt(pw.effect_median_diff)} |"
                )
            lines.append("")

    # Summary table
    lines.extend(_summary_table(report))

    return "\n".join(lines)


def _summary_table(report: StatsReport) -> list[str]:
    """Build a summary table appended at the end of the Markdown report.

    One row per (problem, difficulty, axis).  Columns:
      * **Friedman** — p-value and ✓/✗ whether it passed α.
      * **Baselines beaten** — count of baselines with p_adj < α / total baselines.
      * **Dominant** — ✓ only when Friedman passed *and* every baseline was beaten.
    """
    if not report.groups:
        return []

    lines: list[str] = []
    lines.append("---")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"A group is **Dominant** when the Friedman test is significant (p < α = {report.alpha}) "
        f"**and** `{report.proposed_method}` has significantly **lower** hypervolume than "
        f"*all* baselines (p_adj < α after Holm-Bonferroni; lower HV = better)."
    )
    lines.append("")
    lines.append(
        "| Problem | Difficulty | Axis | Friedman p | Friedman sig | "
        "Baselines beaten | Dominant |"
    )
    lines.append(
        "|:--------|:-----------|:-----|----------:|:------------:|"
        "----------------:|:--------:|"
    )

    for g in sorted(report.groups, key=lambda x: (x.axis, x.problem, x.difficulty)):
        axis_label = "Evals" if g.axis == "evals" else "Steps"
        friedman_p_str = _fmt(g.friedman_p, 4)
        friedman_sig = "✓" if g.friedman_significant else "✗"

        if g.pairwise:
            n_beaten = sum(1 for pw in g.pairwise if pw.significant)
            n_total = len(g.pairwise)
            beaten_str = f"{n_beaten} / {n_total}"
            dominant = g.friedman_significant and (n_beaten == n_total)
        elif g.note:
            beaten_str = "—"
            dominant = False
        else:
            beaten_str = "—"
            dominant = False

        dominant_str = "**✓**" if dominant else "✗"
        lines.append(
            f"| `{g.problem}` | `{g.difficulty}` | {axis_label} "
            f"| {friedman_p_str} | {friedman_sig} "
            f"| {beaten_str} | {dominant_str} |"
        )

    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Writer: JSON + Markdown side by side
# ---------------------------------------------------------------------------
def write_stats_report(
    report: StatsReport,
    out_dir: Path,
    stem: str = "stats_report",
) -> tuple[Path, Path]:
    """Write ``<stem>.json`` and ``<stem>.md`` to ``out_dir``.

    Returns ``(json_path, md_path)``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(report_to_json(report), encoding="utf-8")
    md_path.write_text(report_to_markdown(report), encoding="utf-8")
    return json_path, md_path


# ---------------------------------------------------------------------------
# Cross-variant dominance table
# ---------------------------------------------------------------------------
def dominance_table_markdown(
    reports: dict[str, "StatsReport"],
    *,
    axis: str = "evals",
) -> str:
    """Render a problem × variant dominance-count table as Markdown.

    Each cell shows ``dominated / total`` difficulty levels for that
    (variant, problem) pair on the chosen ``axis`` (``"evals"`` or
    ``"steps"``).  A difficulty is *dominated* when the Friedman test is
    significant **and** the variant significantly beats every baseline
    (p_adj < α after Holm-Bonferroni).

    Parameters
    ----------
    reports:
        ``{variant_name: StatsReport}`` — one report per itcas variant, as
        returned by :func:`run_stats_per_variant`.
    axis:
        Which x-axis to use for the dominance count (default ``"evals"``).
    """
    if not reports:
        return ""

    variants = sorted(reports)

    # Collect per-(variant, problem, difficulty) dominance.
    # dom[variant][problem][difficulty] = True/False
    dom: dict[str, dict[str, dict[str, bool]]] = {}
    all_problems: set[str] = set()
    for v, report in reports.items():
        dom[v] = {}
        for g in report.groups:
            if g.axis != axis:
                continue
            all_problems.add(g.problem)
            pw = g.pairwise
            dominant = (
                g.friedman_significant
                and len(pw) > 0
                and all(p.significant for p in pw)
            )
            dom[v].setdefault(g.problem, {})[g.difficulty] = dominant

    problems = sorted(all_problems)

    # Header
    v_labels = [f"`{v}`" for v in variants]
    header = "| Problem | " + " | ".join(v_labels) + " |"
    sep = "|:--------|" + "".join(":------:|" for _ in variants)

    lines: list[str] = []
    axis_label = "Evaluations" if axis == "evals" else "Steps"
    lines.append(f"## Dominance Count by Problem and Variant (x-axis: {axis_label})")
    lines.append("")
    lines.append(
        "Each cell shows **dominated / total** difficulty levels.  "
        "A difficulty is *dominated* when the Friedman test passes α "
        "**and** the variant significantly beats every baseline."
    )
    lines.append("")
    lines.append(header)
    lines.append(sep)

    totals = {v: 0 for v in variants}
    total_possible = {v: 0 for v in variants}

    for prob in problems:
        cells: list[str] = []
        for v in variants:
            diffs = dom.get(v, {}).get(prob, {})
            n_dom = sum(1 for d in diffs.values() if d)
            n_tot = len(diffs)
            totals[v] += n_dom
            total_possible[v] += n_tot
            cells.append(f"**{n_dom}** / {n_tot}" if n_dom > 0 else f"0 / {n_tot}")
        lines.append(f"| `{prob}` | " + " | ".join(cells) + " |")

    # Totals row
    total_cells = [
        f"**{totals[v]}** / {total_possible[v]}" for v in variants
    ]
    lines.append("| **Total** | " + " | ".join(total_cells) + " |")
    lines.append("")
    return "\n".join(lines)


def write_dominance_table(
    reports: dict[str, "StatsReport"],
    out_dir: Path,
    stem: str = "dominance_table",
) -> Path:
    """Write the cross-variant dominance table to ``<stem>.md`` in ``out_dir``.

    Generates one section per axis (Evaluations then Steps). Returns the path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{stem}.md"
    content = "# ITCAS Variant Dominance Table\n\n"
    content += dominance_table_markdown(reports, axis="evals") + "\n"
    content += dominance_table_markdown(reports, axis="steps") + "\n"
    md_path.write_text(content, encoding="utf-8")
    return md_path
