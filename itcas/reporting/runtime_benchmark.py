"""Live wall-clock runtime benchmark across methods x synthetic problems.

This is a **timing** tool, not a report over existing result logs: it
constructs an :class:`~itcas.pipeline.ExperimentConfig` and calls
:func:`~itcas.pipeline.run_experiment` directly, in-process, for one
(method, problem) cell at a time, measuring only the ``run_experiment(...)``
call with :func:`time.perf_counter`. It exists to produce a runtime-
comparison table for plotting (e.g. "how expensive is each acquisition,
averaged across the synthetic suite"), which the existing
``itcas.summarize``/``itcas.reporting.summary`` machinery cannot answer since
it only ever reads already-logged JSONL/summary files from disk.

**Methods** (11, the union of the two headline synthetic reports'
own method sets -- see ``itcas.reporting.summary.summarize_synthetic_comparison``
(``SYNTHETIC_METHODS``, forced-sequential baselines vs the proposed batch
method) and ``itcas.reporting.batch_improvement_comparison.FAMILIES``
(5 sequential/batch pairs, ``_lse10`` Family-C split)):

    itcas_ndig                          --method itcas          --quality ndig
    itcas_seq_ndig                       --method itcas_seq       --quality ndig
    random                               --method random
    straddle_then_sample_lse10           --method straddle_then_sample_lse10
    straddle_then_sample_lse10_batch     --method straddle_then_sample_lse10_batch
    bes_then_sample_lse10                --method bes_then_sample_lse10
    bes_then_sample_lse10_batch          --method bes_then_sample_lse10_batch
    cas_eci                              --method cas_eci
    cas_eci_batch                        --method cas_eci_batch
    moc_cas_hard                         --method moc_cas_hard
    moc_cas_hard_batch                   --method moc_cas_hard_batch

**Problems**: the 15 "synthetic" problems (i.e. every entry in
``configs/final_problems.json`` except the two real-world problems,
``spacecraft_formation_flying_a1``/``casd_llm``) -- reused directly from
:func:`itcas.reporting.summary._synthetic_problems` rather than re-derived.

**Sizing** (``budget``/``batch_size``/``n_init``/``eps_archive``): resolved
per problem from ``configs/experiments.json`` using the exact fallback chain
``scripts/run_seedset.sh`` uses (``problems[<problem>][<difficulty>] ->
problems[<problem>].defaults -> defaults``), with difficulty pinned to the
``"defaults"` tier (``run_seedset.sh``'s ``USE_THRESHOLD=0``/"default"
difficulty path) -- legitimate here since ``budget``/``batch_size``/``n_init``
never vary by difficulty for these problems (only ``eps_archive`` does, and
that is still resolved correctly via the ``.defaults`` fallback), so timing
numbers are unaffected by which difficulty tier would otherwise be picked.

**One run per (method, problem)**, fixed seed (default 0) -- this reports
per-problem wall-clock time plus an average across problems, not a seed
sweep with error bars (the user explicitly asked for "one run ... then
averaging across all problems").

**Two tables.** Every one of these 15 problems' resolved ``budget`` is either
100 or 200 (never anything else); rather than hardcoding that grouping as a
second source of truth, the budget groups are derived at runtime by
resolving each problem's own budget from ``configs/experiments.json`` and
grouping by the resulting value(s) -- so this still produces two tables
today and adapts automatically if the config changes, without silently
mislabeling a problem into the wrong table.

**Nothing is persisted**: each timed ``run_experiment`` call is pointed at a
throwaway ``tempfile.mkdtemp()`` directory (via ``--out_dir``), deleted with
``shutil.rmtree`` immediately after that call returns (success or failure).
Only the resulting timing table is meant to be kept -- printed to stdout,
and optionally written to ``--output`` as markdown.

Usage::

    python -m itcas.reporting.runtime_benchmark
    python -m itcas.reporting.runtime_benchmark --device cuda
    python -m itcas.reporting.runtime_benchmark --output results/runtime_benchmark.md
    # quick smoke test on CPU, small subset:
    python -m itcas.reporting.runtime_benchmark \\
        --methods random cas_eci --problems zdt3_6d alpine_12d --device cpu

Import order matters for fair timing: the heavy BoTorch/GPyTorch/PyTorch
stack is imported once at module load (via the ``itcas.pipeline`` import
below), so per-call timings measure only algorithmic cost, not repeated
process/import startup (which a subprocess-per-run design -- e.g. shelling
out to ``python -m itcas.cli`` 11 x 15 times -- would otherwise pay on every
single cell).
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

from ..pipeline import ExperimentConfig, PROBLEM_REGISTRY, run_experiment
from .summary import _synthetic_problems

_DEFAULT_EXPERIMENTS_CONFIG = "configs/experiments.json"
_DEFAULT_PROBLEMS_CONFIG = "configs/final_problems.json"

# (method_id -> (cli `--method` spelling, `--quality` or None)). Order is the
# row order used in the output tables: proposed method first, then the 5
# baseline families in the order given by
# itcas.reporting.summary.SYNTHETIC_BASELINE_METHODS, then the remaining
# batch/sequential siblings pulled in by
# itcas.reporting.batch_improvement_comparison.FAMILIES that aren't already
# covered above.
METHOD_SPECS: dict[str, tuple[str, Optional[str]]] = {
    "itcas_ndig": ("itcas", "ndig"),
    "itcas_seq_ndig": ("itcas_seq", "ndig"),
    "random": ("random", None),
    "straddle_then_sample_lse10": ("straddle_then_sample_lse10", None),
    "straddle_then_sample_lse10_batch": ("straddle_then_sample_lse10_batch", None),
    "bes_then_sample_lse10": ("bes_then_sample_lse10", None),
    "bes_then_sample_lse10_batch": ("bes_then_sample_lse10_batch", None),
    "cas_eci": ("cas_eci", None),
    "cas_eci_batch": ("cas_eci_batch", None),
    "moc_cas_hard": ("moc_cas_hard", None),
    "moc_cas_hard_batch": ("moc_cas_hard_batch", None),
}
DEFAULT_METHODS: tuple[str, ...] = tuple(METHOD_SPECS.keys())


def _resolve_experiment_params(
    problem: str, experiments_config: str | Path = _DEFAULT_EXPERIMENTS_CONFIG,
) -> dict[str, Optional[float]]:
    """Resolve budget/batch_size/n_init/eps_archive for ``problem``.

    Mirrors ``scripts/run_seedset.sh``'s ``xget`` fallback chain
    (``problems[<problem>][<difficulty>] -> problems[<problem>].defaults ->
    defaults``) pinned to the ``"defaults"`` difficulty tier -- since looking
    up ``problems[<problem>]["defaults"]`` first is identical to falling
    straight through to ``problems[<problem>].defaults``, this collapses to
    just that plus the top-level ``defaults`` fallback.
    """
    with Path(experiments_config).open() as f:
        cfg = json.load(f)
    top_defaults: dict = cfg.get("defaults", {})
    prob_cfg: dict = cfg.get("problems", {}).get(problem, {})
    prob_defaults: dict = prob_cfg.get("defaults", {})

    def _get(key: str):
        if key in prob_defaults and prob_defaults[key] is not None:
            return prob_defaults[key]
        return top_defaults.get(key)

    out = {
        "budget": _get("budget"),
        "batch_size": _get("batch_size"),
        "n_init": _get("n_init"),
        "eps_archive": _get("eps_archive"),
    }
    missing = [k for k in ("budget", "batch_size", "n_init") if out[k] is None]
    if missing:
        raise KeyError(
            f"could not resolve {missing} for problem '{problem}' from "
            f"'{experiments_config}' (problems.{problem}.defaults / defaults)"
        )
    return out


def _time_one_run(
    method_id: str,
    problem_name: str,
    *,
    seed: int,
    device: str,
    experiments_config: str | Path = _DEFAULT_EXPERIMENTS_CONFIG,
    budget_override: Optional[int] = None,
) -> float:
    """Run one (method, problem) cell and return wall-clock seconds.

    Builds the problem and :class:`ExperimentConfig` first (outside the timed
    region), points ``out_dir`` at a fresh temp directory, times only the
    ``run_experiment(...)`` call, then deletes the temp directory regardless
    of outcome. ``budget_override`` exists purely for fast smoke tests (bypass
    the real 100/200-evaluation sizing without touching the CLI/config path).
    """
    method, quality = METHOD_SPECS[method_id]
    if problem_name not in PROBLEM_REGISTRY:
        raise KeyError(f"unknown problem '{problem_name}'; choices: {sorted(PROBLEM_REGISTRY)}")

    params = _resolve_experiment_params(problem_name, experiments_config)
    budget = budget_override if budget_override is not None else params["budget"]

    problem = PROBLEM_REGISTRY[problem_name]()

    tmp_dir = tempfile.mkdtemp(prefix="itcas_runtime_bench_")
    try:
        cfg_kwargs: dict = dict(
            method=method,
            budget=budget,
            batch_size=params["batch_size"],
            n_init=params["n_init"],
            seed=seed,
            out_dir=tmp_dir,
            run_name=f"{problem_name}__{method_id}",
            device=device,
        )
        if quality is not None:
            cfg_kwargs["quality"] = quality
        if params.get("eps_archive") is not None:
            cfg_kwargs["eps_archive"] = params["eps_archive"]
        cfg = ExperimentConfig(**cfg_kwargs)

        t0 = time.perf_counter()
        run_experiment(problem, cfg)
        return time.perf_counter() - t0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def run_benchmark(
    methods: list[str],
    problems: list[str],
    *,
    seed: int = 0,
    device: str = "cuda",
    experiments_config: str | Path = _DEFAULT_EXPERIMENTS_CONFIG,
    budget_override: Optional[int] = None,
    verbose: bool = True,
) -> dict[str, dict[str, float]]:
    """Time every (method, problem) cell once. Returns timings[method][problem] = seconds."""
    timings: dict[str, dict[str, float]] = {m: {} for m in methods}
    total = len(methods) * len(problems)
    done = 0
    for problem_name in problems:
        for method_id in methods:
            elapsed = _time_one_run(
                method_id, problem_name,
                seed=seed, device=device,
                experiments_config=experiments_config,
                budget_override=budget_override,
            )
            timings[method_id][problem_name] = elapsed
            done += 1
            if verbose:
                print(
                    f"[{done}/{total}] {method_id} x {problem_name}: {elapsed:.2f}s",
                    flush=True,
                )
    return timings


def _format_seconds(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.1f}"


def _budget_groups(
    problems: list[str], experiments_config: str | Path,
) -> dict[int, list[str]]:
    """Group ``problems`` by their resolved ``budget``, preserving input order within each group."""
    groups: dict[int, list[str]] = {}
    for p in problems:
        budget = int(_resolve_experiment_params(p, experiments_config)["budget"])
        groups.setdefault(budget, []).append(p)
    return groups


def build_markdown_table(
    budget: int,
    problems: list[str],
    methods: list[str],
    timings: dict[str, dict[str, float]],
) -> str:
    """One table: rows = methods, columns = problems + an Average column."""
    lines: list[str] = []
    lines.append(f"## {budget}-evaluation synthetic problems ({len(problems)} problems)")
    lines.append("")
    header = "| Method | " + " | ".join(f"`{p}`" for p in problems) + " | **Average (s)** |"
    sep = "|:---|" + "---:|" * (len(problems) + 1)
    lines.append(header)
    lines.append(sep)
    for method_id in methods:
        row = timings.get(method_id, {})
        vals = [row.get(p) for p in problems]
        cells = " | ".join(_format_seconds(v) for v in vals)
        valid = [v for v in vals if v is not None]
        avg_str = _format_seconds(sum(valid) / len(valid) if valid else None)
        lines.append(f"| `{method_id}` | {cells} | **{avg_str}** |")
    return "\n".join(lines)


def build_report(
    timings: dict[str, dict[str, float]],
    methods: list[str],
    problems: list[str],
    experiments_config: str | Path,
) -> str:
    """Render one markdown table per budget group, sorted by budget ascending."""
    groups = _budget_groups(problems, experiments_config)
    sections = [
        "# Runtime benchmark: wall-clock seconds per (method, problem), one run per cell",
        "",
    ]
    for budget in sorted(groups):
        sections.append(build_markdown_table(budget, groups[budget], methods, timings))
        sections.append("")
    return "\n".join(sections)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "itcas.reporting.runtime_benchmark",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--methods", type=str, nargs="+", default=None,
        help=f"Method ids to time (default: all {len(DEFAULT_METHODS)}). "
             f"Choices: {', '.join(DEFAULT_METHODS)}.",
    )
    p.add_argument(
        "--problems", type=str, nargs="+", default=None,
        help="Problems to time (default: all 15 synthetic problems from --problems-config).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--device", type=str, default="cuda",
        help="Compute device (default: cuda, matching scripts/jobs.json's cluster default). "
             "Use 'cpu' for local smoke testing.",
    )
    p.add_argument("--experiments-config", type=str, default=_DEFAULT_EXPERIMENTS_CONFIG)
    p.add_argument("--problems-config", type=str, default=_DEFAULT_PROBLEMS_CONFIG)
    p.add_argument(
        "--output", type=str, default=None,
        help="Optional path to also write the markdown report to.",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    methods = args.methods if args.methods else list(DEFAULT_METHODS)
    unknown_methods = [m for m in methods if m not in METHOD_SPECS]
    if unknown_methods:
        raise SystemExit(
            f"Unknown method id(s) {unknown_methods}. Choices: {sorted(METHOD_SPECS)}"
        )

    problems = args.problems if args.problems else _synthetic_problems(args.problems_config)
    unknown_problems = [p for p in problems if p not in PROBLEM_REGISTRY]
    if unknown_problems:
        raise SystemExit(
            f"Unknown problem(s) {unknown_problems}. Choices: {sorted(PROBLEM_REGISTRY)}"
        )

    timings = run_benchmark(
        methods, problems,
        seed=args.seed, device=args.device,
        experiments_config=args.experiments_config,
    )

    report = build_report(timings, methods, problems, args.experiments_config)
    print("\n" + report)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report + "\n")
        print(f"\n[runtime_benchmark] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
