"""CLI entry point. Usage:

    python -m itcas.cli --config configs/smoke.yaml
    python -m itcas.cli --method itcas --problem two_circles_2d --budget 30 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, replace

from .pipeline import ExperimentConfig, PROBLEM_REGISTRY, run_experiment
from .utils.seeds import (
    RunBusyError,
    RunLock,
    is_run_complete,
    parse_seed_spec,
    pending_seeds,
    run_name_for_seed,
)


def _load_yaml(path: str) -> dict:
    try:
        import yaml  # type: ignore
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # Allow JSON configs without a YAML dependency
        with open(path) as f:
            return json.load(f)


def parse_args(argv=None) -> tuple[ExperimentConfig, str]:
    p = argparse.ArgumentParser("itcas")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--problem", type=str, default="two_circles_2d")
    p.add_argument("--method", type=str, default="itcas")
    p.add_argument(
        "--quality", type=str, default=None,
        help="itcas candidate-quality variant feeding the QD-DPP: "
             "'roi_mi' (global region-of-interest MI, default), "
             "'efig' (PoF-weighted info gain), "
             "'edig' (depth-weighted info gain, resolves EFIG saturation/cold-start), or "
             "'ndig' (rationally-squashed depth-weighted info gain, restores DPP context diversity).",
    )
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--n_init", type=int, default=None)
    p.add_argument("--n_candidates", type=int, default=None)
    p.add_argument("--n_ts_samples", type=int, default=None)
    p.add_argument("--dpp_lambda", type=float, default=None)
    p.add_argument("--dpp_lambda_ctx", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None,
                   help="Smooth-margin (LogSumExp) softmin constant for itcas.")
    p.add_argument("--n_restarts", type=int, default=None,
                   help="Multi-start restarts for the continuous itcas optimizer.")
    p.add_argument("--n_opt_steps", type=int, default=None,
                   help="Gradient steps per restart for the continuous itcas optimizer.")
    p.add_argument("--opt_lr", type=float, default=None,
                   help="Adam learning rate for the continuous itcas optimizer.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--seeds", type=str, default=None,
        help="Run a sequence of seeds, skipping ones already completed. "
             "Spec is comma-separated singletons and inclusive ranges, e.g. "
             "'1-10' or '1,3,6-8'. Each seed's run is named per the run_name "
             "template (use the literal '{seed}' to control placement, else "
             "'_seed<n>' is appended). Overrides --seed.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="With --seeds, re-run seeds even if their summary already exists.",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="With --seeds, print which seeds would run (completed vs pending) "
             "without executing them.",
    )
    p.add_argument(
        "--compare_dir", type=str, default=None,
        help="Directory under which per-metric comparison PDFs are written at "
             "the end of a --seeds sweep. Defaults to the highest ancestor of "
             "--out_dir that still contains runs for this problem and at least "
             "two methods.",
    )
    p.add_argument("--target_X", type=int, default=None)
    p.add_argument("--radius", type=float, default=None)
    p.add_argument("--obj_radius", type=float, default=None)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--soft_lambda", type=float, default=None)
    p.add_argument("--eps_archive", type=float, default=None,
                   help="ε-Archive Size distance threshold (normalised objective space). "
                        "Defaults to the problem's built-in value.")
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument(
        "--device", type=str, default=None,
        help="Compute device: auto (default) | cpu | cuda | cuda:N. "
             "'auto' uses CUDA when available, else CPU.",
    )
    p.add_argument(
        "--threshold_pct", type=float, default=None,
        help="Load calibrated thresholds for this difficulty (joint-feasible "
             "fraction) from configs/thresholds.json instead of the problem "
             "defaults.",
    )
    p.add_argument(
        "--thresholds_path", type=str, default=None,
        help="Path to the calibrated thresholds store (default: configs/thresholds.json).",
    )
    args = p.parse_args(argv)

    cfg_dict: dict = {}
    if args.config:
        cfg_dict.update(_load_yaml(args.config))
    problem = cfg_dict.pop("problem", args.problem)
    # threshold selection is consumed in main(), not part of ExperimentConfig
    threshold_pct = cfg_dict.pop("threshold_pct", None)
    thresholds_path = cfg_dict.pop("thresholds_path", None)
    # seed-sweep controls are also consumed in main(), not config fields
    seeds_spec = cfg_dict.pop("seeds", None)
    force = bool(cfg_dict.pop("force", False))
    dry_run = bool(cfg_dict.pop("dry_run", False))
    if args.threshold_pct is not None:
        threshold_pct = args.threshold_pct
    if args.thresholds_path is not None:
        thresholds_path = args.thresholds_path
    if args.seeds is not None:
        seeds_spec = args.seeds
    if args.force:
        force = True
    if args.dry_run:
        dry_run = True

    for k in ExperimentConfig.__dataclass_fields__:
        v = getattr(args, k, None)
        if v is not None:
            cfg_dict[k] = v
    cfg = ExperimentConfig(**cfg_dict)
    cfg.extra["threshold_pct"] = threshold_pct
    cfg.extra["thresholds_path"] = thresholds_path
    cfg.extra["seeds"] = seeds_spec
    cfg.extra["force"] = force
    cfg.extra["dry_run"] = dry_run
    cfg.extra["compare_dir"] = args.compare_dir
    return cfg, problem


def _build_problem(problem_name: str, cfg: ExperimentConfig):
    threshold_pct = cfg.extra.get("threshold_pct")
    thresholds_path = cfg.extra.get("thresholds_path")
    if threshold_pct is not None:
        from .pipeline.thresholds import DEFAULT_THRESHOLDS_PATH, load_thresholds
        tau = load_thresholds(
            problem_name, percentage=threshold_pct,
            path=thresholds_path or DEFAULT_THRESHOLDS_PATH,
        )
        return PROBLEM_REGISTRY[problem_name](thresholds=tau.tolist())
    return PROBLEM_REGISTRY[problem_name]()


def _run_one(problem, cfg: ExperimentConfig) -> dict:
    os.makedirs(cfg.out_dir, exist_ok=True)
    # Hold an advisory lock for the whole run so a second concurrent
    # invocation targeting the same (out_dir, run_name) aborts immediately
    # instead of truncating the in-progress JSONL.
    with RunLock(cfg.out_dir, cfg.run_name):
        summary = run_experiment(problem, cfg)
    print(json.dumps({"config": asdict(cfg), "summary": {
        k: v for k, v in summary.items()
        if k not in {"cumulative_positives", "config"}
    }}, indent=2, default=str))
    return summary


def _run_seed_sweep(problem, cfg: ExperimentConfig, seeds_spec) -> int:
    """Run a sequence of seeds, skipping those already completed."""
    force = bool(cfg.extra.get("force", False))
    dry_run = bool(cfg.extra.get("dry_run", False))

    all_seeds = parse_seed_spec(seeds_spec)
    if not all_seeds:
        raise SystemExit(f"--seeds '{seeds_spec}' expanded to no seeds.")

    todo = pending_seeds(all_seeds, cfg.out_dir, cfg.run_name, force=force)
    done = [s for s in all_seeds if s not in set(todo)]

    print(
        f"[seed-sweep] target={all_seeds}\n"
        f"[seed-sweep] completed (skipped)={done}\n"
        f"[seed-sweep] pending (to run)={todo}"
        + ("  [FORCE: re-running all]" if force else "")
    )
    if dry_run:
        print("[seed-sweep] dry run; not executing.")
        return 0

    for s in todo:
        run_name = run_name_for_seed(cfg.run_name, s)
        seed_cfg = replace(cfg, seed=s, run_name=run_name)
        seed_cfg.extra = dict(cfg.extra)
        print(f"\n[seed-sweep] === seed {s} -> {run_name} ===")
        try:
            _run_one(problem, seed_cfg)
        except RunBusyError as exc:
            print(f"[seed-sweep] seed {s} skipped: {exc}")
            continue
    print(f"\n[seed-sweep] done: ran {len(todo)} seed(s), skipped {len(done)}.")

    # Auto-generate per-metric comparison plots across every method that has
    # finished runs in this output directory for this problem. Runs are
    # typically organised as ``<root>/<problem>/<difficulty>/<method>/``, so the
    # per-method ``out_dir`` would only see one method; walk up to the highest
    # ancestor that still has runs for this problem and contains >=2 methods.
    try:
        from .reporting.visualize import resolve_comparison_root, visualize_benchmark

        compare_dir = cfg.extra.get("compare_dir")
        if compare_dir is None:
            compare_dir = str(resolve_comparison_root(cfg.out_dir, problem=problem.name))
        paths = visualize_benchmark(compare_dir, benchmark=problem.name)
        if paths:
            print(f"[seed-sweep] wrote {len(paths)} comparison PDF(s) under {compare_dir}:")
            for p in paths:
                print(f"  {p}")
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        print(f"[seed-sweep] visualization skipped: {exc}")
    return 0


def main(argv=None) -> int:
    cfg, problem_name = parse_args(argv)
    if problem_name not in PROBLEM_REGISTRY:
        raise SystemExit(f"Unknown problem '{problem_name}'. Choices: {list(PROBLEM_REGISTRY)}")

    problem = _build_problem(problem_name, cfg)

    seeds_spec = cfg.extra.get("seeds")
    if seeds_spec is not None:
        return _run_seed_sweep(problem, cfg, seeds_spec)

    # Single-seed mode (typical Slurm-array pattern): also skip if the run
    # already has a finalized summary, unless --force was passed. This
    # prevents overlapping invocations on the same path from truncating an
    # in-progress JSONL and producing interleaved/duplicated records.
    force = bool(cfg.extra.get("force", False))
    dry_run = bool(cfg.extra.get("dry_run", False))
    if not force and is_run_complete(cfg.out_dir, cfg.run_name):
        print(
            f"[skip] run already complete: "
            f"{os.path.join(cfg.out_dir, cfg.run_name)}.summary.json "
            "(pass --force to re-run)"
        )
        return 0
    if dry_run:
        print(f"[dry-run] would run {cfg.run_name} in {cfg.out_dir}")
        return 0

    _run_one(problem, cfg)
    return 0


def main_cli(argv=None) -> int:
    try:
        return main(argv)
    except RunBusyError as exc:
        # Exit 75 == EX_TEMPFAIL: signals "another worker has this; try later"
        # so a Slurm wrapper can distinguish concurrent-attempt aborts from
        # real failures.
        print(f"[busy] {exc}")
        return 75


if __name__ == "__main__":
    sys.exit(main_cli())
