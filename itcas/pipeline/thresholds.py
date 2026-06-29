"""Threshold (tau) calibration for the synthetic benchmarks.

Goal
----
For each benchmark we need feasibility thresholds tau = [tau_1, ..., tau_m]
such that a *target percentage* p of points sampled uniformly from the joint
input-context space satisfy **all** constraints simultaneously, i.e.

    Pr_{z ~ U(domain)} [ f_i(z) >= tau_i  for all i ]  ≈  p .

Procedure (matches the requested recipe)
----------------------------------------
1. Sample a large number N of points uniformly from the domain.
2. Evaluate the (noiseless) objectives -> Y in R^{N x m}.
3. Record the per-objective maxima (and minima) of the evaluations.
4. Iteratively adjust the thresholds until exactly the target fraction of the
   sampled points are jointly feasible.

How the thresholds are coupled (resolved ambiguity)
---------------------------------------------------
"Adjust each threshold until p% satisfy all constraints" is under-determined
for m > 1 (many tau vectors give the same joint fraction). We pick the unique,
scale-invariant choice that puts **every objective at the same marginal
quantile** q: tau_i = Quantile_q(f_i). Raising q raises every tau_i, so the
joint feasible fraction is monotonically non-increasing in q -> we bisect q in
[0, 1] until the joint fraction equals the target p. This anchors each tau_i
between its sampled min (q=0) and its sampled max (q=1) and keeps the per-
objective "difficulty" balanced.

The percentage p is a difficulty knob (smaller p => harder); default 0.10.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, Sequence

import torch

from .problems import REGISTRY, Problem

DEFAULT_THRESHOLDS_PATH = os.path.join("configs", "thresholds.json")
# Default difficulty levels to generate (target joint-feasible fractions).
DEFAULT_PERCENTAGES = (0.20, 0.10, 0.05, 0.01)


@dataclass
class CalibrationResult:
    problem: str
    target_fraction: float
    achieved_fraction: float
    thresholds: list[float]
    maxima: list[float]
    minima: list[float]
    quantile_level: float
    margin_below_max: list[float]
    n_samples: int
    seed: int
    method: str
    created: str

    def to_record(self) -> dict:
        return asdict(self)


def _pct_key(p: float) -> str:
    """Stable string key for a percentage, e.g. 0.1 -> '0.1', 0.05 -> '0.05'."""
    return f"{p:.6g}"


def _joint_fraction(Y: torch.Tensor, tau: torch.Tensor) -> float:
    return float((Y >= tau).all(dim=-1).to(torch.double).mean().item())


def calibrate_thresholds(
    problem: Problem,
    target_fraction: float = 0.10,
    n_samples: int = 200_000,
    seed: int = 0,
    max_iter: int = 64,
) -> CalibrationResult:
    """Find tau so that ~`target_fraction` of uniform samples are jointly feasible.

    Uses the noiseless objective (``problem.fn``) so the constraints describe the
    true feasible region S, independent of observation noise.
    """
    if not 0.0 < target_fraction < 1.0:
        raise ValueError(f"target_fraction must be in (0, 1); got {target_fraction}")

    X = problem.sample_uniform(n_samples, seed=seed).to(torch.double)
    Y = problem.fn(X).to(torch.double)  # (N, m), noiseless
    if Y.dim() != 2:
        raise ValueError(f"problem.fn must return (N, m); got shape {tuple(Y.shape)}")

    maxima = Y.max(dim=0).values
    minima = Y.min(dim=0).values

    def frac_for_q(q: float) -> tuple[float, torch.Tensor]:
        # Per-objective marginal quantile -> coupled, scale-invariant thresholds.
        tau = torch.quantile(Y, q, dim=0)
        return _joint_fraction(Y, tau), tau

    # Joint fraction is non-increasing in q: q=0 -> tau=min -> fraction 1;
    # q=1 -> tau=max -> fraction ~ 0. Bisect for fraction == target.
    lo, hi = 0.0, 1.0
    q = 0.5
    _, tau = frac_for_q(q)
    for _ in range(max_iter):
        q = 0.5 * (lo + hi)
        frac, tau = frac_for_q(q)
        if frac > target_fraction:
            lo = q  # too many feasible -> raise thresholds (increase q)
        else:
            hi = q  # too few feasible -> lower thresholds (decrease q)

    achieved = _joint_fraction(Y, tau)
    return CalibrationResult(
        problem=problem.name,
        target_fraction=float(target_fraction),
        achieved_fraction=float(achieved),
        thresholds=[float(v) for v in tau.tolist()],
        maxima=[float(v) for v in maxima.tolist()],
        minima=[float(v) for v in minima.tolist()],
        quantile_level=float(q),
        margin_below_max=[float(m - t) for m, t in zip(maxima.tolist(), tau.tolist())],
        n_samples=int(n_samples),
        seed=int(seed),
        method="equal-marginal-quantile-bisection",
        created=_dt.datetime.now().isoformat(timespec="seconds"),
    )


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def load_threshold_store(path: str = DEFAULT_THRESHOLDS_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_calibration(
    result: CalibrationResult, path: str = DEFAULT_THRESHOLDS_PATH
) -> None:
    """Merge one calibration into the JSON store keyed by [problem][pct]."""
    store = load_threshold_store(path)
    store.setdefault(result.problem, {})[_pct_key(result.target_fraction)] = (
        result.to_record()
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(store, f, indent=2, sort_keys=True)


def load_thresholds(
    problem_name: str,
    percentage: float = 0.10,
    path: str = DEFAULT_THRESHOLDS_PATH,
) -> torch.Tensor:
    """Load calibrated thresholds for (problem, percentage) as a tensor.

    Raises KeyError with a helpful message if the entry is missing.
    """
    store = load_threshold_store(path)
    key = _pct_key(percentage)
    if problem_name not in store or key not in store[problem_name]:
        avail = {p: list(d.keys()) for p, d in store.items()}
        raise KeyError(
            f"No calibrated thresholds for problem='{problem_name}', "
            f"percentage={key} in '{path}'. Available: {avail}. "
            f"Run: python -m itcas.pipeline.thresholds --percentages {percentage}"
        )
    return torch.tensor(store[problem_name][key]["thresholds"], dtype=torch.double)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _calibrate_problems(
    problem_names: Sequence[str],
    percentages: Sequence[float],
    n_samples: int,
    seed: int,
    path: str,
) -> list[CalibrationResult]:
    results: list[CalibrationResult] = []
    for name in problem_names:
        problem = REGISTRY[name]()
        for p in percentages:
            res = calibrate_thresholds(
                problem, target_fraction=p, n_samples=n_samples, seed=seed
            )
            save_calibration(res, path=path)
            results.append(res)
            print(
                f"[{name:22s}] p={_pct_key(p):>5s} -> "
                f"achieved={res.achieved_fraction*100:6.3f}%  "
                f"tau={[round(t, 4) for t in res.thresholds]}"
            )
    return results


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser("itcas.pipeline.thresholds")
    parser.add_argument(
        "--problems", nargs="*", default=None,
        help="Benchmark names (default: the four synthetic benchmarks).",
    )
    parser.add_argument(
        "--percentages", nargs="*", type=float, default=list(DEFAULT_PERCENTAGES),
        help="Target joint-feasible fractions (difficulty levels). Default 10%% set.",
    )
    parser.add_argument("--n-samples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=DEFAULT_THRESHOLDS_PATH)
    args = parser.parse_args(argv)

    default_problems = [
        "sphere2_6d", "rosenbrock_sphere_6d", "multimodal_trap_20d", "dtlz2_6d",
    ]
    problem_names = args.problems or default_problems
    unknown = [p for p in problem_names if p not in REGISTRY]
    if unknown:
        raise SystemExit(f"Unknown problem(s): {unknown}. Choices: {list(REGISTRY)}")

    _calibrate_problems(
        problem_names, args.percentages, args.n_samples, args.seed, args.out
    )
    print(f"\nSaved calibrated thresholds to: {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
