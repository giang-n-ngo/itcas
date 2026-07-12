#!/usr/bin/env python3
"""Analyze A1_hard historical data against FF input/context bounds and constraints.

Computes:
1) Set A: points within full FF bounds (design + context)
2) Set B: points in A that satisfy historical filter thresholds
3) Per-constraint satisfaction counts on B
4) Simultaneous all-constraint satisfaction count on B

With --scan, sweeps all combinations of constraint thresholds and writes a
single sorted output (smaller threshold sums first).

Usage:
    python scripts/analyze_ff_initial_data.py
    python scripts/analyze_ff_initial_data.py --scan
    python scripts/analyze_ff_initial_data.py --smartsat-root /path/to/SmartSat
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Allow direct execution via: python scripts/analyze_ff_initial_data.py
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from itcas.pipeline.problems import (
    _DT_HI,
    _DT_LO,
    _FF_BOUNDS,
    _FUEL_THRESHOLD_G,
    _HIST_FUEL_FILTER_KG,
    _HIST_RMSE_FILTER_M,
    _PEAK_THRUST_PENALTY,
    _PEAK_THRUST_THRESHOLD,
    _Q_LOG_HI,
    _Q_LOG_LO,
    _R_LOG_HI,
    _R_LOG_LO,
    _RMSE_THRESHOLD,
    _resolve_smartsat_root,
    _transform_fuel_g,
    _transform_rmse,
)

_DEFAULT_OUT_JSON = (
    f"results/ff_initial_data/"
    f"ff_initial_data_{_RMSE_THRESHOLD}_{_FUEL_THRESHOLD_G}_{_PEAK_THRUST_THRESHOLD}.json"
)
_DEFAULT_SCAN_OUT_JSON = "results/ff_initial_data/ff_initial_data_scan.json"

# Default scan grids (user-specified values).
_SCAN_RMSE_M    = [50.0, 100.0, 150.0, 200.0]
_SCAN_FUEL_G    = [50.0, 100.0, 150.0, 200.0]
_SCAN_THRUST_N  = [1.0, 2.0, 5.0, 10.0]


# ---------------------------------------------------------------------------
# Precomputed sets
# ---------------------------------------------------------------------------

@dataclass
class _PrecomputedB:
    """Set A / Set B counts and per-run transformed metrics for Set B.

    Stored once so that many threshold combinations can be evaluated cheaply.
    """
    results_dir: str
    rmse_filter_m: float
    fuel_filter_kg: float
    runs_total: int
    runs_missing_or_bad_params: int
    set_A_in_bounds: int
    # Per-run pre-transformed metrics for every point in Set B.
    b_transformed_rmse: list[float] = field(default_factory=list)
    b_transformed_fuel_g: list[float] = field(default_factory=list)
    b_peak_thrust: list[float] = field(default_factory=list)

    @property
    def set_B_after_filters(self) -> int:
        return len(self.b_transformed_rmse)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_param_sets(smartsat_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    std_settings_path = (
        smartsat_root
        / "scarlet-gamma-deakin-dev"
        / "settings"
        / "settings_A1.json"
    )
    opt_settings_dir = (
        smartsat_root
        / "scarlet-gamma-deakin-dev"
        / "settings"
        / "optimized_hyperparameters"
        / "A1"
        / "settings"
    )

    std_sets: dict[str, dict[str, Any]] = {}
    if std_settings_path.exists():
        with open(std_settings_path) as f:
            data = json.load(f)
        for item in data.get("hyperparameter_sets", []):
            name = str(item.get("name"))
            params = item.get("parameters")
            if isinstance(params, dict):
                std_sets[name] = params

    opt_sets: dict[str, dict[str, Any]] = {}
    if opt_settings_dir.exists():
        for p in sorted(opt_settings_dir.glob("*.json")):
            try:
                with open(p) as f:
                    params = json.load(f)
                if isinstance(params, dict):
                    opt_sets[p.stem] = params
            except Exception:
                continue

    return std_sets, opt_sets


def _design_in_bounds(params: dict[str, Any]) -> bool:
    q_lo, q_hi = 10.0 ** _Q_LOG_LO, 10.0 ** _Q_LOG_HI
    r_lo, r_hi = 10.0 ** _R_LOG_LO, 10.0 ** _R_LOG_HI

    dt = float(params.get("control_update_interval", float("nan")))
    if not math.isfinite(dt) or not (_DT_LO <= dt <= _DT_HI):
        return False

    q = params.get("Q_weight")
    if isinstance(q, list):
        if len(q) != 6 or any((not math.isfinite(float(v)) or not (q_lo <= float(v) <= q_hi)) for v in q):
            return False
    elif isinstance(q, (int, float)):
        qv = float(q)
        if not math.isfinite(qv) or not (q_lo <= qv <= q_hi):
            return False
    else:
        return False

    r = params.get("R_weight")
    if isinstance(r, list):
        if len(r) != 3 or any((not math.isfinite(float(v)) or not (r_lo <= float(v) <= r_hi)) for v in r):
            return False
    elif isinstance(r, (int, float)):
        rv = float(r)
        if not math.isfinite(rv) or not (r_lo <= rv <= r_hi):
            return False
    else:
        return False

    return True


def _context_in_bounds(pos: list[Any], vel: list[Any]) -> bool:
    if len(pos) != 3 or len(vel) != 3:
        return False

    lo = _FF_BOUNDS[0].tolist()
    hi = _FF_BOUNDS[1].tolist()

    try:
        p = [float(v) for v in pos]
        v = [float(vv) for vv in vel]
    except Exception:
        return False

    for i in range(3):
        if not (lo[10 + i] <= p[i] <= hi[10 + i]):
            return False
    for i in range(3):
        if not (lo[13 + i] <= v[i] <= hi[13 + i]):
            return False
    return True


def _passes_filter_thresholds(run: dict[str, Any], rmse_filter_m: float, fuel_filter_kg: float) -> bool:
    rmse = run.get("rmse_position")
    fuel = run.get("fuel_consumption")
    if rmse is None or fuel is None:
        return False
    try:
        rmse_f = float(rmse)
        fuel_f = float(fuel)
    except Exception:
        return False
    if not math.isfinite(rmse_f) or not math.isfinite(fuel_f):
        return False
    return (rmse_f < rmse_filter_m) and (fuel_f < fuel_filter_kg)


# ---------------------------------------------------------------------------
# Core data scan (runs once)
# ---------------------------------------------------------------------------

def _precompute_sets(smartsat_root: Path, rmse_filter_m: float, fuel_filter_kg: float) -> _PrecomputedB:
    """Scan A1_hard run files and build Set A / Set B, storing pre-transformed
    metrics for every B-point so threshold combinations can be evaluated fast."""
    results_dir = smartsat_root / "scarlet-gamma-deakin-dev" / "results" / "A1_hard"
    if not results_dir.exists():
        raise FileNotFoundError(f"A1_hard results directory not found: {results_dir}")

    std_sets, opt_sets = _load_param_sets(smartsat_root)

    def resolve_params(setting_dir: Path, sid: str) -> Optional[dict[str, Any]]:
        first_run = next(iter(sorted(setting_dir.glob("run_*.json"))), None)
        hp_id = None
        if first_run is not None:
            try:
                with open(first_run) as f:
                    rr = json.load(f)
                hp_raw = rr.get("hyperparameter_set")
                if hp_raw is not None:
                    hp_id = str(hp_raw)
            except Exception:
                hp_id = None
        if hp_id is not None:
            params = std_sets.get(hp_id) or opt_sets.get(hp_id)
            if params is not None:
                return params
        return std_sets.get(sid) or opt_sets.get(sid)

    pre = _PrecomputedB(
        results_dir=str(results_dir),
        rmse_filter_m=rmse_filter_m,
        fuel_filter_kg=fuel_filter_kg,
        runs_total=0,
        runs_missing_or_bad_params=0,
        set_A_in_bounds=0,
    )

    for setting_dir in sorted(results_dir.glob("setting_*")):
        if not setting_dir.is_dir():
            continue

        sid = setting_dir.name.replace("setting_", "")
        params = resolve_params(setting_dir, sid)
        if params is None:
            for _ in setting_dir.glob("run_*.json"):
                pre.runs_total += 1
                pre.runs_missing_or_bad_params += 1
            continue

        if not _design_in_bounds(params):
            for _ in setting_dir.glob("run_*.json"):
                pre.runs_total += 1
            continue

        for run_path in sorted(setting_dir.glob("run_*.json")):
            pre.runs_total += 1
            try:
                with open(run_path) as f:
                    run = json.load(f)
            except Exception:
                continue

            pos = run.get("initial_position")
            vel = run.get("initial_velocity")
            if not isinstance(pos, list) or not isinstance(vel, list):
                continue
            if not _context_in_bounds(pos, vel):
                continue

            pre.set_A_in_bounds += 1

            if not _passes_filter_thresholds(run, rmse_filter_m=rmse_filter_m, fuel_filter_kg=fuel_filter_kg):
                continue

            # Pre-transform metrics so threshold sweeps are O(1) per point.
            rmse_raw = run.get("rmse_position")
            fuel_kg_raw = run.get("fuel_consumption")
            thrust_raw = run.get("peak_thrust_command")

            fuel_g = (
                float(fuel_kg_raw) * 1000.0
                if fuel_kg_raw is not None and math.isfinite(float(fuel_kg_raw))
                else None
            )
            thrust_n = (
                float(thrust_raw)
                if thrust_raw is not None and math.isfinite(float(thrust_raw))
                else None
            )

            pre.b_transformed_rmse.append(_transform_rmse(rmse_raw))
            pre.b_transformed_fuel_g.append(_transform_fuel_g(fuel_g))
            pre.b_peak_thrust.append(thrust_n if thrust_n is not None else _PEAK_THRUST_PENALTY)

    return pre


# ---------------------------------------------------------------------------
# Per-threshold counting (fast inner loop)
# ---------------------------------------------------------------------------

def _count_for_thresholds(
    pre: _PrecomputedB,
    rmse_t: float,
    fuel_g_t: float,
    thrust_t: float,
) -> dict[str, int]:
    c1 = c2 = c3 = c_all = 0
    for tr, tf, tt in zip(pre.b_transformed_rmse, pre.b_transformed_fuel_g, pre.b_peak_thrust):
        s0 = tr <= rmse_t
        s1 = tf <= fuel_g_t
        s2 = tt <= thrust_t
        if s0:
            c1 += 1
        if s1:
            c2 += 1
        if s2:
            c3 += 1
        if s0 and s1 and s2:
            c_all += 1
    return {
        "B_constraint_1_rmse": c1,
        "B_constraint_2_fuel": c2,
        "B_constraint_3_peak_thrust": c3,
        "B_all_constraints": c_all,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_a1_hard(smartsat_root: Path, rmse_filter_m: float, fuel_filter_kg: float) -> dict[str, Any]:
    """Single-threshold analysis (backward-compatible entry point)."""
    pre = _precompute_sets(smartsat_root, rmse_filter_m, fuel_filter_kg)
    counts = _count_for_thresholds(pre, _RMSE_THRESHOLD, _FUEL_THRESHOLD_G, _PEAK_THRUST_THRESHOLD)
    return {
        "results_dir": pre.results_dir,
        "filter_thresholds": {
            "rmse_position_lt_m": rmse_filter_m,
            "fuel_consumption_lt_kg": fuel_filter_kg,
        },
        "counts": {
            "runs_total": pre.runs_total,
            "runs_missing_or_bad_params": pre.runs_missing_or_bad_params,
            "set_A_in_bounds": pre.set_A_in_bounds,
            "set_B_after_filters": pre.set_B_after_filters,
            **counts,
        },
    }


def scan_thresholds(
    smartsat_root: Path,
    rmse_filter_m: float,
    fuel_filter_kg: float,
    rmse_values: list[float],
    fuel_g_values: list[float],
    thrust_values: list[float],
) -> dict[str, Any]:
    """Sweep all (rmse, fuel, thrust) threshold combinations.

    Set A and Set B are computed once from the filter thresholds; only the
    constraint-satisfaction counts are re-evaluated per combination.
    Combinations are returned sorted by ascending sum of threshold values.
    """
    pre = _precompute_sets(smartsat_root, rmse_filter_m, fuel_filter_kg)

    combos = sorted(
        itertools.product(rmse_values, fuel_g_values, thrust_values),
        key=lambda c: c[0] + c[1] + c[2],
    )

    scan_results = []
    for rmse_t, fuel_g_t, thrust_t in combos:
        counts = _count_for_thresholds(pre, rmse_t, fuel_g_t, thrust_t)
        scan_results.append({
            "rmse_threshold_m": rmse_t,
            "fuel_threshold_g": fuel_g_t,
            "peak_thrust_threshold_n": thrust_t,
            "threshold_sum": rmse_t + fuel_g_t + thrust_t,
            **counts,
        })

    return {
        "results_dir": pre.results_dir,
        "filter_thresholds": {
            "rmse_position_lt_m": rmse_filter_m,
            "fuel_consumption_lt_kg": fuel_filter_kg,
        },
        "shared": {
            "runs_total": pre.runs_total,
            "runs_missing_or_bad_params": pre.runs_missing_or_bad_params,
            "set_A_in_bounds": pre.set_A_in_bounds,
            "set_B_after_filters": pre.set_B_after_filters,
        },
        "threshold_scan": scan_results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_floats(s: str) -> list[float]:
    return [float(v) for v in s.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smartsat-root", type=str, default=None)
    parser.add_argument(
        "--rmse-filter-m", type=float, default=_HIST_RMSE_FILTER_M,
        help="Historical RMSE filter threshold (meters) for Set B.",
    )
    parser.add_argument(
        "--fuel-filter-kg", type=float, default=_HIST_FUEL_FILTER_KG,
        help="Historical fuel filter threshold (kg) for Set B.",
    )
    parser.add_argument(
        "--out-json", type=str, default=None,
        help="Output path. Defaults depend on --scan.",
    )
    parser.add_argument(
        "--scan", action="store_true",
        help="Sweep all threshold combinations instead of a single evaluation.",
    )
    parser.add_argument(
        "--rmse-thresholds", type=str,
        default=",".join(str(v) for v in _SCAN_RMSE_M),
        help="Comma-separated RMSE constraint thresholds in meters (scan mode).",
    )
    parser.add_argument(
        "--fuel-thresholds-g", type=str,
        default=",".join(str(v) for v in _SCAN_FUEL_G),
        help="Comma-separated fuel constraint thresholds in grams (scan mode).",
    )
    parser.add_argument(
        "--thrust-thresholds", type=str,
        default=",".join(str(v) for v in _SCAN_THRUST_N),
        help="Comma-separated peak-thrust constraint thresholds in Newtons (scan mode).",
    )
    args = parser.parse_args()

    smartsat_root = _resolve_smartsat_root(args.smartsat_root)

    if args.scan:
        out_json = args.out_json or _DEFAULT_SCAN_OUT_JSON
        result = scan_thresholds(
            smartsat_root=smartsat_root,
            rmse_filter_m=float(args.rmse_filter_m),
            fuel_filter_kg=float(args.fuel_filter_kg),
            rmse_values=_parse_floats(args.rmse_thresholds),
            fuel_g_values=_parse_floats(args.fuel_thresholds_g),
            thrust_values=_parse_floats(args.thrust_thresholds),
        )
    else:
        out_json = args.out_json or _DEFAULT_OUT_JSON
        result = analyze_a1_hard(
            smartsat_root=smartsat_root,
            rmse_filter_m=float(args.rmse_filter_m),
            fuel_filter_kg=float(args.fuel_filter_kg),
        )

    text = json.dumps(result, indent=2)
    print(text)

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n")
    print(f"[ff_init_data] written to: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
