"""Per-metric extractors operating on a per-run ``RunSeries``.

Each extractor returns a list of values aligned with ``RunSeries.x_steps`` and
``RunSeries.x_evals`` (one value per cumulative state, including the initial
dataset as the first point).

The plotted metrics are (see ``contexts/metrics.md``):

* ``cumulative_positives`` — Number of Positives ``P(t)``;
* ``feasible_context_fill_distance`` (FCFD);
* ``feasible_convex_hull_volume`` (FCHV);
* ``epsilon_archive_size`` (ε-Archive Size).

AUP is a single number (``sum_t P(t)``) and is deliberately *not* a curve, so it
is not part of the plotting registry.
"""
from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from ..metrics import (
    epsilon_archive_size,
    feasible_context_fill_distance,
    feasible_convex_hull_volume,
    is_feasible,
    transform_feasible_for_archive,
)
from ..metrics.reference import ReferenceData, build_reference_data
from ..pipeline.problems import REGISTRY as PROBLEM_REGISTRY


@dataclass
class RunSeries:
    """Per-run cumulative state arrays.

    All lists have the same length ``T+1`` where ``T`` is the number of
    iterations actually executed: index 0 is the post-init state, index ``t``
    after iteration ``t``. ``X_per_step[t]`` and ``Y_per_step[t]`` are tensors
    of all evaluated points up to and including step ``t``.
    """

    problem: str
    method: str
    run_name: str
    seed: int
    thresholds: torch.Tensor
    context_dims: tuple[int, ...]
    config: dict
    x_evals: list[int]
    x_steps: list[int]
    feasible_per_step: list[list[bool]]
    X_per_step: list[torch.Tensor]
    Y_per_step: list[torch.Tensor]


# ---------------------------------------------------------------------------
# Number of Positives (cumulative)
# ---------------------------------------------------------------------------
def _cum_pos(run: RunSeries) -> list[float]:
    out, c = [], 0
    for flags in run.feasible_per_step:
        c += sum(bool(f) for f in flags)
        out.append(float(c))
    return out


def cumulative_positives(run: RunSeries) -> list[float]:
    return _cum_pos(run)


# ---------------------------------------------------------------------------
# Fill-distance curves (need problem-specific reference sets)
# ---------------------------------------------------------------------------
def build_reference(run: RunSeries, *, n_context: int = 4000) -> Optional[ReferenceData]:
    """Reconstruct the run's problem and build its context reference set.

    Returns ``None`` when the problem is not in the registry.
    """
    factory = PROBLEM_REGISTRY.get(run.problem)
    if factory is None:
        return None
    try:
        problem = factory(thresholds=run.thresholds.tolist())
    except TypeError:
        problem = factory()
    return build_reference_data(
        problem,
        n_context=n_context,
        seed=int(run.seed),
    )


def feasible_context_fill_distance_curve(run: RunSeries, ref: ReferenceData) -> Optional[list[float]]:
    """FCFD curve — fully vectorised.

    Feasibility is immutable once a point is evaluated (Y values are fixed).
    We precompute the feasibility mask from the final Y, build one
    ``cdist(ctx_ref, feas_ctx_final)`` call, and take its column-wise
    ``cummin``. Per-step indices are resolved with ``bisect`` (pure Python,
    no PyTorch per step); values are gathered in one shot.
    """
    cdims = list(run.context_dims)
    if not cdims or ref.context_ref is None:
        return None
    ctx_ref = ref.context_ref
    penalty = ref.context_penalty
    T = len(run.X_per_step)

    X_final = Y_final = None
    for i in range(len(run.X_per_step) - 1, -1, -1):
        X_c, Y_c = run.X_per_step[i], run.Y_per_step[i]
        if X_c.numel() > 0 and Y_c.numel() > 0:
            X_final, Y_final = X_c, Y_c
            break
    if X_final is None:
        return [penalty] * T

    feas_mask_all = is_feasible(Y_final, run.thresholds)  # (N_total,)
    feas_indices_t = feas_mask_all.nonzero(as_tuple=True)[0]  # sorted
    if feas_indices_t.numel() == 0:
        return [penalty] * T

    feas_idx_list = feas_indices_t.tolist()
    feas_ctx_all = X_final[feas_indices_t][:, cdims]          # (N_feas, d_c)
    D_feas = torch.cdist(ctx_ref, feas_ctx_all)               # (n_ref, N_feas)
    D_feas_cummin = torch.cummin(D_feas, dim=1).values        # (n_ref, N_feas)

    # k_per_step[t] = number of feasible points seen through step t.
    k_per_step: list[int] = []
    for i, X in enumerate(run.X_per_step):
        Y = run.Y_per_step[i]
        if X.numel() == 0 or Y.numel() == 0:
            k_per_step.append(0)
        else:
            k_per_step.append(bisect.bisect_left(feas_idx_list, X.shape[0]))

    valid = [(t, k - 1) for t, k in enumerate(k_per_step) if k > 0]
    out = [penalty] * T
    if valid:
        ts, col_idx = zip(*valid)
        idx = torch.tensor(col_idx, dtype=torch.long)
        fd_vals = D_feas_cummin[:, idx].max(dim=0).values.tolist()
        for t, fd in zip(ts, fd_vals):
            out[t] = fd
    return out


def feasible_convex_hull_volume_curve(run: RunSeries) -> list[float]:
    """FCHV curve with skip-if-unchanged optimisation.

    ConvexHull is only recomputed when the number of feasible points grows.
    Steps where no new feasible point was found reuse the previous value —
    this avoids O(T) hull computations and reduces to O(n_feas_final) calls.
    """
    out: list[float] = []
    prev_feas_n = -1
    cached_val = 0.0
    for Y in run.Y_per_step:
        if Y.numel() == 0:
            out.append(0.0)
            prev_feas_n = 0
            cached_val = 0.0
            continue
        mask = is_feasible(Y, run.thresholds)
        disc = Y[mask] if bool(mask.any()) else Y[:0]
        n = int(disc.shape[0])
        if n != prev_feas_n:
            cached_val = feasible_convex_hull_volume(disc)
            prev_feas_n = n
        out.append(cached_val)
    return out


_EXPERIMENTS_PATH = Path(__file__).parent.parent.parent / "configs" / "experiments.json"


def _diff_key_candidates(threshold_pct) -> list[str]:
    """Candidate ``configs/experiments.json`` difficulty-key strings for ``threshold_pct``.

    ``threshold_pct`` reaches this function as a Python float (parsed back out
    of a run's own logged config), but the config's own keys are plain JSON
    string literals written by hand/by the calibration tooling in two
    different styles depending on the problem: bare integers for FF/CASD's
    own difficulty scales (``"1"``..``"10"``, ``"1"``..``"4"``) vs. decimal
    fractions for every "standard" problem's shared threshold_pct scale
    (``"0.01"``, ``"0.05"``, ``"0.1"``, ``"0.2"``). ``str(threshold_pct)``
    alone only ever produces the decimal style (``str(2.0) == "2.0"``, never
    matching a bare ``"2"`` key), so FF/CASD's per-difficulty overrides were
    silently invisible to this lookup -- always falling through to the
    problem-level default -- until this was fixed. Returns candidates in
    lookup-preference order: the bare-integer form first when
    ``threshold_pct`` is a whole number (matching FF/CASD's style), then the
    plain ``str()`` form (matching every other problem's style) so a config
    that genuinely uses a decimal key for a whole-number-valued level (not
    currently the case anywhere, but not excluded either) still resolves.
    """
    candidates = []
    try:
        f = float(threshold_pct)
    except (TypeError, ValueError):
        f = None
    if f is not None and f.is_integer():
        candidates.append(str(int(f)))
    candidates.append(str(threshold_pct))
    return candidates


def _eps_from_experiments(problem: str, threshold_pct: Optional[str]) -> Optional[float]:
    """Return eps_archive for (problem, difficulty) from configs/experiments.json.

    Resolution order mirrors the shell launcher:
      problems[problem][threshold_pct] -> problems[problem].defaults -> defaults

    ``threshold_pct`` is matched against the config's difficulty keys via
    :func:`_diff_key_candidates`, which tries both the bare-integer style
    (FF/CASD) and the plain ``str()`` decimal style (every other problem) --
    see that function's docstring for why a single ``str(threshold_pct)``
    lookup silently missed FF/CASD's per-difficulty overrides.

    Returns None if the file is missing or the problem is not listed.
    """
    try:
        data = json.loads(_EXPERIMENTS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    top_default = data.get("defaults", {}).get("eps_archive")
    prob = data.get("problems", {}).get(problem)
    if prob is None:
        return top_default
    prob_default = prob.get("defaults", {}).get("eps_archive")
    if threshold_pct is not None:
        for diff_key in _diff_key_candidates(threshold_pct):
            diff_eps = prob.get(diff_key, {}).get("eps_archive")
            if diff_eps is not None:
                return float(diff_eps)
    if prob_default is not None:
        return float(prob_default)
    return float(top_default) if top_default is not None else None


def eps_archive_used(run: RunSeries) -> float:
    """Resolve the ``eps`` value :func:`epsilon_archive_size_curve` uses for ``run``.

    Single source of truth for this resolution (``threshold_pct`` from
    ``run.config["extra"]`` -> :func:`_eps_from_experiments` ->
    ``run.config["eps_archive"]`` -> ``0.05``), so any caller that needs to
    know *which* eps a cached curve was computed under (to detect a stale
    on-disk cache after ``configs/experiments.json`` is recalibrated -- see
    ``itcas.reporting.summary._precompute_cached``) resolves it exactly the
    same way the curve itself was computed, instead of duplicating this logic.
    """
    threshold_pct = (run.config.get("extra") or {}).get("threshold_pct")
    return (
        _eps_from_experiments(run.problem, threshold_pct)
        or float(run.config.get("eps_archive", 0.05))
    )


def epsilon_archive_size_curve(run: RunSeries) -> list[float]:
    """ε-Archive Size curve.

    Feasible objective vectors are log-transformed (``y' = log1p(y -
    thresholds)``, matching ``itcas.reporting.tune_eps_archive``'s offline
    calibration space — see `contexts/metrics.md` §4) and processed in
    chronological order. A point is admitted to the archive only if its
    transform is at least ``eps`` away (Euclidean) from every existing archive
    member's transform, where ``eps`` is resolved by :func:`eps_archive_used`
    (``configs/experiments.json`` calibration, falling back to
    ``run.config["eps_archive"]`` then 0.05). The curve is the archive size
    at each algorithmic step.

    The archive is built once over the full feasible set (in insertion order)
    and per-step values are resolved with a single binary-search pass, so cost
    is O(N_feas²) in the worst case but avoids redundant recomputation per step.
    """
    eps: float = eps_archive_used(run)

    Y_final = run.Y_per_step[-1] if run.Y_per_step else None
    if Y_final is None or Y_final.numel() == 0:
        return [0.0] * len(run.Y_per_step)

    feas_mask_all = is_feasible(Y_final, run.thresholds)
    if not bool(feas_mask_all.any()):
        return [0.0] * len(run.Y_per_step)

    feas_indices_t = feas_mask_all.nonzero(as_tuple=True)[0]  # sorted
    feas_idx_list = feas_indices_t.tolist()
    disc_all = transform_feasible_for_archive(Y_final[feas_indices_t], run.thresholds)
    N_feas = int(disc_all.shape[0])

    # Build greedy ε-archive in chronological order (in transformed space).
    # archive_size_by_k[k] = archive size after the k-th feasible point is seen.
    archive_pts: list[torch.Tensor] = []
    archive_size_by_k: list[int] = [0]  # index 0 = zero feasible points seen
    for k in range(N_feas):
        y = disc_all[k]
        if not archive_pts:
            archive_pts.append(y)
        else:
            arch = torch.stack(archive_pts)  # (|archive|, m)
            dists = torch.linalg.norm(arch - y.unsqueeze(0), dim=-1)
            if float(dists.min().item()) >= eps:
                archive_pts.append(y)
        archive_size_by_k.append(len(archive_pts))

    out: list[float] = []
    for Y in run.Y_per_step:
        if Y.numel() == 0:
            out.append(0.0)
            continue
        n_t = Y.shape[0]
        k = bisect.bisect_left(feas_idx_list, n_t)
        out.append(float(archive_size_by_k[k]))
    return out


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str  # human-readable for axis label / file name
    needs_context: bool = False
    higher_is_better: bool = False  # if True, larger area = better performance


# Order is the plotting order; keys are stable filename slugs.
REGISTRY: dict[str, MetricSpec] = {
    "cumulative_positives": MetricSpec(
        "cumulative_positives", "Number of positives",
        higher_is_better=True,
    ),
    "feasible_context_fill_distance": MetricSpec(
        "feasible_context_fill_distance",
        "Feasible context fill distance (lower is better)",
        needs_context=True,
    ),
    "feasible_convex_hull_volume": MetricSpec(
        "feasible_convex_hull_volume", "Feasible convex hull volume (higher is better)",
        higher_is_better=True,
    ),
    "epsilon_archive_size": MetricSpec(
        "epsilon_archive_size", "ε-Archive Size (higher is better)",
        higher_is_better=True,
    ),
}


def compute_metric(
    run: RunSeries,
    spec: MetricSpec,
    *,
    ref: Optional[ReferenceData] = None,
) -> Optional[list[float]]:
    """Dispatch to the curve function for ``spec``. Returns ``None`` if a
    required input (e.g. the reference sets) is unavailable for this run."""
    key = spec.key
    if key == "cumulative_positives":
        return cumulative_positives(run)
    if key == "feasible_convex_hull_volume":
        return feasible_convex_hull_volume_curve(run)
    if key == "epsilon_archive_size":
        return epsilon_archive_size_curve(run)
    if ref is None:
        return None
    if key == "feasible_context_fill_distance":
        return feasible_context_fill_distance_curve(run, ref)
    raise KeyError(f"unknown metric '{key}'")
