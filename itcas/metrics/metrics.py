"""Evaluation metrics per `contexts/metrics.md`.

The C-MO-CAS evaluation suite consists of exactly five metrics:

1. **Context Fill Distance (CFD)** — how evenly the *queried* contexts cover the
   context space.
2. **Feasible Context Fill Distance (FCFD)** — how evenly the contexts in which
   feasible designs were *found* cover the context space.
3. **Feasible Convex Hull Volume (FCHV)** — objective-space macro-spread of
    discovered feasible vectors.
4. **ε-Archive Size** — cardinality of a greedy ε-net over discovered feasible
    objective vectors (gridless micro-diversity).
5. **Number of Positives** — cumulative count of feasible ``(x, c)`` queries.
6. **Area Under the Positives Curve (AUP)** — ``sum_t P(t)``; a single number.

A "positive"/feasible sample satisfies ``f_i(x, c) >= tau_i`` for every objective
``i``. Fill distances are *lower-is-better*; FCHV and ε-Archive Size are
*higher-is-better*.
"""
from __future__ import annotations

import torch


def is_feasible(y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """y: (..., m), h: (m,) -> bool tensor of shape (...)."""
    return (y >= h).all(dim=-1)


def positive_samples(Y: torch.Tensor, h: torch.Tensor) -> int:
    """Number of Positives: count of feasible observations."""
    if Y.numel() == 0:
        return 0
    return int(is_feasible(Y, h).sum().item())


def cumulative_positives(per_iter_feasible: list[bool]) -> list[int]:
    """P(t): cumulative number of positives after each query."""
    out, c = [], 0
    for f in per_iter_feasible:
        c += int(bool(f))
        out.append(c)
    return out


def aup(per_iter_feasible: list[bool]) -> int:
    """Area Under the Positives Curve = ``sum_{t=1}^T P(t)``.

    A single scalar summarising how quickly positives accumulate (rewards
    finding feasible samples early). Reported as a number, never plotted.
    """
    return int(sum(cumulative_positives(per_iter_feasible)))


def _fill_distance(
    samples: torch.Tensor, reference: torch.Tensor, *, penalty: float
) -> float:
    """Radius of the largest empty sphere within ``reference`` w.r.t. ``samples``.

    ``max_{r in reference} min_{s in samples} ||r - s||_2``. Returns ``penalty``
    when ``samples`` is empty (nothing has been placed yet) and ``0.0`` when the
    reference set itself is empty.
    """
    if reference.numel() == 0:
        return 0.0
    if samples.numel() == 0:
        return float(penalty)
    d = torch.cdist(reference, samples)
    return float(d.min(dim=-1).values.max().item())


def context_fill_distance(
    eval_contexts: torch.Tensor, ref_contexts: torch.Tensor, *, penalty: float
) -> float:
    """CFD: ``max_{c in C_ref} min_{c_t in C_evaluated} ||c - c_t||_2``.

    ``eval_contexts`` are the context columns of every evaluated query.
    """
    return _fill_distance(eval_contexts, ref_contexts, penalty=penalty)


def feasible_context_fill_distance(
    feasible_contexts: torch.Tensor, ref_contexts: torch.Tensor, *, penalty: float
) -> float:
    """FCFD: same as CFD but over the unique contexts where feasible designs were
    found. ``penalty`` (e.g. the context-space diagonal) is returned when no
    positives have been found yet.
    """
    return _fill_distance(feasible_contexts, ref_contexts, penalty=penalty)


def feasible_convex_hull_volume(
    feasible_Y: torch.Tensor,
) -> float:
    """FCHV: volume of the convex hull of feasible objective vectors.

    Returns ``0.0`` when there are fewer than ``m + 1`` feasible points or
    when the discovered points are lower-dimensional/degenerate.
    """
    if feasible_Y.numel() == 0:
        return 0.0
    Y = feasible_Y.detach().double()
    if Y.ndim != 2:
        raise ValueError(f"feasible_Y must be 2D, got shape {tuple(Y.shape)}")
    n, m = int(Y.shape[0]), int(Y.shape[1])
    if n < m + 1:
        return 0.0
    if m == 1:
        return float((Y.max() - Y.min()).item())
    centred = Y - Y.mean(dim=0, keepdim=True)
    if int(torch.linalg.matrix_rank(centred).item()) < m:
        return 0.0
    try:
        from scipy.spatial import ConvexHull
    except ImportError as exc:  # pragma: no cover - environment issue
        raise ImportError("scipy is required for feasible_convex_hull_volume") from exc
    try:
        hull = ConvexHull(Y.cpu().numpy())
    except Exception:
        return 0.0
    return float(hull.volume)


def transform_feasible_for_archive(
    feasible_Y: torch.Tensor, thresholds: torch.Tensor
) -> torch.Tensor:
    """``y' = log1p(y - tau)`` per `contexts/metrics.md` §4.

    This is the *same* transform ``itcas.reporting.tune_eps_archive`` applies
    (offline) to the pooled feasible set before calibrating ``eps`` as a
    percentile of pairwise distances. Any runtime consumer of that calibrated
    ``eps`` must build its ε-net in this same transformed space, or the
    threshold is compared against the wrong scale (see git history for the
    bug this fixes: raw-space archives silently collapsed onto the
    Number-of-Positives count whenever ``y - tau`` was large).

    ``feasible_Y`` is assumed to already be filtered to strictly feasible rows
    (``is_feasible(feasible_Y, thresholds)`` all True), so ``y - tau >= 0``
    elementwise and ``log1p`` is always defined.
    """
    return torch.log1p(feasible_Y.detach().double() - thresholds.detach().double())


def epsilon_archive_size(
    feasible_Y: torch.Tensor,
    *,
    thresholds: torch.Tensor,
    eps: float = 0.05,
) -> int:
    """ε-Archive Size: cardinality of a greedy ε-net over feasible objective vectors.

    Per `contexts/metrics.md` §4, the ε-net is built in *log-transformed*
    space: ``y' = log1p(y - thresholds)``, exactly mirroring the offline
    calibration in ``itcas.reporting.tune_eps_archive`` that produced ``eps``
    in the first place (a raw-space ``eps`` comparison here would silently
    collapse this metric onto Number-of-Positives whenever the feasible
    margin ``y - thresholds`` is large).

    Objective vectors are processed in order. A point is added to the archive
    only if its transform lies at least ``eps`` away (Euclidean) from every
    existing archive member's transform. Returns the final archive size — a
    strictly monotone, boundary-free micro-diversity count.
    """
    if feasible_Y.numel() == 0:
        return 0
    if feasible_Y.ndim != 2:
        raise ValueError(f"feasible_Y must be 2D, got shape {tuple(feasible_Y.shape)}")
    Y = transform_feasible_for_archive(feasible_Y, thresholds)
    N = int(Y.shape[0])
    archive: list[torch.Tensor] = []
    for k in range(N):
        y = Y[k]
        if not archive:
            archive.append(y)
        else:
            arch = torch.stack(archive)  # (|archive|, m)
            dists = torch.linalg.norm(arch - y.unsqueeze(0), dim=-1)
            if float(dists.min().item()) >= eps:
                archive.append(y)
    return len(archive)
