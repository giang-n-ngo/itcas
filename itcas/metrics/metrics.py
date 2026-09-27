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

Additionally, **Localized (Context-Conditioned) Feasible Convex Hull Volume**
(`localized_feasible_convex_hull_volume`, `contexts/metrics.md` §6) partitions
the context space into ``K`` regions (via K-means on *all* evaluated contexts)
and averages the per-region FCHV over the nominal ``K`` regions, penalising
algorithms that concentrate objective-space diversity in a single context
region while leaving the rest of the context space unexplored.

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


# Nominal number of K-means context regions used as the shared default `k`
# for `localized_feasible_convex_hull_volume` by both
# `itcas.reporting.metrics.localized_feasible_convex_hull_volume_curve` (the
# plotted per-run curve) and `itcas.pipeline.loop.run_experiment`'s live
# final-value summary -- defined here, in the core stateless metrics module
# (rather than in `itcas.reporting.metrics`, where the curve wrapper lives),
# so both of those modules -- one of which (`itcas.pipeline.loop`) sits
# "below" `itcas.reporting` in the package's own import graph -- can import
# it without creating a cycle. There is no per-(problem, difficulty)
# calibration pipeline for this value analogous to `eps_archive`'s
# `tune_eps_archive` (see `itcas.reporting.metrics.eps_archive_used`), so a
# single fixed default is used everywhere: 8 regions is coarse enough to
# stay meaningful under realistic per-trial query budgets (tens to a few
# hundred points) even for the higher-context-dimensionality registered
# problems (e.g. `multimodal_trap_20d`, `dtlz4_12d`), while still fine
# enough to distinguish "spread across the context space" from
# "concentrated in one corner".
LOCALIZED_FCHV_K = 8


def localized_feasible_convex_hull_volume(
    feasible_Y: torch.Tensor,
    feasible_contexts: torch.Tensor,
    all_contexts: torch.Tensor,
    *,
    k: int,
    seed: int = 0,
) -> float:
    """Localized (context-conditioned) FCHV per `contexts/metrics.md` §6.

    Partitions the *context* space into ``k`` local regions via K-means
    (``scipy.cluster.vq.kmeans2``) fit on **every evaluated context**
    (``all_contexts``) -- not just the feasible subset -- mirroring how a
    predefined grid's bin edges would partition the whole context space
    independent of feasibility. Only afterwards are the strictly-feasible
    rows' contexts (``feasible_contexts``, already filtered by the caller to
    the same rows as ``feasible_Y``, i.e. row ``i`` of both corresponds to
    the same evaluated tuple) assigned to their nearest centroid
    (``scipy.cluster.vq.vq``) to group ``feasible_Y`` per region. Each
    region's convex-hull volume is computed with
    :func:`feasible_convex_hull_volume` (``0.0`` if that region has fewer
    than ``m + 1`` non-coplanar feasible points, exactly the global FCHV's
    rule, reused verbatim here).

    The final scalar is the mean over the *nominal* ``k`` regions
    (``sum_k V_k / k``): an empty or degenerate region contributes ``0`` to
    the sum but the denominator is always ``k``, never the count of
    non-empty regions -- an algorithm that only explores one context region
    cannot inflate this score by concentrating all its diversity there.

    Edge cases (each returns ``0.0`` rather than raising, since one trial's
    degenerate partition should never crash a whole batch's summary
    computation, mirroring :func:`feasible_convex_hull_volume`'s own
    ``try/except Exception: return 0.0`` around ``ConvexHull``):

    * ``k <= 0`` -> ``0.0`` (no regions requested).
    * ``all_contexts`` is empty (nothing evaluated yet) -> ``0.0``.
    * ``feasible_Y`` / ``feasible_contexts`` is empty (no feasible points to
      place) -> ``0.0``.
    * Fewer evaluated points than ``k`` -- ``kmeans2`` requires
      ``n_points >= n_clusters`` -- the *effective* cluster count used to fit
      the partition is clamped to ``min(k, n_eval)``, but the sum is still
      divided by the nominal ``k`` (the "missing" ``k - n_eval`` regions
      contribute ``0``, same as any other empty/degenerate region).
    * ``kmeans2``/``vq`` raising (e.g. malformed input) is caught and treated
      as a fully degenerate ("all regions empty") partition, i.e. ``0.0``.
      ``kmeans2``'s own *warning* (not exception) on an empty cluster during
      iteration is suppressed rather than propagated -- it does not indicate
      failure (the empty cluster's centroid simply keeps its previous
      position; see ``scipy.cluster.vq.kmeans2`` source), so surfacing it
      here would just be per-trial log noise across a large batch of runs.

    ``seed`` makes the K-means fit (and hence this metric) deterministic; it
    is passed straight through to ``kmeans2``'s own ``seed=`` kwarg. Per that
    function's docstring, an ``int`` seed spins up a fresh
    ``numpy.random.RandomState`` internal to the call, so this never reads or
    mutates global NumPy RNG state. Initialisation uses ``minit="points"``
    (centroids are a random subset of the actual evaluated contexts) rather
    than the default ``"random"`` (Gaussian-moment-matched centroids), since
    the latter needs a well-defined per-dimension variance and can misbehave
    on tiny/duplicate-heavy context sets (e.g. a single evaluated point, or
    many repeated contexts) -- exactly the small-``n_eval`` edge cases this
    function must handle gracefully.
    """
    if k <= 0:
        return 0.0
    if all_contexts.numel() == 0:
        return 0.0
    if feasible_Y.numel() == 0 or feasible_contexts.numel() == 0:
        return 0.0
    if feasible_Y.ndim != 2:
        raise ValueError(f"feasible_Y must be 2D, got shape {tuple(feasible_Y.shape)}")
    if feasible_contexts.ndim != 2:
        raise ValueError(
            f"feasible_contexts must be 2D, got shape {tuple(feasible_contexts.shape)}"
        )
    if feasible_contexts.shape[0] != feasible_Y.shape[0]:
        raise ValueError(
            "feasible_contexts and feasible_Y must have the same number of rows "
            f"(got {feasible_contexts.shape[0]} vs {feasible_Y.shape[0]})"
        )

    import warnings

    try:
        from scipy.cluster.vq import kmeans2, vq
    except ImportError as exc:  # pragma: no cover - environment issue
        raise ImportError(
            "scipy is required for localized_feasible_convex_hull_volume"
        ) from exc

    all_ctx_np = all_contexts.detach().double().cpu().numpy()
    n_eval = int(all_ctx_np.shape[0])
    effective_k = min(int(k), n_eval)
    if effective_k < 1:
        return 0.0

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            centroids, _ = kmeans2(
                all_ctx_np, effective_k, minit="points", seed=int(seed)
            )
    except Exception:
        return 0.0

    feas_ctx_np = feasible_contexts.detach().double().cpu().numpy()
    try:
        labels, _ = vq(feas_ctx_np, centroids)
    except Exception:
        return 0.0

    Y = feasible_Y.detach().double()
    total = 0.0
    for region in range(effective_k):
        mask = labels == region
        if not mask.any():
            continue
        Y_region = Y[torch.from_numpy(mask)]
        total += feasible_convex_hull_volume(Y_region)
    return total / float(k)
