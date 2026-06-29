"""Reference-set construction for metrics that need problem references.

These helpers turn a :class:`itcas.pipeline.problems.Problem` into the reference
sets needed by the contextual fill-distance metric:

* a dense sample of the *context* space (for CFD and FCFD), plus its diagonal
  (the penalty distance returned when no feasible context exists yet).

The functions are duck-typed on ``problem`` (need ``bounds``, ``context_dims``),
so they work both inside the live experiment loop and in offline reporting where
the problem is reconstructed from the registry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ReferenceData:
    """Bundled reference sets for the fill-distance metrics of one problem."""

    context_dims: tuple[int, ...]
    context_ref: Optional[torch.Tensor]   # (n_ctx, d_c) dense context grid
    context_penalty: float                # diagonal of the context box


def context_diagonal(problem) -> float:
    """Euclidean diagonal of the context-space bounding box."""
    cdims = list(problem.context_dims)
    if not cdims:
        return 0.0
    lo = problem.bounds[0, cdims]
    hi = problem.bounds[1, cdims]
    return float(torch.linalg.norm((hi - lo).double()).item())


def context_reference(problem, n: int, seed: int) -> Optional[torch.Tensor]:
    """Uniformly sample ``n`` reference points in the context space.

    Returns ``None`` for problems without context dimensions.
    """
    cdims = list(problem.context_dims)
    if not cdims:
        return None
    g = torch.Generator().manual_seed(int(seed))
    lo = problem.bounds[0, cdims].double()
    hi = problem.bounds[1, cdims].double()
    u = torch.rand(n, len(cdims), generator=g, dtype=torch.double)
    return lo + (hi - lo) * u


def build_reference_data(
    problem,
    *,
    n_context: int = 4000,
    seed: int = 0,
    thresholds: Optional[torch.Tensor] = None,
) -> ReferenceData:
    """Build the context reference set needed by the fill-distance metric."""
    ctx_ref = context_reference(problem, n_context, seed=seed + 7)
    return ReferenceData(
        context_dims=tuple(problem.context_dims),
        context_ref=ctx_ref,
        context_penalty=context_diagonal(problem),
    )
