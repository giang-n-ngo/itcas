"""Shared "score -> QD-DPP L-ensemble -> greedy batch" boilerplate.

The four discrete-pool DPP-batch baselines (``random_batch``,
``straddle_batch``, ``cas_eci_batch``, ``moc_cas_hard_batch``) all follow the
same three-step recipe used by the proposed method's own discrete variant
(``algorithms/itcas.py::select_batch``): take a per-candidate quality score,
build the Quality-Diversity L-ensemble over the predicted objective-space
(and, if available, context-space) coordinates, then greedily maximise
``log det(I + L_B)``. This module factors that shared plumbing out so each
baseline only needs to supply its (already-computed) score vector.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
from botorch.models import SingleTaskGP

from ..algorithms.qd_dpp import build_qd_l_ensemble, greedy_dpp_batch
from ..utils.gp import posterior_mean_std


def score_to_dpp_batch(
    models: Sequence[SingleTaskGP],
    cand: torch.Tensor,
    score: torch.Tensor,
    batch_size: int,
    *,
    context_dims: Optional[Sequence[int]] = None,
    dpp_lambda: Optional[float] = None,
    dpp_lambda_ctx: Optional[float] = None,
) -> list[int]:
    """Build the QD L-ensemble from ``score`` and greedily select a batch.

    ``score`` is clamped to be non-negative (the L-ensemble requires
    non-negative "quality"; negative acquisition scores are floored at 0 so
    they simply cannot be selected rather than corrupting the PSD ensemble).
    """
    mu, _ = posterior_mean_std(models, cand)
    ctx = None
    if context_dims:
        ctx = cand[:, list(context_dims)]
    L = build_qd_l_ensemble(
        quality=score.clamp_min(0.0), mu=mu, ctx=ctx,
        lam=dpp_lambda, lam_ctx=dpp_lambda_ctx,
    )
    return greedy_dpp_batch(L, batch_size=batch_size)
