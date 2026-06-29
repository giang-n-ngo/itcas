"""Top-level acquisition combining ROI-MI quality + QD-DPP diversity."""
from __future__ import annotations

from typing import Optional, Sequence

import torch
from botorch.models import SingleTaskGP

from ..utils.gp import posterior_mean_std
from .qd_dpp import build_qd_l_ensemble, greedy_dpp_batch
from .roi_mi import roi_mi_quality


def select_batch(
    models: list[SingleTaskGP],
    cand: torch.Tensor,
    h: torch.Tensor,
    batch_size: int = 1,
    n_ts_samples: int = 1,
    dpp_lambda: Optional[float] = None,
    dpp_lambda_ctx: Optional[float] = None,
    context_dims: Optional[Sequence[int]] = None,
    rng_seed: Optional[int] = None,
) -> tuple[list[int], dict]:
    """Run ROI-MI + QD-DPP greedy selection over `cand`.

    Args:
        models: per-objective GPs conditioned on D_t.
        cand: (N, d) candidate pool of design-context pairs z = (x, c).
        h: (m,) feasibility thresholds.
        batch_size: number of points b to select.
        n_ts_samples: number of ROI-MI Thompson samples.
        dpp_lambda: objective-space RBF length-scale (median heuristic if None).
        dpp_lambda_ctx: context-space RBF length-scale (median heuristic if None).
        context_dims: indices of the context columns c within `cand`. When
            provided, the QD-DPP uses the joint kernel k_obj * k_ctx to enforce
            context-space diversity; otherwise only objective-space diversity.

    Returns (selected_indices, info_dict).
    """
    quality, ref_mask = roi_mi_quality(
        models=models, cand=cand, h=h, n_samples=n_ts_samples, rng_seed=rng_seed
    )
    mu, _ = posterior_mean_std(models, cand)
    ctx = None
    if context_dims is not None and len(context_dims) > 0:
        ctx = cand[:, list(context_dims)]
    L = build_qd_l_ensemble(
        quality=quality, mu=mu, ctx=ctx, lam=dpp_lambda, lam_ctx=dpp_lambda_ctx
    )
    selected = greedy_dpp_batch(L, batch_size=batch_size)
    info = {
        "quality": quality.detach().cpu(),
        "ref_set_size": int(ref_mask.sum().item()),
        "lambda_obj": dpp_lambda,
        "lambda_ctx": dpp_lambda_ctx,
        "context_dims": list(context_dims) if context_dims else [],
    }
    return selected, info
