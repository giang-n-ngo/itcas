"""Baseline acquisition placeholders.

All baselines share the same signature:

    select_batch(models, cand, h, batch_size, **kwargs) -> (indices, info)

The pipeline additionally passes `X_obs`, `Y_obs`, `radius`, `beta`, etc. via
**kwargs; baselines that don't need them simply ignore them.

Concrete implementations: Random, ONE-S, EZ, EISR, STRADDLE here; CAS/MOC-CAS
in `cas_family.py`. Stubs remain for `eps_constraint` and `moo_cluster`.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
from botorch.models import SingleTaskGP

from ..algorithms.quality import sigma_combined
from ..algorithms.roi_mi import (
    feasibility_probabilities,
    joint_feasibility_probability,
    _binary_entropy,
)
from ..utils.gp import posterior_mean_std
from .batch_dpp import score_to_dpp_batch


def random_baseline(
    models, cand: torch.Tensor, h, batch_size: int = 1, rng_seed: Optional[int] = None,
    **kwargs,
):
    g = torch.Generator()
    if rng_seed is not None:
        g.manual_seed(rng_seed)
    idx = torch.randperm(cand.shape[0], generator=g)[:batch_size].tolist()
    return idx, {"baseline": "random"}


def random_batch(
    models, cand: torch.Tensor, h, batch_size: int = 1,
    context_dims: Optional[Sequence[int]] = None,
    dpp_lambda: Optional[float] = None, dpp_lambda_ctx: Optional[float] = None,
    **kwargs,
):
    """Random + DPP: pure space-filling diversity, no acquisition greediness.

    There is no meaningful acquisition score for Random; per the plan doc,
    "Random + DPP" is a diversity-enforced (space-filling) baseline, so every
    candidate gets uniform quality 1.0 and the batch is chosen purely by the
    QD-DPP diversity kernel (objective-space and, if available, context-space).
    """
    score = torch.ones(cand.shape[0], dtype=cand.dtype, device=cand.device)
    idx = score_to_dpp_batch(
        models, cand, score, batch_size,
        context_dims=context_dims, dpp_lambda=dpp_lambda, dpp_lambda_ctx=dpp_lambda_ctx,
    )
    return idx, {"baseline": "random_batch", "score": score.detach().cpu()}


def one_step_active_search(
    models: list[SingleTaskGP], cand: torch.Tensor, h: torch.Tensor,
    batch_size: int = 1, **kwargs,
):
    """ONE-S: greedily pick highest p(Z(x)=1 | D_t)."""
    mu, sigma = posterior_mean_std(models, cand)
    p = joint_feasibility_probability(feasibility_probabilities(mu, sigma, h))
    idx = torch.topk(p, k=batch_size).indices.tolist()
    return idx, {"baseline": "one_step", "score": p.detach().cpu()}


def ez_mutual_information(
    models, cand: torch.Tensor, h: torch.Tensor, batch_size: int = 1, **kwargs,
):
    """EZ: maximise H(Z) -> binary entropy of joint feasibility prob."""
    mu, sigma = posterior_mean_std(models, cand)
    p = joint_feasibility_probability(feasibility_probabilities(mu, sigma, h))
    H = _binary_entropy(p)
    idx = torch.topk(H, k=batch_size).indices.tolist()
    return idx, {"baseline": "ez", "score": H.detach().cpu()}


def eisr(
    models, cand: torch.Tensor, h: torch.Tensor, batch_size: int = 1, **kwargs,
):
    """EISR: alpha = p(Z=1) * H(y). Approximate H(y) by sum over GPs."""
    import math

    mu, sigma = posterior_mean_std(models, cand)
    p = joint_feasibility_probability(feasibility_probabilities(mu, sigma, h))
    Hy = (0.5 * (1.0 + math.log(2 * math.pi)) + sigma.log()).sum(dim=-1)
    score = p * Hy
    idx = torch.topk(score, k=batch_size).indices.tolist()
    return idx, {"baseline": "eisr", "score": score.detach().cpu()}


def straddle(
    models, cand: torch.Tensor, h: torch.Tensor, batch_size: int = 1,
    beta: float = 1.96, t: int = 0, **kwargs,
):
    """STRADDLE: single-objective alpha_i = beta*sigma_i - |mu_i - h_i|,
    applied via round-robin objective alternation.

    Multi-objective adaptation: rather than aggregating the per-objective
    scores, the acquisition uses *only* objective ``active_obj = t % m``
    (``m = h.numel()``) on pipeline iteration ``t``, cycling through all
    objectives round-robin across iterations. ``t`` defaults to 0 so direct/
    legacy calls without it degenerate to always-objective-0.
    """
    mu, sigma = posterior_mean_std(models, cand)
    m = h.numel()
    active_obj = t % m
    score = beta * sigma[:, active_obj] - (mu[:, active_obj] - h[active_obj]).abs()
    idx = torch.topk(score, k=batch_size).indices.tolist()
    return idx, {
        "baseline": "straddle",
        "score": score.detach().cpu(),
        "active_obj": int(active_obj),
    }


def straddle_batch(
    models, cand: torch.Tensor, h: torch.Tensor, batch_size: int = 1,
    beta: float = 1.96, context_dims: Optional[Sequence[int]] = None,
    dpp_lambda: Optional[float] = None, dpp_lambda_ctx: Optional[float] = None,
    **kwargs,
):
    """STRADDLE + DPP: full-pool straddle score fed into the QD-DPP L-ensemble.

    Reuses ``straddle``'s full-pool score (batch_size=cand.shape[0] makes the
    base call score every candidate, which it already does before top-k) so
    the acquisition math is identical to the sequential baseline; only the
    selection mechanism changes from top-k to greedy submodular diversity.
    """
    _, info = straddle(models, cand, h, batch_size=cand.shape[0], beta=beta, **kwargs)
    score = info["score"].to(cand)
    idx = score_to_dpp_batch(
        models, cand, score, batch_size,
        context_dims=context_dims, dpp_lambda=dpp_lambda, dpp_lambda_ctx=dpp_lambda_ctx,
    )
    return idx, {
        "baseline": "straddle_batch",
        "score": score.detach().cpu(),
        "active_obj": info["active_obj"],
    }


_INTERIOR_POF_TARGET = 0.95  # per contexts/fair_lse_comparison.md Mechanism 3


def interior_sampling_discrete(
    models, cand: torch.Tensor, h: torch.Tensor, batch_size: int = 1,
    pof_target: float = _INTERIOR_POF_TARGET, **kwargs,
):
    """Discrete-pool Stage-2 ("Interior") acquisition (Family C).

    Filters the candidate pool by joint PoF(z) >= ``pof_target`` (default
    0.95) and, among the survivors, picks the highest ``sigma_combined``
    (L2 magnitude of the per-objective posterior std, see
    ``algorithms.quality.sigma_combined``). Score is set to 0 for any
    candidate that fails the PoF gate so it can never be selected by
    ``topk``. If *no* candidate clears the PoF gate the score is uniformly
    zero and the caller's existing empty-index fallback
    (``pipeline.loop._fallback_indices``) takes over -- this function
    intentionally returns an empty-scored vector rather than reimplementing
    that fallback itself, to stay consistent with how every other baseline's
    degenerate case is already handled by ``run_experiment``.
    """
    mu, sigma = posterior_mean_std(models, cand)
    pof = joint_feasibility_probability(feasibility_probabilities(mu, sigma, h))
    feasible_mask = pof >= pof_target
    sc = sigma_combined(models, cand)
    score = torch.where(feasible_mask, sc, torch.zeros_like(sc))
    k = min(batch_size, cand.shape[0])
    if k <= 0 or not bool(feasible_mask.any()):
        return [], {"baseline": "interior_sampling", "score": score.detach().cpu(), "stage": "interior"}
    idx = torch.topk(score, k=k).indices.tolist()
    return idx, {"baseline": "interior_sampling", "score": score.detach().cpu(), "stage": "interior"}


def interior_sampling_discrete_batch(
    models, cand: torch.Tensor, h: torch.Tensor, batch_size: int = 1,
    pof_target: float = _INTERIOR_POF_TARGET,
    context_dims: Optional[Sequence[int]] = None,
    dpp_lambda: Optional[float] = None, dpp_lambda_ctx: Optional[float] = None,
    **kwargs,
):
    """Interior Sampling + DPP: PoF-gated sigma_combined score fed into the QD-DPP L-ensemble.

    Zero-quality (PoF-failing) candidates cannot be selected by
    ``greedy_dpp_batch`` (its L-ensemble diagonal is 0 for them, so their
    marginal gain is 0 and the greedy loop breaks before picking them) --
    this gives the PoF>=0.95 constraint "for free" via the same clamp-at-0
    PSD requirement documented in ``batch_dpp.score_to_dpp_batch``, without
    new plumbing. If no candidate clears the gate, ``score_to_dpp_batch``
    returns an empty list and the caller's existing fallback takes over.
    """
    _, info = interior_sampling_discrete(
        models, cand, h, batch_size=cand.shape[0], pof_target=pof_target,
    )
    score = info["score"].to(cand)
    idx = score_to_dpp_batch(
        models, cand, score, batch_size,
        context_dims=context_dims, dpp_lambda=dpp_lambda, dpp_lambda_ctx=dpp_lambda_ctx,
    )
    return idx, {"baseline": "interior_sampling_batch", "score": score.detach().cpu(), "stage": "interior"}


def eps_constraint_bo(*args, **kwargs):
    raise NotImplementedError("eps-constraint BO baseline: TODO (Scientific Coder).")


def moo_cluster(*args, **kwargs):
    raise NotImplementedError("MOO+Cluster baseline: TODO (Scientific Coder).")


REGISTRY = {
    "random": random_baseline,
    "random_batch": random_batch,
    "one_step": one_step_active_search,
    "ez": ez_mutual_information,
    "eisr": eisr,
    "straddle": straddle,
    "straddle_batch": straddle_batch,
    "interior_sampling": interior_sampling_discrete,
    "interior_sampling_batch": interior_sampling_discrete_batch,
    "eps_constraint": eps_constraint_bo,
    "moo_cluster": moo_cluster,
}

from .cas_family import REGISTRY as _CAS_REGISTRY  # noqa: E402
REGISTRY.update(_CAS_REGISTRY)
