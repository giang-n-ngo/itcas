"""Baseline acquisition placeholders.

All baselines share the same signature:

    select_batch(models, cand, h, batch_size, **kwargs) -> (indices, info)

The pipeline additionally passes `X_obs`, `Y_obs`, `radius`, `beta`, etc. via
**kwargs; baselines that don't need them simply ignore them.

Concrete implementations: Random, ONE-S, EZ, EISR, STRADDLE here; CAS/MOC-CAS
in `cas_family.py`. Stubs remain for `eps_constraint` and `moo_cluster`.
"""
from __future__ import annotations

from typing import Optional

import torch
from botorch.models import SingleTaskGP

from ..algorithms.roi_mi import (
    feasibility_probabilities,
    joint_feasibility_probability,
    _binary_entropy,
)
from ..utils.gp import posterior_mean_std


def random_baseline(
    models, cand: torch.Tensor, h, batch_size: int = 1, rng_seed: Optional[int] = None,
    **kwargs,
):
    g = torch.Generator()
    if rng_seed is not None:
        g.manual_seed(rng_seed)
    idx = torch.randperm(cand.shape[0], generator=g)[:batch_size].tolist()
    return idx, {"baseline": "random"}


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
    beta: float = 1.96, **kwargs,
):
    """STRADDLE: alternates objectives; alpha_i = beta*sigma_i - |mu_i - h_i|.

    Multi-objective adaptation: pick the objective via round-robin per call,
    or aggregate min over objectives. Here we use the min aggregation.
    """
    mu, sigma = posterior_mean_std(models, cand)
    score = (beta * sigma - (mu - h).abs()).min(dim=-1).values
    idx = torch.topk(score, k=batch_size).indices.tolist()
    return idx, {"baseline": "straddle", "score": score.detach().cpu()}


def eps_constraint_bo(*args, **kwargs):
    raise NotImplementedError("eps-constraint BO baseline: TODO (Scientific Coder).")


def moo_cluster(*args, **kwargs):
    raise NotImplementedError("MOO+Cluster baseline: TODO (Scientific Coder).")


REGISTRY = {
    "random": random_baseline,
    "one_step": one_step_active_search,
    "ez": ez_mutual_information,
    "eisr": eisr,
    "straddle": straddle,
    "eps_constraint": eps_constraint_bo,
    "moo_cluster": moo_cluster,
}

from .cas_family import REGISTRY as _CAS_REGISTRY  # noqa: E402
REGISTRY.update(_CAS_REGISTRY)
