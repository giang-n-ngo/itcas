"""CAS (ECI) and MOC-CAS (hard + soft) baselines.

Per user direction: these algorithms are not contextual; we treat the context
vector as additional input dimensions and run acquisition optimization on the
combined input+context space `z = (x, c)`. So `cand` and `X_obs` here may live
in the joint Z = X x C space — no special handling needed.

References: contexts/cas.md, contexts/moccas.md.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from botorch.models import SingleTaskGP

from ..algorithms.roi_mi import (
    _normal_cdf,
    feasibility_probabilities,
    joint_feasibility_probability,
)
from ..utils.gp import posterior_mean_std


def _uniform_in_ball(n: int, dim: int, radius: float, generator) -> torch.Tensor:
    """Draw n samples uniformly from the d-dimensional ball of radius `radius`."""
    eps = torch.randn(n, dim, generator=generator)
    eps = eps / eps.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    u = torch.rand(n, generator=generator).pow(1.0 / dim).unsqueeze(-1)
    return radius * eps * u


def cas_eci(
    models: list[SingleTaskGP],
    cand: torch.Tensor,
    h: torch.Tensor,
    batch_size: int = 1,
    X_obs: Optional[torch.Tensor] = None,
    radius: float = 0.1,
    n_mc: int = 64,
    rng_seed: Optional[int] = None,
    **kwargs,
):
    r"""Expected Coverage Improvement (CAS).

    α(x | D_t) = E_Z[ Vol( {N_r(x) ∩ S_Z} \ N_r(X_t) ) ]

    Monte-Carlo estimate per candidate x:
        1. draw n_mc samples u_k uniformly in N_r(x)
        2. score(x) = Vol(N_r) · mean_k [ p(feasible at u_k | D_t) ·
                                          1[u_k ∉ N_r(X_t)] ]
    The leading Vol(N_r) is constant across candidates and dropped.
    """
    N, d = cand.shape
    g = torch.Generator()
    if rng_seed is not None:
        g.manual_seed(rng_seed)

    offsets = _uniform_in_ball(n_mc, d, radius, g).to(cand)  # (n_mc, d)
    pts = cand.unsqueeze(1) + offsets.unsqueeze(0)           # (N, n_mc, d)
    flat = pts.reshape(-1, d)

    mu, sigma = posterior_mean_std(models, flat)
    p = joint_feasibility_probability(
        feasibility_probabilities(mu, sigma, h)
    ).reshape(N, n_mc)

    if X_obs is not None and X_obs.shape[0] > 0:
        d_to_obs = torch.cdist(flat, X_obs.to(flat))         # (N*n_mc, n_obs)
        covered = (d_to_obs <= radius).any(dim=-1).reshape(N, n_mc)
        weight = (~covered).to(p.dtype)
    else:
        weight = torch.ones_like(p)

    score = (p * weight).mean(dim=-1)
    idx = torch.topk(score, k=min(batch_size, N)).indices.tolist()
    return idx, {"baseline": "cas_eci", "score": score.detach().cpu()}


def moc_cas_hard(
    models: list[SingleTaskGP],
    cand: torch.Tensor,
    h: torch.Tensor,
    batch_size: int = 1,
    Y_obs: Optional[torch.Tensor] = None,
    radius: float = 0.1,
    beta: float = 2.0,
    n_mc: int = 64,
    rng_seed: Optional[int] = None,
    **kwargs,
):
    r"""Hard geometric MOC-CAS acquisition (objective-space coverage).

        α(x) = Z(x) · Vol( (B_r(U(x)) ∩ S) \ ∪_s B_r(y_s) )

    where U(x) = μ(x) + sqrt(β) · σ(x), Z(x) = 1[U(x) ∈ S].
    Vol(B_r) is constant; we report the (uncovered ∩ feasible) fraction.
    """
    mu, sigma = posterior_mean_std(models, cand)         # (N, m)
    U = mu + math.sqrt(beta) * sigma                     # (N, m)
    m = U.shape[-1]

    Z_ind = (U >= h).all(dim=-1).to(U.dtype)             # (N,)

    g = torch.Generator()
    if rng_seed is not None:
        g.manual_seed(rng_seed)
    offsets = _uniform_in_ball(n_mc, m, radius, g).to(U)  # (n_mc, m)
    pts = U.unsqueeze(1) + offsets.unsqueeze(0)           # (N, n_mc, m)
    feas = (pts >= h).all(dim=-1)                         # (N, n_mc)

    if Y_obs is not None and Y_obs.shape[0] > 0:
        flat = pts.reshape(-1, m)
        d_to_y = torch.cdist(flat, Y_obs.to(flat))       # (N*n_mc, n_obs)
        covered = (d_to_y <= radius).any(dim=-1).reshape(pts.shape[0], pts.shape[1])
        new = feas & (~covered)
    else:
        new = feas

    score = Z_ind * new.to(U.dtype).mean(dim=-1)
    idx = torch.topk(score, k=min(batch_size, cand.shape[0])).indices.tolist()
    return idx, {"baseline": "moc_cas_hard", "score": score.detach().cpu()}


def moc_cas_soft(
    models: list[SingleTaskGP],
    cand: torch.Tensor,
    h: torch.Tensor,
    batch_size: int = 1,
    Y_obs: Optional[torch.Tensor] = None,
    radius: float = 0.1,
    beta: float = 2.0,
    lam: float = 0.1,
    **kwargs,
):
    r"""Soft surrogate MOC-CAS acquisition.

        α(x) = V_m(r) · p_sat(U(x)) · n(U(x))

    with the smooth probit gate
        p_sat(U) = Π_i Φ((U_i - τ_i) / λ)
    and the bounded soft-OR (unit-mass Gaussian kernel) novelty
        n(U) = Π_s (1 - exp(-||U - y_s||² / (2 r²)))
              ∈ [0,1] — small if any y_s is near U; →1 when U is far from all y_s.
    V_m(r) is a constant and dropped.
    """
    mu, sigma = posterior_mean_std(models, cand)
    U = mu + math.sqrt(beta) * sigma                     # (N, m)

    p_sat = _normal_cdf((U - h) / max(lam, 1e-8)).clamp(1e-12, 1.0).prod(dim=-1)

    if Y_obs is None or Y_obs.shape[0] == 0:
        n_term = torch.ones_like(p_sat)
    else:
        d2 = torch.cdist(U, Y_obs.to(U)).pow(2)          # (N, n_obs)
        k = torch.exp(-d2 / (2.0 * radius * radius))     # (N, n_obs)
        n_term = (1.0 - k).clamp(0.0, 1.0).prod(dim=-1)  # (N,)

    score = p_sat * n_term
    idx = torch.topk(score, k=min(batch_size, cand.shape[0])).indices.tolist()
    return idx, {"baseline": "moc_cas_soft", "score": score.detach().cpu()}


REGISTRY = {
    "cas_eci": cas_eci,
    "moc_cas_hard": moc_cas_hard,
    "moc_cas_soft": moc_cas_soft,
}
