"""Pluggable candidate-quality variants for the continuous C-MO-CAS pipeline.

The continuous acquisition (``continuous.select_batch_continuous``) factors into
two parts: a differentiable *candidate quality* ``q(z)`` and the Quality-Diversity
DPP that promotes batch diversity. This module owns the *quality* half and makes
it swappable so new information-theoretic measures can be added without touching
the pipeline.

A **quality provider** is a builder that turns the current GP posteriors into a
differentiable quality function ``q: (N, d) -> (N,)`` (plus an info dict). It is
registered under a short name and selected at run time via the ``--quality`` CLI
flag / ``quality`` config field.

Currently registered variants (see ``methodology.tex``):

``roi_mi`` -- Region-of-Interest Mutual Information. Global information gain of a
    candidate against a hallucinated *continuous* reference set ``Z_ref`` built by
    differentiable Thompson sampling + multi-start margin ascent. Targets the deep
    feasible interior but pays the cost of reference-set construction each round.

``efig`` -- Expected Feasible Information Gain. Weights the candidate's continuous
    objective information gain by its Probability of Feasibility (PoF), so the
    incentive shifts from the feasibility *boundary* to the feasible *interior*::

        q(z) = ( prod_i Phi((mu_i - tau_i)/sigma_i) )
               * sum_i 1/2 log(1 + sigma_{f,i}^2(z) / sigma_{eps,i}^2)

    It needs neither Thompson sampling nor a reference set, so it is a cheap,
    purely local interior-seeking quality.

``edig`` -- Expected Depth Information Gain. Replaces the PoF multiplier in EFIG
    with the Standardized Expected Feasible Margin ``Z_i*Phi(Z_i) + phi(Z_i)``,
    resolving EFIG's saturation trap (PoF hits 1.0 in the deep interior) and
    cold-start trap (PoF vanishes far from feasibility). The depth term grows
    linearly with predicted feasibility depth, pulling the search into the deepest
    objective-space topography::

        Z_i    = (mu_i - tau_i) / sigma_i
        depth  = prod_i [ Z_i * Phi(Z_i) + phi(Z_i) ]
        q(z)   = depth * sum_i 1/2 log(1 + sigma_{f,i}^2(z) / sigma_{eps,i}^2)

    Like EFIG, it requires no reference set and is fully differentiable.

``ndig`` -- Normalized Depth Information Gain. Applies a parameter-free rational
    squashing function to each EDIG depth term, bounding the per-objective factor
    to ``[0, 1)`` so unbounded depth scores cannot destabilise the QD-DPP
    L-ensemble and sacrifice context-space diversity::

        d_i    = Z_i * Phi(Z_i) + phi(Z_i)
        depth  = prod_i d_i / (1 + d_i)          # -> [0, 1)
        q(z)   = depth * sum_i 1/2 log(1 + sigma_{f,i}^2(z) / sigma_{eps,i}^2)

    Preserves EDIG's interior-pull and cold-start robustness while restoring
    the DPP's ability to enforce context-space diversity (CFD/FCFD).

Adding a new variant is a one-liner::

    @register_quality("my_variant")
    def _build_my_variant(models, bounds, tau, **kw):
        def q(z): ...
        return q, {"quality": "my_variant"}
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
from botorch.models import SingleTaskGP

from torch.distributions import Normal as _Normal

from ..utils.gp import observation_noise
from .roi_mi import (
    feasibility_probabilities,
    joint_feasibility_probability,
)

_STDNORM = _Normal(0.0, 1.0)

_EPS = 1e-12

# A quality function maps (N, d) query points to (N,) quality scores and must be
# differentiable w.r.t. its input (the greedy DPP optimizer ascends through it).
QualityFn = Callable[[torch.Tensor], torch.Tensor]
# A builder turns posteriors + domain into a (QualityFn, info) pair.
QualityBuilder = Callable[..., tuple[QualityFn, dict]]

QUALITY_REGISTRY: dict[str, QualityBuilder] = {}


def register_quality(name: str) -> Callable[[QualityBuilder], QualityBuilder]:
    """Register a quality-provider builder under ``name``."""

    def deco(fn: QualityBuilder) -> QualityBuilder:
        if name in QUALITY_REGISTRY:
            raise ValueError(f"Quality variant '{name}' already registered.")
        QUALITY_REGISTRY[name] = fn
        return fn

    return deco


def available_qualities() -> list[str]:
    """Names of all registered quality variants."""
    return sorted(QUALITY_REGISTRY)


def build_quality_fn(
    variant: str,
    models: Sequence[SingleTaskGP],
    bounds: torch.Tensor,
    tau: torch.Tensor,
    *,
    context_dims: Sequence[int] = (),
    gamma: float = 0.1,
    n_restarts: int = 16,
    n_opt_steps: int = 60,
    opt_lr: float = 0.05,
    n_ts_samples: int = 1,
    rng_seed: Optional[int] = None,
) -> tuple[QualityFn, dict]:
    """Dispatch to the requested quality variant and build its ``q(z)``.

    All variants share this signature; each consumes whatever keyword arguments
    it needs and ignores the rest, so the pipeline can pass a uniform set of
    hyperparameters regardless of which variant is active.

    Returns ``(quality_fn, info)``.
    """
    if variant not in QUALITY_REGISTRY:
        raise ValueError(
            f"Unknown quality variant '{variant}'. "
            f"Choices: {available_qualities()}"
        )
    return QUALITY_REGISTRY[variant](
        models,
        bounds,
        tau,
        context_dims=context_dims,
        gamma=gamma,
        n_restarts=n_restarts,
        n_opt_steps=n_opt_steps,
        opt_lr=opt_lr,
        n_ts_samples=n_ts_samples,
        rng_seed=rng_seed,
    )


# ---------------------------------------------------------------------------
# EFIG: Expected Feasible Information Gain
# ---------------------------------------------------------------------------
def efig_quality(
    models: Sequence[SingleTaskGP],
    Z_eval: torch.Tensor,
    h: torch.Tensor,
) -> torch.Tensor:
    """Differentiable EFIG quality ``q(z) = p(z) * I(f(z); y(z) | D_t)``.

    The continuous objective information gain (the log-determinant of the joint
    predictive covariance over the ``m`` independent GP objectives) weighted by
    the Probability of Feasibility (PoF). With independent GPs and Gaussian
    observation noise it reduces to::

        p(z)   = prod_i Phi((mu_{t,i}(z) - tau_i) / sigma_{t,i}(z))
        I(z)   = sum_i 1/2 log(1 + sigma_{t,i}^2(z) / sigma_{eps,i}^2)
        q(z)   = p(z) * I(z)

    Weighting by PoF moves the exploratory incentive off the feasibility
    boundary (where pure entropy / LSE methods stall) and into the feasible
    interior. The computation keeps autograd enabled so ``q`` is differentiable
    w.r.t. ``Z_eval`` for the continuous QD-DPP optimizer.

    Returns ``(N,)``.
    """
    mus, sigmas = [], []
    info_gain = torch.zeros(Z_eval.shape[0], dtype=Z_eval.dtype, device=Z_eval.device)
    for gp in models:
        post = gp.posterior(Z_eval)
        mu_i = post.mean.squeeze(-1)
        var_i = post.variance.clamp_min(_EPS).squeeze(-1)
        noise_i = observation_noise(gp)
        mus.append(mu_i)
        sigmas.append(var_i.sqrt())
        info_gain = info_gain + 0.5 * torch.log1p(var_i / noise_i)
    mu = torch.stack(mus, dim=-1)       # (N, m)
    sigma = torch.stack(sigmas, dim=-1)  # (N, m)
    pof = joint_feasibility_probability(feasibility_probabilities(mu, sigma, h))
    return pof * info_gain


@register_quality("efig")
def _build_efig(
    models: Sequence[SingleTaskGP],
    bounds: torch.Tensor,
    tau: torch.Tensor,
    **_: object,
) -> tuple[QualityFn, dict]:
    """EFIG provider: no reference set, just the local PoF-weighted info gain."""

    def q(z: torch.Tensor) -> torch.Tensor:
        return efig_quality(models, z, tau)

    info = {"quality": "efig", "ref_set_size": 0, "cold_start": False}
    return q, info


# ---------------------------------------------------------------------------
# EDIG: Expected Depth Information Gain
# ---------------------------------------------------------------------------
def edig_quality(
    models: Sequence[SingleTaskGP],
    Z_eval: torch.Tensor,
    h: torch.Tensor,
) -> torch.Tensor:
    """Differentiable EDIG quality.

    Replaces the PoF multiplier of EFIG with the Standardized Expected Feasible
    Margin per objective, resolving the saturation and cold-start traps::

        Z_i    = (mu_{t,i}(z) - tau_i) / sigma_{t,i}(z)
        depth  = prod_i [ Z_i * Phi(Z_i) + phi(Z_i) ]   # grows ~ Z_i for Z_i >> 0
        I(z)   = sum_i 1/2 log(1 + sigma_{t,i}^2(z) / sigma_{eps,i}^2)
        q(z)   = depth * I(z)

    For Z_i >> 0 the term Z_i*Phi(Z_i)+phi(Z_i) ≈ Z_i, so the reward grows
    linearly with predicted depth, continuously pulling the search into the deep
    feasible interior. For Z_i < 0 (cold-start) phi(Z_i) prevents the gradient
    from vanishing, unlike the CDF alone in EFIG.

    Returns ``(N,)``.
    """
    depth = torch.ones(Z_eval.shape[0], dtype=Z_eval.dtype, device=Z_eval.device)
    info_gain = torch.zeros_like(depth)
    for gp, tau_i in zip(models, h):
        post = gp.posterior(Z_eval)
        mu_i = post.mean.squeeze(-1)              # (N,)
        var_i = post.variance.clamp_min(_EPS).squeeze(-1)  # (N,)
        sigma_i = var_i.sqrt()                    # (N,)
        noise_i = observation_noise(gp)
        # Standardized depth score
        z_i = (mu_i - tau_i) / sigma_i           # (N,)
        # Expected feasible margin: E[Z | Z > 0] envelope valid for all Z
        # = Z * Phi(Z) + phi(Z)
        phi_z = torch.exp(_STDNORM.log_prob(z_i))  # phi(Z_i), (N,)
        Phi_z = _STDNORM.cdf(z_i)                  # Phi(Z_i), (N,)
        depth = depth * (z_i * Phi_z + phi_z)
        info_gain = info_gain + 0.5 * torch.log1p(var_i / noise_i)
    return depth * info_gain


@register_quality("edig")
def _build_edig(
    models: Sequence[SingleTaskGP],
    bounds: torch.Tensor,
    tau: torch.Tensor,
    **_: object,
) -> tuple[QualityFn, dict]:
    """EDIG provider: depth-weighted info gain, no reference set needed."""

    def q(z: torch.Tensor) -> torch.Tensor:
        return edig_quality(models, z, tau)

    info = {"quality": "edig", "ref_set_size": 0, "cold_start": False}
    return q, info


# ---------------------------------------------------------------------------
# NDIG: Normalized Depth Information Gain
# ---------------------------------------------------------------------------
def ndig_quality(
    models: Sequence[SingleTaskGP],
    Z_eval: torch.Tensor,
    h: torch.Tensor,
) -> torch.Tensor:
    """Differentiable NDIG quality.

    Applies a parameter-free rational squashing function to EDIG's depth
    multiplier, bounding each per-objective factor to ``[0, 1)`` and preventing
    unbounded quality scores from destabilising the QD-DPP L-ensemble::

        Z_i      = (mu_{t,i}(z) - tau_i) / sigma_{t,i}(z)
        d_i      = Z_i * Phi(Z_i) + phi(Z_i)          # EDIG depth
        norm_d_i = d_i / (1 + d_i)                    # rational squash -> [0,1)
        depth    = prod_i norm_d_i
        I(z)     = sum_i 1/2 log(1 + sigma_{t,i}^2(z) / sigma_{eps,i}^2)
        q(z)     = depth * I(z)

    As Z_i -> +inf, norm_d_i -> 1 (no saturation cliff, but bounded).
    For Z_i < 0 (cold-start) the phi(Z_i) term keeps the gradient non-zero.
    The cap prevents trivial contexts from dominating the DPP, restoring
    context-space diversity (CFD/FCFD).

    Returns ``(N,)``.
    """
    depth = torch.ones(Z_eval.shape[0], dtype=Z_eval.dtype, device=Z_eval.device)
    info_gain = torch.zeros_like(depth)
    for gp, tau_i in zip(models, h):
        post = gp.posterior(Z_eval)
        mu_i = post.mean.squeeze(-1)                        # (N,)
        var_i = post.variance.clamp_min(_EPS).squeeze(-1)  # (N,)
        sigma_i = var_i.sqrt()                              # (N,)
        noise_i = observation_noise(gp)
        z_i = (mu_i - tau_i) / sigma_i                     # (N,)
        phi_z = torch.exp(_STDNORM.log_prob(z_i))          # phi(Z_i)
        Phi_z = _STDNORM.cdf(z_i)                          # Phi(Z_i)
        d_i = z_i * Phi_z + phi_z                          # EDIG depth per obj
        depth = depth * (d_i / (1.0 + d_i))                # rational squash
        info_gain = info_gain + 0.5 * torch.log1p(var_i / noise_i)
    return depth * info_gain


@register_quality("ndig")
def _build_ndig(
    models: Sequence[SingleTaskGP],
    bounds: torch.Tensor,
    tau: torch.Tensor,
    **_: object,
) -> tuple[QualityFn, dict]:
    """NDIG provider: rationally-squashed depth-weighted info gain."""

    def q(z: torch.Tensor) -> torch.Tensor:
        return ndig_quality(models, z, tau)

    info = {"quality": "ndig", "ref_set_size": 0, "cold_start": False}
    return q, info


# ---------------------------------------------------------------------------
# ROI-MI: Region-of-Interest Mutual Information
# ---------------------------------------------------------------------------
@register_quality("roi_mi")
def _build_roi_mi(
    models: Sequence[SingleTaskGP],
    bounds: torch.Tensor,
    tau: torch.Tensor,
    *,
    gamma: float = 0.1,
    n_restarts: int = 16,
    n_opt_steps: int = 60,
    opt_lr: float = 0.05,
    n_ts_samples: int = 1,
    rng_seed: Optional[int] = None,
    **_: object,
) -> tuple[QualityFn, dict]:
    """ROI-MI provider: build the continuous reference set, then score against it.

    Imported lazily to avoid a circular import with ``continuous`` (which imports
    this registry).
    """
    from .continuous import build_reference_set, roi_mi_quality_continuous

    Z_ref, info = build_reference_set(
        models,
        bounds,
        tau,
        gamma=gamma,
        n_restarts=n_restarts,
        n_steps=n_opt_steps,
        lr=opt_lr,
        n_ts_samples=n_ts_samples,
        rng_seed=rng_seed,
    )

    def q(z: torch.Tensor) -> torch.Tensor:
        return roi_mi_quality_continuous(models, z, Z_ref, tau)

    info = {**info, "quality": "roi_mi"}
    return q, info
