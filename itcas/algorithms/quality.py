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

NDIG-B component ablations (reviewer-requested; see ``--method itcas
    --quality <name>``, batch-labeled ``itcas_<name>`` by ``reporting/
    visualize.py``, compared in ``reporting/ndig_b_component_ablation_
    comparison.py``):

    - "Remove information gain" -> ``ndig_no_infogain``: NDIG's normalized
      depth term alone, dropping the info-gain multiplier entirely.
    - "Remove normalization of expected depth" -> reuses ``edig`` unchanged
      (EDIG *is* NDIG's unbounded, unnormalized depth term times info gain).
    - "PoF times information gain" -> reuses ``efig`` unchanged (EFIG *is*
      exactly this formulation).
    - "Binary entropy of PoF" -> ``ndig_pof_entropy``: the base-2 binary
      entropy of the joint Probability of Feasibility, with no info-gain
      term at all.

``ndig_no_infogain`` -- NDIG with the information-gain multiplier removed,
    isolating the contribution of the normalized depth term alone::

        d_i    = Z_i * Phi(Z_i) + phi(Z_i)
        depth  = prod_i d_i / (1 + d_i)          # -> [0, 1)
        q(z)   = depth

``ndig_pof_entropy`` -- Replaces NDIG's depth * info-gain product with the
    base-2 binary entropy of the joint Probability of Feasibility, isolating
    a pure feasibility-boundary-seeking signal with no depth or info-gain
    term::

        p_i(z) = Phi((mu_i - tau_i) / sigma_i)
        p(z)   = prod_i p_i(z)
        q(z)   = -[p(z) log2 p(z) + (1-p(z)) log2(1-p(z))]

``cr_ndig`` -- Context-Repulsive NDIG (see ``contexts/sequential_ndig.md``).
    A purely *sequential* variant (batch_size always forced to 1 -- no QD-DPP
    is available to enforce context-space diversity between candidates within
    a batch). It multiplies the NDIG score by a repulsion penalty against
    every previously *evaluated* context (the full observed dataset
    ``D_{t-1}``, not just this iteration's picks), so proposing a context
    close to one already visited is exponentially discounted::

        q_CR-NDIG(x, c) = q_NDIG(x, c) * prod_{i=1}^{t-1} (1 - k_ctx(c, c_i))

    where ``k_ctx`` is an RBF kernel on the context subspace only. This
    restores, via acquisition-history repulsion, the context-space
    exploration pressure that QD-DPP normally supplies for batch methods.

``c2lse``, ``bes`` -- Family-B "naive cartographer" baselines
    (Confidence-based Continuous LSE, Binary Entropy Search; see
    ``contexts/fair_lse_comparison.md`` Section 2). These are exposed as their
    own top-level ``--method`` values (not selected via ``--quality``); see
    ``pipeline/loop.py``. They are registered here only so they can reuse the
    same continuous multistart / QD-DPP machinery as the proposed method.
    Both are extensions of a *single-objective* LSE formula from the
    literature into this codebase's ``m``-independent-GP setting; rather than
    aggregating all ``m`` objectives' per-objective scores into one scalar,
    each uses **round-robin objective alternation**: on pipeline iteration
    ``t`` (0-based), the acquisition is applied to *only* objective
    ``active_obj = t % m``, cycling through all objectives across iterations
    and ignoring the other ``m-1`` objectives that round. ``t`` is threaded
    in via ``build_quality_fn``'s ``t`` keyword. (RMILE was implemented and
    later removed -- its per-candidate joint-posterior-covariance solve over
    a reference pool was too expensive to keep in the benchmark matrix.)

``interior_sampling`` -- Family-C Stage 2 ("Interior") quality, shared by the
    ``<base>_then_sample`` baselines (Confidence-based Continuous LSE, Binary
    Entropy Search, and Straddle's continuous cousin) once the two-stage
    budget switches from "Search" to "Interior" (``t >= stage1_fraction*T``,
    default ``stage1_fraction=0.5`` unless overridden via a ``_lseNN`` method
    suffix; see ``pipeline/loop.py``'s ``_parse_two_stage``/``_stage_for``).
    Pure combined-uncertainty exploration
    soft-penalized to stay inside the predicted feasible region (joint
    PoF >= 0.95)::

        sigma_combined(z) = sqrt( sum_i sigma_i(z)^2 )
        PoF(z)            = prod_i Phi((mu_i(z) - tau_i) / sigma_i(z))
        q(z) = sigma_combined(z) - LAMBDA_PENALTY * relu(0.95 - PoF(z))

    This is a *soft*-constraint relaxation of the plan doc's literal hard
    constraint (BoTorch ``optimize_acqf`` with ``nonlinear_inequality_
    constraints``); see ``pipeline/loop.py`` module docs / the Phase-2 final
    report for why. It is registered here (rather than under
    ``CONTINUOUS_BASELINE_QUALITY`` next to c2lse/bes) because it is
    selected dynamically per-iteration by the two-stage dispatch in
    ``pipeline/loop.py``, not by a fixed ``--method`` -> quality mapping.

Adding a new variant is a one-liner::

    @register_quality("my_variant")
    def _build_my_variant(models, bounds, tau, **kw):
        def q(z): ...
        return q, {"quality": "my_variant"}
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import math

import numpy as np
import torch
from botorch.models import SingleTaskGP

from torch.distributions import Normal as _Normal

from ..utils.gp import observation_noise
from .qd_dpp import median_heuristic_lambda
from .roi_mi import (
    _binary_entropy,
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
    t: int = 0,
    X_obs: Optional[torch.Tensor] = None,
) -> tuple[QualityFn, dict]:
    """Dispatch to the requested quality variant and build its ``q(z)``.

    All variants share this signature; each consumes whatever keyword arguments
    it needs and ignores the rest, so the pipeline can pass a uniform set of
    hyperparameters regardless of which variant is active.

    ``t`` is the 0-based pipeline iteration index. Only the Family-B
    round-robin variants (``c2lse``, ``bes``) consume it, computing
    ``active_obj = t % len(models)`` to select which single objective drives
    the score on this iteration; every other builder discards it via ``**_``.

    ``X_obs`` is the full accumulated observed dataset ``D_{t-1}`` (design +
    context coordinates, before this iteration's new points). Only ``cr_ndig``
    consumes it (to build the context-repulsion penalty against acquisition
    history); every other builder discards it via ``**_``.

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
        t=t,
        X_obs=X_obs,
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
# NDIG-B component ablations: remove info gain / replace depth*IG with PoF entropy
# ---------------------------------------------------------------------------
def ndig_no_infogain_quality(
    models: Sequence[SingleTaskGP],
    Z_eval: torch.Tensor,
    h: torch.Tensor,
) -> torch.Tensor:
    """Differentiable NDIG quality with the information-gain multiplier removed.

    Ablation of NDIG-B's "remove information gain" component: the normalized
    depth term alone, with no info-gain multiplier::

        Z_i   = (mu_{t,i}(z) - tau_i) / sigma_{t,i}(z)
        d_i   = Z_i * Phi(Z_i) + phi(Z_i)      # EDIG depth per objective
        depth = prod_i d_i / (1 + d_i)         # rational squash -> [0, 1)
        q(z)  = depth

    Returns ``(N,)``.
    """
    depth = torch.ones(Z_eval.shape[0], dtype=Z_eval.dtype, device=Z_eval.device)
    for gp, tau_i in zip(models, h):
        post = gp.posterior(Z_eval)
        mu_i = post.mean.squeeze(-1)                        # (N,)
        var_i = post.variance.clamp_min(_EPS).squeeze(-1)  # (N,)
        sigma_i = var_i.sqrt()                              # (N,)
        z_i = (mu_i - tau_i) / sigma_i                     # (N,)
        phi_z = torch.exp(_STDNORM.log_prob(z_i))          # phi(Z_i)
        Phi_z = _STDNORM.cdf(z_i)                          # Phi(Z_i)
        d_i = z_i * Phi_z + phi_z                          # EDIG depth per obj
        depth = depth * (d_i / (1.0 + d_i))                # rational squash
    return depth


@register_quality("ndig_no_infogain")
def _build_ndig_no_infogain(
    models: Sequence[SingleTaskGP],
    bounds: torch.Tensor,
    tau: torch.Tensor,
    **_: object,
) -> tuple[QualityFn, dict]:
    """NDIG-B ablation provider: normalized depth alone, no info-gain term."""

    def q(z: torch.Tensor) -> torch.Tensor:
        return ndig_no_infogain_quality(models, z, tau)

    info = {"quality": "ndig_no_infogain", "ref_set_size": 0, "cold_start": False}
    return q, info


def ndig_pof_entropy_quality(
    models: Sequence[SingleTaskGP],
    Z_eval: torch.Tensor,
    h: torch.Tensor,
) -> torch.Tensor:
    """Differentiable binary entropy of the joint Probability of Feasibility.

    Ablation of NDIG-B's "binary entropy of PoF" component: replaces the
    depth * info-gain product entirely with the base-2 binary entropy of the
    joint PoF, so the score is highest exactly where PoF is least certain
    (near the feasibility boundary) rather than deep in the interior::

        p_i(z) = Phi((mu_{t,i}(z) - tau_i) / sigma_{t,i}(z))
        p(z)   = prod_i p_i(z)
        q(z)   = -[p(z) log2 p(z) + (1-p(z)) log2(1-p(z))]

    Returns ``(N,)``.
    """
    mus, sigmas = [], []
    for gp in models:
        post = gp.posterior(Z_eval)
        mu_i = post.mean.squeeze(-1)
        var_i = post.variance.clamp_min(_EPS).squeeze(-1)
        mus.append(mu_i)
        sigmas.append(var_i.sqrt())
    mu = torch.stack(mus, dim=-1)       # (N, m)
    sigma = torch.stack(sigmas, dim=-1)  # (N, m)
    pof = joint_feasibility_probability(feasibility_probabilities(mu, sigma, h))
    return _binary_entropy(pof)


@register_quality("ndig_pof_entropy")
def _build_ndig_pof_entropy(
    models: Sequence[SingleTaskGP],
    bounds: torch.Tensor,
    tau: torch.Tensor,
    **_: object,
) -> tuple[QualityFn, dict]:
    """NDIG-B ablation provider: binary entropy of joint PoF, no depth/info-gain."""

    def q(z: torch.Tensor) -> torch.Tensor:
        return ndig_pof_entropy_quality(models, z, tau)

    info = {"quality": "ndig_pof_entropy", "ref_set_size": 0, "cold_start": False}
    return q, info


# ---------------------------------------------------------------------------
# CR-NDIG: Context-Repulsive NDIG (purely sequential, no QD-DPP)
# ---------------------------------------------------------------------------
def cr_ndig_quality(
    models: Sequence[SingleTaskGP],
    Z_eval: torch.Tensor,
    h: torch.Tensor,
    prev_ctx: Optional[torch.Tensor],
    context_dims: Sequence[int],
    lam_ctx: Optional[float] = None,
) -> torch.Tensor:
    """Differentiable CR-NDIG quality (contexts/sequential_ndig.md).

    Multiplies the plain NDIG score by a context-repulsion penalty against
    every previously *evaluated* context ``c_1, ..., c_{t-1}``::

        q(x, c) = q_NDIG(x, c) * prod_{i=1}^{t-1} (1 - k_ctx(c, c_i))

    with ``k_ctx`` an RBF kernel on the context subspace only. This is the
    sole context-space exploration pressure available to purely sequential
    methods (batch_size always 1, so QD-DPP's context kernel never applies).

    If there are no context dimensions or no prior observations, the
    repulsion concept is degenerate/inapplicable and this returns the plain
    NDIG score unchanged -- NOT a trivially-constant context (which would
    incorrectly zero out every candidate via ``1 - k_ctx(c, c) == 0``).

    The product is accumulated in log-space for numerical stability::

        log_repulsion = sum_i log1p(-k_ctx(c, c_i))
        q(x, c)       = q_NDIG(x, c) * exp(log_repulsion)

    Kept fully differentiable (no ``torch.no_grad()``) since this is
    optimized via ``multistart_ascent``'s autograd.

    Returns ``(N,)``.
    """
    base = ndig_quality(models, Z_eval, h)
    if prev_ctx is None or prev_ctx.shape[0] == 0 or not context_dims:
        return base

    ctx_eval = Z_eval[:, list(context_dims)]                  # (N, d_c)
    lam = lam_ctx if lam_ctx is not None else median_heuristic_lambda(prev_ctx)
    sq = torch.cdist(ctx_eval, prev_ctx).pow(2)               # (N, t-1)
    k_ctx = torch.exp(-sq / (2.0 * lam * lam))                # (N, t-1)
    log_repulsion = torch.log1p(-k_ctx.clamp(max=1.0 - 1e-6)).sum(dim=-1)  # (N,)
    return base * torch.exp(log_repulsion)


@register_quality("cr_ndig")
def _build_cr_ndig(
    models: Sequence[SingleTaskGP],
    bounds: torch.Tensor,
    tau: torch.Tensor,
    *,
    context_dims: Sequence[int] = (),
    dpp_lambda_ctx: Optional[float] = None,
    X_obs: Optional[torch.Tensor] = None,
    **_: object,
) -> tuple[QualityFn, dict]:
    """CR-NDIG provider: NDIG penalized by repulsion against every previously
    evaluated context (``X_obs``); no reference set, no QD-DPP (sequential-only)."""
    prev_ctx = (
        X_obs[:, list(context_dims)].detach()
        if (context_dims and X_obs is not None and X_obs.shape[0] > 0)
        else None
    )

    def q(z: torch.Tensor) -> torch.Tensor:
        return cr_ndig_quality(models, z, tau, prev_ctx, context_dims, lam_ctx=dpp_lambda_ctx)

    info = {
        "quality": "cr_ndig",
        "ref_set_size": 0,
        "cold_start": False,
        "n_prev_contexts": 0 if prev_ctx is None else int(prev_ctx.shape[0]),
    }
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
    context_dims=(),     # previously silently discarded; now forwarded to build_reference_set
    **_: object,
) -> tuple[QualityFn, dict]:
    """ROI-MI provider: build the continuous reference set, then score against it.

    Imported lazily to avoid a circular import with ``continuous`` (which imports
    this registry).  ``context_dims`` is forwarded so ``build_reference_set`` can
    stratify the Sobol pool over the context sub-space (Step 1 of the spec).
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
        context_dims=context_dims,
    )

    def q(z: torch.Tensor) -> torch.Tensor:
        return roi_mi_quality_continuous(models, z, Z_ref, tau)

    info = {**info, "quality": "roi_mi"}
    return q, info


# ---------------------------------------------------------------------------
# C2LSE: Confidence-based Continuous LSE (Family B baseline)
# ---------------------------------------------------------------------------
def c2lse_quality(
    models: Sequence[SingleTaskGP],
    Z_eval: torch.Tensor,
    tau: torch.Tensor,
    eps: float = 1e-2,
    active_obj: int = 0,
) -> torch.Tensor:
    """Differentiable C2LSE quality (single-objective LSE, round-robin extension).

    Confidence-based LSE score ``a(z) = sigma_i(z) / max(eps, |mu_i(z) -
    tau_i|)`` evaluated on *only* the active objective ``i = active_obj``
    (round-robin alternation across pipeline iterations; see the module
    docstring and ``_build_c2lse``). Highest where the GP for that single
    objective is least confident about its classification against its
    threshold.

    Returns ``(N,)``, non-negative and differentiable w.r.t. ``Z_eval``.
    """
    gp = models[active_obj]
    tau_i = tau[active_obj]
    post = gp.posterior(Z_eval)
    mu_i = post.mean.squeeze(-1)                       # (N,)
    sigma_i = post.variance.clamp_min(_EPS).sqrt().squeeze(-1)  # (N,)
    margin = (mu_i - tau_i).abs().clamp_min(eps)
    return sigma_i / margin


@register_quality("c2lse")
def _build_c2lse(
    models: Sequence[SingleTaskGP],
    bounds: torch.Tensor,
    tau: torch.Tensor,
    *,
    t: int = 0,
    **_: object,
) -> tuple[QualityFn, dict]:
    """C2LSE provider: no reference set, pure local confidence score on the
    round-robin active objective ``active_obj = t % m``."""
    active_obj = t % len(models)

    def q(z: torch.Tensor) -> torch.Tensor:
        return c2lse_quality(models, z, tau, active_obj=active_obj)

    info = {
        "quality": "c2lse",
        "ref_set_size": 0,
        "cold_start": False,
        "active_obj": int(active_obj),
    }
    return q, info


# ---------------------------------------------------------------------------
# BES: Binary Entropy Search (Family B baseline)
# ---------------------------------------------------------------------------
# Fixed Gauss-Hermite quadrature nodes/weights (constants -> plain numpy is
# fine; only the affine transform of the roots into y_x samples needs to stay
# in torch/autograd, see bes_quality below).
_GH_DEGREE = 20
_GH_NODES_NP, _GH_WEIGHTS_NP = np.polynomial.hermite.hermgauss(_GH_DEGREE)
_LOG_SQRT_PI = 0.5 * math.log(math.pi)


def _bes_gh_tensors(dtype: torch.dtype, device: torch.device):
    nodes = torch.as_tensor(_GH_NODES_NP, dtype=dtype, device=device)
    weights = torch.as_tensor(_GH_WEIGHTS_NP, dtype=dtype, device=device)
    return nodes, weights


def _safe_ndtr(x: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF, clamped away from {0, 1} to avoid log(0)."""
    return torch.special.ndtr(x).clamp(1e-12, 1.0 - 1e-12)


def bes_quality(
    models: Sequence[SingleTaskGP],
    Z_eval: torch.Tensor,
    tau: torch.Tensor,
    gh_degree: int = _GH_DEGREE,
    active_obj: int = 0,
) -> torch.Tensor:
    """Differentiable BES (Binary Entropy Search) quality, per contexts/fair_lse_comparison.md §2.2.

    Single-objective LSE formula, round-robin extension: evaluated on *only*
    the active objective ``i = active_obj`` (see the module docstring and
    ``_build_bes``)::

        p(y_x | y_D)  = N(mu_i(x), sigma_i(x)^2 + sigma_n_i^2)
        sigma_+       = sqrt(sigma_i(x)^2 + sigma_n_i^2)
        h_x(tau_i)    = (tau_i - mu_i(x)) / sigma_i(x)
        g_x(y_x,tau_i)= (sigma_+^2 tau_i - sigma_n_i^2 mu_i(x) - sigma_i(x)^2 y_x)
                        / (sigma_i(x) sigma_n_i sigma_+)
        alpha_i(x)    = E_{y_x ~ p(y_x|y_D)} [
                            sum_{g in {-1,1}} Phi(g * g_x) log( Phi(g*g_x) / Phi(g*h_x) )
                        ]

    The expectation over the 1D Gaussian ``p(y_x|y_D)`` is evaluated via
    fixed-node Gauss-Hermite quadrature (``numpy.polynomial.hermite.hermgauss``
    supplies the constant roots/weights; the affine map of the roots into
    ``y_x`` samples happens in torch so the whole expression stays
    differentiable w.r.t. ``z`` through ``mu_i``/``sigma_i``).

    Returns ``(N,)``, differentiable w.r.t. ``Z_eval``.
    """
    N = Z_eval.shape[0]
    nodes, weights = _bes_gh_tensors(Z_eval.dtype, Z_eval.device)  # (K,), (K,)
    sqrt2 = math.sqrt(2.0)

    gp = models[active_obj]
    tau_i = tau[active_obj]
    post = gp.posterior(Z_eval)
    mu_i = post.mean.squeeze(-1)                                  # (N,)
    sigma_i = post.variance.clamp_min(_EPS).sqrt().squeeze(-1)     # (N,)
    noise_var_i = observation_noise(gp)                            # scalar
    sigma_n_i = noise_var_i.clamp_min(_EPS).sqrt()
    sigma_plus = (sigma_i.pow(2) + noise_var_i).clamp_min(_EPS).sqrt()  # (N,)

    h_x = (tau_i - mu_i) / sigma_i.clamp_min(_EPS)                 # (N,)

    # Affine map of the fixed GH roots into y_x samples (differentiable
    # w.r.t. mu_i/sigma_plus, hence w.r.t. z): y_x = mu_i + sqrt(2)*sigma_+*root.
    y_x = mu_i.unsqueeze(-1) + sqrt2 * sigma_plus.unsqueeze(-1) * nodes.unsqueeze(0)  # (N, K)

    denom = (sigma_i * sigma_n_i * sigma_plus).clamp_min(_EPS)     # (N,)
    g_x = (
        sigma_plus.pow(2).unsqueeze(-1) * tau_i
        - noise_var_i * mu_i.unsqueeze(-1)
        - sigma_i.pow(2).unsqueeze(-1) * y_x
    ) / denom.unsqueeze(-1)                                        # (N, K)

    # Integrand: sum_{g in {-1,1}} Phi(g*g_x) log(Phi(g*g_x)/Phi(g*h_x)).
    Phi_gx_pos = _safe_ndtr(g_x)
    Phi_gx_neg = _safe_ndtr(-g_x)
    Phi_hx_pos = _safe_ndtr(h_x).unsqueeze(-1)
    Phi_hx_neg = _safe_ndtr(-h_x).unsqueeze(-1)

    integrand = (
        Phi_gx_pos * (Phi_gx_pos.log() - Phi_hx_pos.log())
        + Phi_gx_neg * (Phi_gx_neg.log() - Phi_hx_neg.log())
    )  # (N, K)

    # Gauss-Hermite quadrature: E_{x~N(m,v)}[f(x)] ~ (1/sqrt(pi)) sum_k w_k f(m + sqrt(2v) root_k).
    alpha_i = (integrand * weights.unsqueeze(0)).sum(dim=-1) / math.sqrt(math.pi)  # (N,)

    return alpha_i


@register_quality("bes")
def _build_bes(
    models: Sequence[SingleTaskGP],
    bounds: torch.Tensor,
    tau: torch.Tensor,
    *,
    t: int = 0,
    **_: object,
) -> tuple[QualityFn, dict]:
    """BES provider: no reference set, Gauss-Hermite quadrature on the
    round-robin active objective ``active_obj = t % m``."""
    active_obj = t % len(models)

    def q(z: torch.Tensor) -> torch.Tensor:
        return bes_quality(models, z, tau, active_obj=active_obj)

    info = {
        "quality": "bes",
        "ref_set_size": 0,
        "cold_start": False,
        "active_obj": int(active_obj),
    }
    return q, info


# ---------------------------------------------------------------------------
# Interior Sampling: Family-C Stage 2 ("Interior") quality
# ---------------------------------------------------------------------------
_INTERIOR_POF_TARGET = 0.95  # per contexts/fair_lse_comparison.md Mechanism 3
_INTERIOR_LAMBDA_PENALTY_DEFAULT = 75.0  # dominates typical sigma_combined magnitudes


def sigma_combined(models: Sequence[SingleTaskGP], Z_eval: torch.Tensor) -> torch.Tensor:
    """Combined per-objective uncertainty ``sqrt(sum_i sigma_i(z)^2)``.

    The codebase has no existing convention for aggregating ``m`` independent
    GPs' posterior std devs into one scalar (the plan doc's ``gamma*sigma(x+)``
    term is written for a single-output problem). We use the L2 magnitude of
    the per-objective posterior std vector -- a natural, scale-respecting
    generalization: it grows with any objective's uncertainty and reduces to
    the single-output ``sigma(x+)`` for ``m=1``. See module docs / final
    report for the full justification.

    Returns ``(N,)``, non-negative and differentiable w.r.t. ``Z_eval``.
    """
    sq = torch.zeros(Z_eval.shape[0], dtype=Z_eval.dtype, device=Z_eval.device)
    for gp in models:
        post = gp.posterior(Z_eval)
        var_i = post.variance.clamp_min(_EPS).squeeze(-1)
        sq = sq + var_i
    return sq.clamp_min(0.0).sqrt()


def interior_sampling_quality(
    models: Sequence[SingleTaskGP],
    Z_eval: torch.Tensor,
    tau: torch.Tensor,
    lambda_penalty: float = _INTERIOR_LAMBDA_PENALTY_DEFAULT,
    pof_target: float = _INTERIOR_POF_TARGET,
) -> torch.Tensor:
    """Differentiable Stage-2 "Interior" quality (Family C, Mechanism 3).

    Pure combined-uncertainty exploration, soft-penalized to stay inside the
    predicted feasible interior (joint PoF >= ``pof_target``, default 0.95)::

        raw(z) = sigma_combined(z) - lambda_penalty * relu(pof_target - PoF(z))
        q(z)   = softplus(raw(z))

    ``lambda_penalty`` is fixed large enough (default 75) to dominate typical
    ``sigma_combined`` magnitudes so ``raw(z)`` is strongly negative whenever
    PoF(z) < 0.95, pulling the ascent back inside the feasible interior;
    empirically verified in ``tests/test_smoke.py`` and the Phase-2 smoke
    runs (see final report). This is a soft-constraint relaxation of the plan
    doc's literal hard constraint (BoTorch ``optimize_acqf`` + ``nonlinear_
    inequality_constraints``) so Stage 2 can reuse the same unconstrained
    ``multistart_ascent`` machinery as every other continuous quality variant
    rather than standing up a second, parallel constrained-optimization stack.

    The outer ``softplus`` (rather than a hard ``clamp_min(0)``) is required
    -- not merely stylistic -- for correctness under gradient ascent: every
    other quality variant in this module is non-negative by construction
    (PoF/probability products, log1p of a non-negative ratio, ...), but the
    raw penalized score here is frequently very negative outside the
    feasible interior. If it were left unclamped, `multistart_ascent` and
    the QD-DPP `_qd_marginal_gain` machinery it feeds (which itself applies
    `clamp_min(0)` to the quality *before* squaring into the L-ensemble
    diagonal, see `continuous._qd_marginal_gain`) would see an all-zero
    gradient for any restart that starts with `raw(z) < 0` -- `d/dz
    clamp_min(0, x) = 0` for `x < 0` -- silently stranding restarts far from
    the feasible region instead of pulling them toward it. Softplus keeps
    the composed objective smooth and strictly positive everywhere (so
    gradients never vanish) while still being monotonic in ``raw(z)``, so
    the argmax over restarts is unaffected. Confirmed empirically (Phase-2
    final report): with a hard `clamp_min(0)` instead of `softplus`, roughly
    half of `multistart_ascent`'s 16 random restarts on `two_circles_2d`
    never left a zero-gradient region and the *DPP-selected batch* itself
    could include points with PoF as low as 0.02. Under `softplus`, the
    best-scoring restart (and every point `select_batch_continuous` actually
    selects, since it always ranks by quality / QD-DPP marginal gain, never
    picks a random restart) reliably satisfies PoF>=0.95, and a live 5-point
    DPP batch on the same problem/seed had 100% of its selections clear the
    gate (vs. only 3/5 under the hard-clamp variant).

    Returns ``(N,)``, non-negative and differentiable w.r.t. ``Z_eval``.
    """
    mus, sigmas = [], []
    for gp in models:
        post = gp.posterior(Z_eval)
        mu_i = post.mean.squeeze(-1)
        var_i = post.variance.clamp_min(_EPS).squeeze(-1)
        mus.append(mu_i)
        sigmas.append(var_i.sqrt())
    mu = torch.stack(mus, dim=-1)       # (N, m)
    sigma = torch.stack(sigmas, dim=-1)  # (N, m)

    sigma_comb = sigma.pow(2).sum(dim=-1).clamp_min(0.0).sqrt()  # (N,)
    pof = joint_feasibility_probability(feasibility_probabilities(mu, sigma, tau))  # (N,)
    penalty = (pof_target - pof).clamp_min(0.0)  # relu(0.95 - PoF)
    raw = sigma_comb - lambda_penalty * penalty
    return torch.nn.functional.softplus(raw)


@register_quality("interior_sampling")
def _build_interior_sampling(
    models: Sequence[SingleTaskGP],
    bounds: torch.Tensor,
    tau: torch.Tensor,
    **_: object,
) -> tuple[QualityFn, dict]:
    """Interior-sampling provider: no reference set, local penalized combined-uncertainty."""

    def q(z: torch.Tensor) -> torch.Tensor:
        return interior_sampling_quality(models, z, tau)

    info = {"quality": "interior_sampling", "ref_set_size": 0, "cold_start": False}
    return q, info
