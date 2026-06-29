"""ROI-MI: Region-of-Interest Mutual Information acquisition.

Implements the proposal in `methodology.tex`:

  Step 1 (Thompson sampling the interior):
      f^(s) ~ GP(mu_t, Sigma_t) drawn jointly across a candidate pool.
      Z_ref = { z in Z_cand : f^(s)_i(z) > h_i, for all i }

  Step 2 (Marginal information gain):
      q(z) = I( y(z) ; Y(Z_ref) | D_t )
           = H(Y(Z_ref) | D_t)
             - E_{y(z)}[ H(Y(Z_ref) | D_t cup {z, y(z)}) ]

We approximate the per-objective binary feasibility probability
    p_i(z) := P(f_i(z) > h_i | D_t) = Phi( (mu_i - h_i) / sigma_i )
and treat the joint feasibility probability across objectives
    p(z) = prod_i p_i(z)
as a Bernoulli variable Y(z), assuming approximate conditional independence
across objectives (standard CAS-style assumption).

H(Y(Z_ref)) is approximated by the sum of per-point binary entropies (the
DPP/diversity term in the next module compensates for inter-point correlation).
After "hallucinating" an observation at z, we update each per-objective GP via
the closed-form rank-1 update of the predictive mean/variance.
"""
from __future__ import annotations

from typing import Optional

import math
import torch
from botorch.models import SingleTaskGP

from ..utils.gp import joint_posterior_sample, posterior_mean_std

_LOG2 = math.log(2.0)
_SQRT2 = math.sqrt(2.0)


def _binary_entropy(p: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    one_minus = (1.0 - p).clamp_min(eps)
    return -(p * p.log() + one_minus * one_minus.log()) / _LOG2


def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / _SQRT2))


def feasibility_probabilities(
    mu: torch.Tensor, sigma: torch.Tensor, h: torch.Tensor
) -> torch.Tensor:
    """Per-objective P(f_i > h_i) at each candidate.

    mu, sigma: (N, m)
    h: (m,)
    Returns (N, m).
    """
    return _normal_cdf((mu - h) / sigma.clamp_min(1e-12))


def joint_feasibility_probability(p_per_obj: torch.Tensor) -> torch.Tensor:
    """Independence approximation: prod_i p_i. Input (N, m) -> (N,)."""
    return p_per_obj.clamp(1e-12, 1.0 - 1e-12).prod(dim=-1)


def thompson_reference_set(
    models: list[SingleTaskGP],
    cand: torch.Tensor,
    h: torch.Tensor,
    rng_seed: Optional[int] = None,
) -> torch.Tensor:
    """Draw one joint posterior sample per objective and return the boolean
    mask over `cand` of points that satisfy all thresholds under that draw.

    Returns: (N,) boolean tensor.
    """
    if rng_seed is not None:
        torch.manual_seed(rng_seed)
    sample = joint_posterior_sample(models, cand, num_samples=1).squeeze(0)  # (N, m)
    return (sample > h).all(dim=-1)


def conditional_variance_after_obs(
    gp: SingleTaskGP,
    Z_ref: torch.Tensor,
    z_query: torch.Tensor,
) -> torch.Tensor:
    """Posterior variance of GP at Z_ref AFTER (hallucinated) observing
    a single point z_query, using the closed-form predictive update:

        sigma_new^2(zr) = sigma^2(zr) - k(zr, z_q)^2 / (sigma^2(z_q) + sigma_n^2)

    Returns (|Z_ref|,) tensor of post-observation variances.
    """
    z_query = z_query.unsqueeze(0) if z_query.ndim == 1 else z_query
    with torch.no_grad():
        post_zr = gp.posterior(Z_ref)
        sig2_zr = post_zr.variance.squeeze(-1).clamp_min(1e-12)  # (N,)

        post_zq = gp.posterior(z_query)
        sig2_zq = post_zq.variance.squeeze(-1).clamp_min(1e-12)  # (1,)

        # Unwhitened cross-covariance via the underlying kernel on the (possibly
        # transformed) inputs. SingleTaskGP applies its input_transform inside
        # forward, but here we call covar_module directly on raw inputs to mirror
        # how posterior() forms K(z, z'). For simplicity we fall back to using
        # the "double-posterior" identity:
        #   k(zr, zq) = E[ (f(zr)-mu(zr))(f(zq)-mu(zq)) ]
        # which is computed by querying the joint posterior over [zr; zq].
        joint = torch.cat([Z_ref, z_query], dim=0)
        post_joint = gp.posterior(joint)
        cov_full = post_joint.mvn.covariance_matrix  # (N+1, N+1)
        n_ref = Z_ref.shape[0]
        k_cross = cov_full[:n_ref, n_ref:].squeeze(-1)  # (N,)

        try:
            noise = gp.likelihood.noise.mean().clamp_min(1e-8)
        except Exception:
            noise = torch.tensor(1e-6, dtype=Z_ref.dtype, device=Z_ref.device)

        denom = sig2_zq + noise
        sig2_new = (sig2_zr - k_cross.pow(2) / denom).clamp_min(1e-12)
    return sig2_new


def roi_mi_quality(
    models: list[SingleTaskGP],
    cand: torch.Tensor,
    h: torch.Tensor,
    n_samples: int = 1,
    rng_seed: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the ROI-MI quality q(z) for every candidate in `cand`.

    Returns (q, ref_mask):
        q: (N,) marginal information gain per candidate
        ref_mask: (N,) bool mask of points belonging to Z_ref (last sample)
    """
    mu, sigma = posterior_mean_std(models, cand)  # (N, m)
    p_per_obj = feasibility_probabilities(mu, sigma, h)  # (N, m)

    N = cand.shape[0]
    q_total = torch.zeros(N, dtype=cand.dtype, device=cand.device)
    last_mask = torch.zeros(N, dtype=torch.bool, device=cand.device)

    for s in range(n_samples):
        seed_s = None if rng_seed is None else rng_seed + s
        ref_mask = thompson_reference_set(models, cand, h, rng_seed=seed_s)
        last_mask = ref_mask
        if not bool(ref_mask.any()):
            continue
        Z_ref = cand[ref_mask]  # (R, d)

        # Current joint entropy proxy: sum of binary entropies of p(zr)
        p_ref = joint_feasibility_probability(p_per_obj[ref_mask])  # (R,)
        H_now = _binary_entropy(p_ref).sum()  # scalar

        # For each candidate z, compute the expected entropy after observing z.
        # We update each per-objective GP variance via closed form (mean update
        # is averaged out by E_{y(z)}, so does not affect E[H] under the
        # Bernoulli/Gaussian approximation we use below).
        for j in range(N):
            zq = cand[j : j + 1]
            new_p_per_obj = []
            for i, gp in enumerate(models):
                sig2_new = conditional_variance_after_obs(gp, Z_ref, zq)  # (R,)
                # Mean update under E_{y(z)} preserves the prior mean in
                # expectation over noisy obs, so we keep mu and shrink sigma:
                mu_i_ref = mu[ref_mask, i]
                sig_i_new = sig2_new.sqrt()
                new_p_per_obj.append(
                    _normal_cdf((mu_i_ref - h[i]) / sig_i_new.clamp_min(1e-12))
                )
            new_p = torch.stack(new_p_per_obj, dim=-1).clamp(1e-12, 1.0 - 1e-12).prod(
                dim=-1
            )
            H_after = _binary_entropy(new_p).sum()
            q_total[j] += (H_now - H_after).clamp_min(0.0)

    q_total = q_total / max(n_samples, 1)

    # Degenerate ROI: under tight thresholds the Thompson reference set is
    # often empty for every sample, leaving q == 0 everywhere. With no quality
    # signal the QD-DPP returns an empty batch and the search would stall. Fall
    # back to the joint feasibility probability p(z) = prod_i P(f_i > h_i), so
    # the policy keeps exploring toward the predicted satisfactory region (the
    # CAS objective) instead of terminating early.
    if not bool((q_total > 0).any()):
        q_total = joint_feasibility_probability(p_per_obj)

    return q_total, last_mask
