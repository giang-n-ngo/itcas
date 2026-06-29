"""GP model construction and posterior helpers (BoTorch / GPyTorch)."""
from __future__ import annotations

from typing import Optional

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood


def build_independent_gps(
    X: torch.Tensor,
    Y: torch.Tensor,
    bounds: Optional[torch.Tensor] = None,
) -> list[SingleTaskGP]:
    """Build one independent SingleTaskGP per output dimension.

    Args:
        X: (N, d) tensor of inputs.
        Y: (N, m) tensor of outputs (one column per objective).
        bounds: optional (2, d) bounds for input normalization.

    Returns:
        List of m fitted SingleTaskGP models.
    """
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError(f"Expected 2D X and Y; got X={X.shape}, Y={Y.shape}")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of rows.")

    d = X.shape[-1]
    m = Y.shape[-1]
    models: list[SingleTaskGP] = []
    for i in range(m):
        Yi = Y[:, i : i + 1]
        input_tf = Normalize(d=d, bounds=bounds) if bounds is not None else None
        gp = SingleTaskGP(
            train_X=X,
            train_Y=Yi,
            input_transform=input_tf,
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        gp.eval()
        models.append(gp)
    return models


def posterior_mean_std(
    models: list[SingleTaskGP], X: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-objective posterior mean and std at X.

    Args:
        models: list of m SingleTaskGPs.
        X: (N, d) tensor.

    Returns:
        mu: (N, m), sigma: (N, m).
    """
    mus, sigmas = [], []
    with torch.no_grad():
        for gp in models:
            post = gp.posterior(X)
            mus.append(post.mean.squeeze(-1))
            sigmas.append(post.variance.clamp_min(1e-12).sqrt().squeeze(-1))
    return torch.stack(mus, dim=-1), torch.stack(sigmas, dim=-1)


def observation_noise(gp: SingleTaskGP) -> torch.Tensor:
    """Observation-noise variance ``sigma_eps^2`` in the ORIGINAL output space.

    ``gp.likelihood.noise`` lives in the *standardized* output space (the GP is
    fit with a ``Standardize`` outcome transform), whereas ``posterior().variance``
    is reported back in the original space. To combine the two consistently --
    e.g. in the information-gain term ``log(1 + sigma_f^2 / sigma_eps^2)`` -- the
    noise must be rescaled by the standardizer variance ``stdvs^2``.

    Returns a scalar tensor.
    """
    noise = gp.likelihood.noise.mean().clamp_min(1e-8)
    octf = getattr(gp, "outcome_transform", None)
    if octf is not None and hasattr(octf, "stdvs"):
        scale = (octf.stdvs.reshape(-1) ** 2).mean().clamp_min(1e-12)
        noise = noise * scale
    return noise


def joint_posterior_sample(
    models: list[SingleTaskGP],
    X: torch.Tensor,
    num_samples: int = 1,
) -> torch.Tensor:
    """Draw joint posterior samples per objective at X.

    Returns (num_samples, N, m).
    """
    samples = []
    with torch.no_grad():
        for gp in models:
            post = gp.posterior(X)
            s = post.rsample(sample_shape=torch.Size([num_samples]))
            samples.append(s.squeeze(-1))
    # each: (S, N) -> stack -> (S, N, m)
    return torch.stack(samples, dim=-1)
