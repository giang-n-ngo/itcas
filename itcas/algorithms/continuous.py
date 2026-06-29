"""Continuous C-MO-CAS acquisition (revised ``methodology.tex``).

This module implements the *continuous* proposed method end-to-end, replacing
the earlier discrete candidate-pool approximation:

Step 1 - **Differentiable Thompson sampling.** A joint posterior realization
    ``f^(s) ~ GP(mu_t, Sigma_t)`` is drawn with Random Fourier Features
    (BoTorch Matheron pathwise sampling). The draw is a deterministic, fully
    differentiable function of the joint input-context point ``z = (x, c)``.

Step 2 - **Smooth margin function.** Feasibility depth is measured by the
    smooth (negative LogSumExp / softmin) margin

        M^(s)(z) = -gamma * log sum_i exp( -(f_i^(s)(z) - tau_i) / gamma )

    which lower-bounds ``min_i (f_i - tau_i)`` and is differentiable, so
    ``M^(s)(z) > 0`` certifies strict feasibility under the hallucinated draw.

Step 3 - **Multi-start optimization and Z_ref construction.** A continuous
    multi-start gradient-ascent optimizer locates the local maxima
    ``Z_opt = argmax M^(s)`` over ``X x C``. The reference set is then built
    *without* relaxation hyperparameters:

        - if ``max M^(s) > 0``  ->  Z_ref = { z in Z_opt : M^(s)(z) > 0 }   (deep interior)
        - else (cold start)     ->  Z_ref = argmax_z M^(s)(z)               (closest peaks)

Step 4 - **Marginal information gain.** The ROI-MI quality of any continuous
    candidate ``z`` is the mutual information between its (noisy) observation and
    the *continuous objective values* of the reference set,
    ``q(z) = I(y(z); f(Z_ref) | D_t)`` (see :func:`roi_mi_quality_continuous`).
    Targeting the exact performance metrics ``f(Z_ref)`` -- rather than the binary
    feasibility of ``Z_ref`` -- natively drives objective-space diversity inside
    the safe interior.

Step 5 - **Continuous submodular maximization (QD-DPP).** The batch is built
    greedily; at each step a continuous multi-start gradient ascent over
    ``X x C`` selects the point of maximal marginal gain of
    ``F(B) = log det(I + L_B)`` with the Quality-Diversity L-ensemble

        L_ij = q(z_i) [ k_obj(mu_i, mu_j) k_ctx(c_i, c_j) ] q(z_j).

    Monotone submodularity yields the ``(1 - 1/e)`` guarantee.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
from botorch.models import SingleTaskGP
from botorch.sampling.pathwise import draw_matheron_paths

from ..utils.gp import posterior_mean_std, observation_noise
from .qd_dpp import median_heuristic_lambda
from .quality import build_quality_fn
from .roi_mi import (
    feasibility_probabilities,
    joint_feasibility_probability,
)

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Step 1-2: differentiable Thompson sampling + smooth margin
# ---------------------------------------------------------------------------
def draw_objective_paths(
    models: Sequence[SingleTaskGP], n_samples: int = 1
):
    """Draw ``n_samples`` joint RFF posterior paths, one set per objective.

    Returns a list of ``MatheronPath`` callables (length ``m``). Each path maps
    ``z`` of shape ``(N, d)`` to a differentiable sample of shape
    ``(n_samples, N)`` in the *original* (untransformed) objective space.
    """
    shape = torch.Size([n_samples])
    with torch.no_grad():
        return [draw_matheron_paths(gp, sample_shape=shape) for gp in models]


def evaluate_paths(paths, z: torch.Tensor) -> torch.Tensor:
    """Evaluate all per-objective paths at ``z``.

    Args:
        paths: list of ``m`` Matheron paths (each maps ``(N, d) -> (S, N)``).
        z: ``(N, d)`` query points.

    Returns: ``(S, N, m)`` differentiable sample tensor.
    """
    cols = [p(z) for p in paths]  # each (S, N)
    return torch.stack(cols, dim=-1)  # (S, N, m)


def smooth_margin(f: torch.Tensor, tau: torch.Tensor, gamma: float) -> torch.Tensor:
    """Smooth softmin margin ``M = -gamma log sum_i exp(-(f_i - tau_i)/gamma)``.

    Args:
        f: ``(..., m)`` (sampled) objective values.
        tau: ``(m,)`` feasibility thresholds.
        gamma: positive smoothing constant.

    Returns: ``(...)`` margin. ``M > 0`` implies ``f_i > tau_i`` for all ``i``.
    """
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    return -gamma * torch.logsumexp(-(f - tau) / gamma, dim=-1)


# ---------------------------------------------------------------------------
# Multi-start continuous optimizer
# ---------------------------------------------------------------------------
def _uniform_starts(
    bounds: torch.Tensor, n: int, seed: Optional[int]
) -> torch.Tensor:
    lo, hi = bounds[0], bounds[1]
    d = bounds.shape[-1]
    g = torch.Generator().manual_seed(int(seed)) if seed is not None else None
    u = torch.rand(n, d, generator=g)
    return (lo.cpu() + (hi.cpu() - lo.cpu()) * u).to(device=bounds.device, dtype=bounds.dtype)


def multistart_ascent(
    objective: Callable[[torch.Tensor], torch.Tensor],
    bounds: torch.Tensor,
    n_restarts: int,
    n_steps: int,
    lr: float = 0.05,
    seed: Optional[int] = None,
    starts: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Projected-gradient (Adam) multi-start maximization of ``objective``.

    Args:
        objective: maps ``(N, d) -> (N,)`` and is differentiable w.r.t. the input.
        bounds: ``(2, d)`` box constraints; iterates are clamped each step.
        n_restarts: number of random restarts (ignored if ``starts`` given).
        n_steps: gradient steps per restart.
        lr: Adam learning rate.
        seed: RNG seed for the restart initialization.
        starts: optional ``(n_restarts, d)`` initial points.

    Returns: ``(Z_opt, values)`` with the converged points and their objective.
    """
    lo, hi = bounds[0], bounds[1]
    if starts is None:
        starts = _uniform_starts(bounds, n_restarts, seed)
    z = starts.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(max(n_steps, 0)):
        opt.zero_grad(set_to_none=True)
        val = objective(z)
        (-val.sum()).backward()
        opt.step()
        with torch.no_grad():
            z.clamp_(lo, hi)
    with torch.no_grad():
        final = objective(z)
    return z.detach(), final.detach()


def _dedup(points: torch.Tensor, tol: float) -> torch.Tensor:
    """Greedy de-duplication of near-identical points (L2 < tol)."""
    if points.shape[0] <= 1:
        return points
    keep: list[int] = []
    for i in range(points.shape[0]):
        if not keep:
            keep.append(i)
            continue
        d = torch.cdist(points[i : i + 1], points[keep]).min()
        if float(d) > tol:
            keep.append(i)
    return points[keep]


# ---------------------------------------------------------------------------
# Step 3: reference-set construction
# ---------------------------------------------------------------------------
def build_reference_set(
    models: Sequence[SingleTaskGP],
    bounds: torch.Tensor,
    tau: torch.Tensor,
    *,
    gamma: float = 0.1,
    n_restarts: int = 16,
    n_steps: int = 60,
    lr: float = 0.05,
    n_ts_samples: int = 1,
    cold_start_k: int = 4,
    dedup_tol: float = 1e-3,
    rng_seed: Optional[int] = None,
) -> tuple[torch.Tensor, dict]:
    """Construct the continuous reference set ``Z_ref`` (Steps 1-3).

    For each of ``n_ts_samples`` RFF posterior draws we run multi-start gradient
    ascent of the smooth margin and collect the reference points per the
    dynamic feasible / cold-start rule, then take the union across draws.

    Returns ``(Z_ref, info)``. ``Z_ref`` has shape ``(R, d)`` (always non-empty
    as long as the optimizer produces at least one finite point).
    """
    refs: list[torch.Tensor] = []
    n_feasible_samples = 0
    for s in range(max(n_ts_samples, 1)):
        seed_s = None if rng_seed is None else rng_seed + s
        if seed_s is not None:
            torch.manual_seed(seed_s)
        paths = draw_objective_paths(models, n_samples=1)

        def margin_obj(z: torch.Tensor) -> torch.Tensor:
            f = evaluate_paths(paths, z).squeeze(0)  # (N, m)
            return smooth_margin(f, tau, gamma)

        Z_opt, M_opt = multistart_ascent(
            margin_obj, bounds, n_restarts, n_steps, lr=lr, seed=seed_s
        )
        finite = torch.isfinite(M_opt)
        Z_opt, M_opt = Z_opt[finite], M_opt[finite]
        if Z_opt.numel() == 0:
            continue

        max_m = float(M_opt.max())
        if max_m > 0.0:
            # Feasible interior exists: keep strictly-feasible peaks.
            n_feasible_samples += 1
            sel = Z_opt[M_opt > 0.0]
        else:
            # Cold start: target the closest peaks to feasibility (top-k by M).
            k = min(cold_start_k, Z_opt.shape[0])
            top = torch.topk(M_opt, k).indices
            sel = Z_opt[top]
        refs.append(_dedup(sel, dedup_tol))

    if not refs:
        # Degenerate optimizer output: fall back to a single random anchor so
        # the downstream ROI-MI / DPP always has a reference point to work with.
        Z_ref = _uniform_starts(bounds, 1, rng_seed)
    else:
        Z_ref = _dedup(torch.cat(refs, dim=0), dedup_tol)

    info = {
        "ref_set_size": int(Z_ref.shape[0]),
        "n_feasible_samples": n_feasible_samples,
        "cold_start": n_feasible_samples == 0,
    }
    return Z_ref, info


# ---------------------------------------------------------------------------
# Step 4: differentiable ROI-MI quality vs a continuous reference set
# ---------------------------------------------------------------------------
def roi_mi_quality_continuous(
    models: Sequence[SingleTaskGP],
    Z_eval: torch.Tensor,
    Z_ref: torch.Tensor,
    h: torch.Tensor,
) -> torch.Tensor:
    """Differentiable ROI-MI quality ``q(z) = I(y(z); f(Z_ref) | D_t)``.

    Mutual information between the candidate's noisy observation ``y(z)`` and the
    *continuous objective values* ``f(Z_ref)`` of the reference set -- the exact
    performance metrics, **not** their binary feasibility. With ``m`` independent
    GP objectives the MI factorizes into a sum of per-objective Gaussian terms.
    For objective ``i``, the matrix-determinant lemma reduces the rank-1
    posterior down-date of ``Cov(f_i(Z_ref))`` to the closed form

        I_i(z) = -1/2 log( 1 - k_i(z, Z_ref) Sigma_i(Z_ref)^{-1} k_i(Z_ref, z)
                                 / (sigma_{f,i}^2(z) + sigma_{eps,i}^2) )

    which is non-negative and differentiable w.r.t. ``z``. ``q(z) = sum_i I_i(z)``.

    Vectorized over the evaluation points ``Z_eval`` (``(N, d)``) against the
    fixed continuous reference set ``Z_ref`` (``(R, d)``). ``h`` is retained for
    API compatibility and the degenerate (empty ``Z_ref``) fallback only.

    Returns ``(N,)`` and is differentiable w.r.t. ``Z_eval``.
    """
    N = Z_eval.shape[0]
    R = Z_ref.shape[0]
    if R == 0:
        mu, sigma = posterior_mean_std(models, Z_eval)
        return joint_feasibility_probability(feasibility_probabilities(mu, sigma, h))

    d = Z_eval.shape[-1]
    Zr = Z_ref.unsqueeze(0).expand(N, R, d)            # (N, R, d)
    Zq = Z_eval.unsqueeze(1)                           # (N, 1, d)
    joint = torch.cat([Zr, Zq], dim=1)                 # (N, R+1, d)

    eye_R = torch.eye(R, dtype=Z_eval.dtype, device=Z_eval.device)
    q = torch.zeros(N, dtype=Z_eval.dtype, device=Z_eval.device)
    for gp in models:
        post = gp.posterior(joint)
        cov = post.mvn.covariance_matrix               # (N, R+1, R+1)
        sig_ref = cov[:, :R, :R] + _EPS * eye_R        # (N, R, R)
        k_cross = cov[:, :R, R]                         # (N, R)
        sig2_zq = cov[:, R, R].clamp_min(_EPS)          # (N,)
        noise = observation_noise(gp)                   # scalar (original space)

        # quad = k^T Sigma_ref^{-1} k via a stable solve (Z_ref fixed).
        sol = torch.linalg.solve(sig_ref, k_cross.unsqueeze(-1)).squeeze(-1)  # (N, R)
        quad = (k_cross * sol).sum(dim=-1)              # (N,)
        ratio = (quad / (sig2_zq + noise)).clamp(0.0, 1.0 - 1e-6)
        q = q + (-0.5) * torch.log1p(-ratio)            # I_i(z) >= 0
    return q.clamp_min(0.0)


# ---------------------------------------------------------------------------
# Step 5: continuous greedy submodular maximization (QD-DPP)
# ---------------------------------------------------------------------------
def _posterior_mean_grad(
    models: Sequence[SingleTaskGP], X: torch.Tensor
) -> torch.Tensor:
    """Per-objective posterior mean at ``X`` *with* autograd enabled.

    Mirrors :func:`utils.gp.posterior_mean_std` but keeps the graph so the
    objective-space diversity kernel is differentiable w.r.t. ``X`` (the
    methodology requires the marginal gain to be completely differentiable).
    """
    cols = [gp.posterior(X).mean.squeeze(-1) for gp in models]
    return torch.stack(cols, dim=-1)


def _context(z: torch.Tensor, context_dims: Sequence[int]) -> Optional[torch.Tensor]:
    if context_dims is None or len(context_dims) == 0:
        return None
    return z[:, list(context_dims)]


def _qd_marginal_gain(
    qN: torch.Tensor,
    muN: torch.Tensor,
    ctxN: Optional[torch.Tensor],
    qB: Optional[torch.Tensor],
    muB: Optional[torch.Tensor],
    ctxB: Optional[torch.Tensor],
    lam_obj: float,
    lam_ctx: Optional[float],
) -> torch.Tensor:
    """Greedy marginal gain ``log det(I+L_{B u z}) - log det(I+L_B)`` per candidate.

    ``gain(z) = log(1 + L_zz - L_zB (I + L_BB)^-1 L_Bz)`` (Schur complement),
    with ``L_zz = q(z)^2`` since the diversity kernel is unit on the diagonal.
    """
    diag = qN.clamp_min(0.0).pow(2)  # (N,)
    if qB is None or qB.numel() == 0:
        return torch.log1p(diag)

    def cross_kernel(mu_a, ctx_a, mu_b, ctx_b):
        # objective-space RBF between two sets
        sq = torch.cdist(mu_a, mu_b).pow(2)
        k = torch.exp(-sq / (2.0 * lam_obj * lam_obj))
        if ctx_a is not None and ctx_b is not None and lam_ctx is not None:
            sqc = torch.cdist(ctx_a, ctx_b).pow(2)
            k = k * torch.exp(-sqc / (2.0 * lam_ctx * lam_ctx))
        return k

    k_BB = cross_kernel(muB, ctxB, muB, ctxB)               # (k, k)
    L_BB = (qB.unsqueeze(0) * k_BB) * qB.unsqueeze(1)
    M = L_BB + torch.eye(qB.shape[0], dtype=qB.dtype, device=qB.device)
    M_inv = torch.linalg.solve(M, torch.eye(qB.shape[0], dtype=qB.dtype, device=qB.device))

    k_zB = cross_kernel(muN, ctxN, muB, ctxB)               # (N, k)
    L_zB = (qN.unsqueeze(1) * k_zB) * qB.unsqueeze(0)       # (N, k)
    quad = (L_zB @ M_inv * L_zB).sum(dim=-1)                # (N,)
    schur = (diag - quad).clamp_min(0.0)
    return torch.log1p(schur)


def select_batch_continuous(
    models: Sequence[SingleTaskGP],
    bounds: torch.Tensor,
    tau: torch.Tensor,
    batch_size: int = 1,
    context_dims: Sequence[int] = (),
    *,
    quality: str = "roi_mi",
    gamma: float = 0.1,
    n_restarts: int = 16,
    n_opt_steps: int = 60,
    opt_lr: float = 0.05,
    n_ts_samples: int = 1,
    dpp_lambda: Optional[float] = None,
    dpp_lambda_ctx: Optional[float] = None,
    rng_seed: Optional[int] = None,
) -> tuple[torch.Tensor, dict]:
    """Run the full continuous C-MO-CAS acquisition and return the batch.

    Args:
        quality: candidate-quality variant feeding the QD-DPP. One of the names
            in ``quality.QUALITY_REGISTRY`` (e.g. ``"roi_mi"`` or ``"lf_mi"``).
            The DPP diversity machinery is identical across variants; only the
            ``q(z)`` term changes.

    Returns ``(X_new, info)`` where ``X_new`` is a ``(b, d)`` tensor of
    selected design-context points ``z = (x, c)`` in the original domain.
    """
    quality_fn, ref_info = build_quality_fn(
        quality, models, bounds, tau,
        context_dims=context_dims, gamma=gamma, n_restarts=n_restarts,
        n_opt_steps=n_opt_steps, opt_lr=opt_lr, n_ts_samples=n_ts_samples,
        rng_seed=rng_seed,
    )

    # Length-scales for the diversity kernels (median heuristic over a domain
    # pool of predicted objectives / contexts unless explicitly provided).
    pool = _uniform_starts(bounds, max(n_restarts * 4, 32), rng_seed)
    with torch.no_grad():
        mu_pool, _ = posterior_mean_std(models, pool)
    lam_obj = dpp_lambda if dpp_lambda is not None else median_heuristic_lambda(mu_pool)
    ctx_pool = _context(pool, context_dims)
    lam_ctx = None
    if ctx_pool is not None:
        lam_ctx = dpp_lambda_ctx if dpp_lambda_ctx is not None else median_heuristic_lambda(ctx_pool)

    selected: list[torch.Tensor] = []
    qB: Optional[torch.Tensor] = None
    muB: Optional[torch.Tensor] = None
    ctxB: Optional[torch.Tensor] = None

    for k in range(batch_size):
        seed_k = None if rng_seed is None else rng_seed + 100 * (k + 1)

        def gain_obj(z: torch.Tensor) -> torch.Tensor:
            qN = quality_fn(z)
            muN = _posterior_mean_grad(models, z)
            ctxN = _context(z, context_dims)
            return _qd_marginal_gain(
                qN, muN, ctxN, qB, muB, ctxB, lam_obj, lam_ctx
            )

        Z_cand, gains = multistart_ascent(
            gain_obj, bounds, n_restarts, n_opt_steps, lr=opt_lr, seed=seed_k
        )
        finite = torch.isfinite(gains)
        if not bool(finite.any()):
            break
        Z_cand, gains = Z_cand[finite], gains[finite]
        best = int(torch.argmax(gains))
        z_star = Z_cand[best : best + 1]

        selected.append(z_star)
        # Refresh the selected-batch features for the next greedy step.
        B = torch.cat(selected, dim=0)
        with torch.no_grad():
            qB = quality_fn(B)
            muB, _ = posterior_mean_std(models, B)
        ctxB = _context(B, context_dims)

    if not selected:
        X_new = _uniform_starts(bounds, batch_size, rng_seed)
    else:
        X_new = torch.cat(selected, dim=0)

    info = {
        **ref_info,
        "quality": quality,
        "lambda_obj": float(lam_obj),
        "lambda_ctx": None if lam_ctx is None else float(lam_ctx),
        "context_dims": list(context_dims) if context_dims else [],
        "n_restarts": int(n_restarts),
        "gamma": float(gamma),
    }
    return X_new.detach(), info
