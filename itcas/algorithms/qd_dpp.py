"""Quality-Diversity DPP greedy batch selection.

Constructs the Quality-Diversity L-ensemble per `methodology.tex`:

    L_ij = q(z_i) * [ k_obj(z_i, z_j) * k_ctx(c_i, c_j) ] * q(z_j)

where the *joint diversity kernel* is the product of

    k_obj(z_i, z_j) = exp( -||mu_t(z_i) - mu_t(z_j)||^2 / (2 lam_obj^2) )   (objective space)
    k_ctx(c_i, c_j) = exp( -||c_i - c_j||^2            / (2 lam_ctx^2) )   (context  space)

The objective-space kernel spreads the batch across the predicted feasible
performance manifold; the context-space kernel prevents the batch from
collapsing onto a few "easy" environments and enforces context-space
space-filling. If no context coordinates are supplied, k_ctx == 1 and the
ensemble reduces to the pure objective-space form.

Selects a batch B of size b by greedily maximising

    F(B) = log det( I + L_B )

which is monotone submodular -> (1 - 1/e) approximation guarantee.

Greedy update uses the standard DPP marginal-gain formulation: at step k,
    gain(z) = log( 1 + L'_zz - L'_z,B (I + L'_B)^{-1} L'_B,z )
where L' = L + I shorthand is folded directly via Cholesky-based incremental
updates (we use the simple but numerically stable form: recompute the small
(k+1) x (k+1) determinant ratio at each greedy step).
"""
from __future__ import annotations

from typing import Optional

import math
import torch


def rbf_kernel(feat: torch.Tensor, lam: float) -> torch.Tensor:
    """RBF Gram matrix k_ij = exp(-||feat_i - feat_j||^2 / (2 lam^2))."""
    if lam <= 0:
        raise ValueError("lambda must be positive")
    sq = torch.cdist(feat, feat).pow(2)
    return torch.exp(-sq / (2.0 * lam * lam))


# Backwards-compatible alias for the objective-space kernel.
def rbf_objective_kernel(mu: torch.Tensor, lam: float) -> torch.Tensor:
    """k_obj(z_i, z_j) = exp(-||mu(z_i) - mu(z_j)||^2 / (2 lam^2))."""
    return rbf_kernel(mu, lam)


def median_heuristic_lambda(feat: torch.Tensor) -> float:
    """Median pairwise distance heuristic for an RBF length-scale."""
    with torch.no_grad():
        d = torch.cdist(feat, feat)
        n = d.shape[0]
        if n <= 1:
            return 1.0
        triu = d[torch.triu_indices(n, n, offset=1, device=d.device).unbind()]
        med = triu.median().item()
    return max(med, 1e-6)


def build_qd_l_ensemble(
    quality: torch.Tensor,
    mu: torch.Tensor,
    ctx: Optional[torch.Tensor] = None,
    lam: Optional[float] = None,
    lam_ctx: Optional[float] = None,
) -> torch.Tensor:
    """Build the Quality-Diversity L-ensemble.

        L_ij = q_i * [ k_obj(mu_i, mu_j) * k_ctx(c_i, c_j) ] * q_j

    Args:
        quality: (N,) ROI-MI quality scores q(z).
        mu: (N, m) predicted objective-space coordinates mu_t(z).
        ctx: optional (N, d_c) context coordinates c. If None the context
            kernel is omitted (k_ctx == 1) and the ensemble is purely
            objective-space diverse.
        lam: objective-space RBF length-scale (median heuristic if None).
        lam_ctx: context-space RBF length-scale (median heuristic if None).
    """
    if lam is None:
        lam = median_heuristic_lambda(mu)
    K = rbf_kernel(mu, lam)
    if ctx is not None and ctx.numel() > 0:
        if lam_ctx is None:
            lam_ctx = median_heuristic_lambda(ctx)
        K = K * rbf_kernel(ctx, lam_ctx)
    q = quality.clamp_min(0.0)
    return (q.unsqueeze(0) * K) * q.unsqueeze(1)


def greedy_dpp_batch(
    L: torch.Tensor,
    batch_size: int,
    forbid: Optional[torch.Tensor] = None,
) -> list[int]:
    """Greedy maximisation of F(B) = log det(I + L_B).

    Args:
        L: (N, N) PSD L-ensemble.
        batch_size: number of points b to select.
        forbid: optional (N,) boolean mask of indices to exclude.

    Returns: list of selected indices (length <= batch_size).
    """
    N = L.shape[0]
    if forbid is None:
        forbid = torch.zeros(N, dtype=torch.bool, device=L.device)
    selected: list[int] = []

    diag = L.diagonal().clone()  # marginal gain at empty set: log(1 + L_ii)

    for _ in range(batch_size):
        if not selected:
            scores = torch.log1p(diag.clamp_min(0.0))
        else:
            B = torch.tensor(selected, device=L.device, dtype=torch.long)
            L_BB = L[B][:, B]
            M = L_BB + torch.eye(len(selected), device=L.device, dtype=L.dtype)
            try:
                M_inv = torch.linalg.inv(M)
            except RuntimeError:
                M_inv = torch.linalg.pinv(M)
            L_zB = L[:, B]                                        # (N, k)
            quad = (L_zB @ M_inv * L_zB).sum(dim=-1)              # (N,)
            schur = (diag - quad).clamp_min(0.0)
            scores = torch.log1p(schur)

        scores = scores.clone()
        scores[forbid] = -float("inf")
        for j in selected:
            scores[j] = -float("inf")

        best = int(torch.argmax(scores).item())
        if not math.isfinite(scores[best].item()) or scores[best].item() <= 0:
            break
        selected.append(best)

    return selected
