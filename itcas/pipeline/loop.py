"""Reproducible active-search experiment loop.

Pipeline (per CAS / MOC-CAS specs):

    1. Initialize D_0
    2. For t = 1..T:
        a. Update GP posteriors on D_{t-1}
        b. Optimize acquisition over a candidate pool to pick a batch
        c. Evaluate true objectives -> y_t
        d. D_t = D_{t-1} cup {(x_t, y_t)}
        e. Log per-iteration metrics

The candidate pool is a Sobol-sampled set re-drawn each iteration (size
configurable). Acquisition is one of {itcas, random, one_step, ez, eisr,
straddle, ...} from the registries.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch

from ..algorithms import itcas_select_batch
from ..baselines import REGISTRY as BASELINES
from ..io import RunLogger
from ..metrics import (
    aup,
    cumulative_positives,
    epsilon_archive_size,
    feasible_context_fill_distance,
    feasible_convex_hull_volume,
    is_feasible,
    positive_samples,
)
from ..metrics.reference import build_reference_data
from ..utils.gp import build_independent_gps, posterior_mean_std
from ..utils.device import resolve_device
from ..utils.seeding import set_global_seed
from .problems import Problem


def _plot_run_scatter(
    *,
    X: torch.Tensor,
    Y: torch.Tensor,
    thresholds: torch.Tensor,
    n_init: int,
    context_dims: tuple[int, ...],
    out_dir: str,
    run_name: str,
) -> list[str]:
    """Write a per-run scatter plot when objective/context spaces are 2D.

    Produces ``<run_name>.spaces2d.png`` in ``out_dir`` when either the
    objective space is 2D or exactly two context dimensions exist. If both are
    2D, objective space is plotted on top and context space on the bottom.
    """
    paths: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return paths

    X_cpu = X.detach().cpu()
    Y_cpu = Y.detach().cpu()
    n_total = int(X_cpu.shape[0])
    n_init_clamped = max(0, min(int(n_init), n_total))
    init_slice = slice(0, n_init_clamped)
    acquired_slice = slice(n_init_clamped, n_total)
    has_objective_2d = Y_cpu.ndim == 2 and int(Y_cpu.shape[1]) == 2
    has_context_2d = len(context_dims) == 2

    panels = int(has_objective_2d) + int(has_context_2d)
    if panels == 0:
        return paths

    fig, axes = plt.subplots(
        panels,
        1,
        figsize=(6.4, 5.4 * panels),
        squeeze=False,
    )
    ax_iter = iter(axes[:, 0])

    if has_objective_2d:
        ax = next(ax_iter)
        if n_init_clamped > 0:
            ax.scatter(
                Y_cpu[init_slice, 0].numpy(),
                Y_cpu[init_slice, 1].numpy(),
                s=20,
                alpha=0.8,
                label="init",
            )
        if n_total > n_init_clamped:
            ax.scatter(
                Y_cpu[acquired_slice, 0].numpy(),
                Y_cpu[acquired_slice, 1].numpy(),
                s=24,
                alpha=0.85,
                marker="x",
                label="acquired",
            )
        tau = thresholds.detach().cpu().flatten()
        if int(tau.numel()) >= 2:
            x_vals = Y_cpu[:, 0].numpy()
            y_vals = Y_cpu[:, 1].numpy()
            t0 = float(tau[0].item())
            t1 = float(tau[1].item())
            x_lo, x_hi = min(float(x_vals.min()), t0), max(float(x_vals.max()), t0)
            y_lo, y_hi = min(float(y_vals.min()), t1), max(float(y_vals.max()), t1)
            x_pad = 0.05 * max(1e-9, x_hi - x_lo)
            y_pad = 0.05 * max(1e-9, y_hi - y_lo)
            ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
            ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
            ax.axvline(t0, color="black", linestyle="--", linewidth=2.0, alpha=0.9,
                       zorder=10, label="threshold obj0")
            ax.axhline(t1, color="dimgray", linestyle="--", linewidth=2.0, alpha=0.9,
                       zorder=10, label="threshold obj1")
        ax.set_title("Sampled points in objective space")
        ax.set_xlabel("objective 0")
        ax.set_ylabel("objective 1")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.25)

    if has_context_2d:
        c0, c1 = int(context_dims[0]), int(context_dims[1])
        ax = next(ax_iter)
        if n_init_clamped > 0:
            ax.scatter(
                X_cpu[init_slice, c0].numpy(),
                X_cpu[init_slice, c1].numpy(),
                s=20,
                alpha=0.8,
                label="init",
            )
        if n_total > n_init_clamped:
            ax.scatter(
                X_cpu[acquired_slice, c0].numpy(),
                X_cpu[acquired_slice, c1].numpy(),
                s=24,
                alpha=0.85,
                marker="x",
                label="acquired",
            )
        ax.set_title("Sampled points in context space")
        ax.set_xlabel(f"x[{c0}]")
        ax.set_ylabel(f"x[{c1}]")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out_path = Path(out_dir) / f"{run_name}.spaces2d.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    paths.append(str(out_path))

    return paths


@dataclass
class ExperimentConfig:
    method: str = "itcas"             # itcas | random | one_step | ez | eisr | straddle |
                                      # cas_eci | moc_cas_hard | moc_cas_soft | ...
    quality: str = "roi_mi"           # itcas candidate-quality variant: roi_mi | efig
    budget: int = 50
    batch_size: int = 1
    n_init: int = 8
    n_candidates: int = 256           # candidate pool size per iter (baselines)
    n_ts_samples: int = 1             # ROI-MI Thompson (RFF) samples
    dpp_lambda: Optional[float] = None        # objective-space RBF length-scale
    dpp_lambda_ctx: Optional[float] = None    # context-space RBF length-scale
    gamma: float = 0.1                # smooth-margin (LogSumExp) softmin constant
    n_restarts: int = 16              # multi-start restarts for continuous opt
    n_opt_steps: int = 60             # gradient steps per restart
    opt_lr: float = 0.05              # Adam learning rate for continuous opt
    seed: int = 0
    target_X: int = 10                # for T@X metric
    radius: float = 0.1               # input-space radius (CAS/ECI, coverage metric)
    obj_radius: float = 0.5           # objective-space radius (MOC-CAS coverage)
    beta: float = 2.0                 # UCB confidence multiplier (MOC-CAS)
    soft_lambda: float = 0.1          # smoothness for MOC-CAS soft probit gate
    eps_archive: Optional[float] = None  # ε-Archive threshold; None → use problem.eps_archive
    out_dir: str = "results"
    run_name: str = "run"
    device: str = "auto"              # auto | cpu | cuda | cuda:N
    extra: dict = field(default_factory=dict)


def _select_baseline(method: str, *, models, cand, h, batch_size, cfg: ExperimentConfig,
                     X_obs, Y_obs):
    """Discrete-pool selection for the baselines (returns indices into ``cand``)."""
    if method not in BASELINES:
        raise ValueError(f"Unknown method: {method}")
    kwargs = dict(
        rng_seed=cfg.seed,
        X_obs=X_obs,
        Y_obs=Y_obs,
        radius=cfg.radius,
        beta=cfg.beta,
        lam=cfg.soft_lambda,
    )
    # cas_eci uses input-space radius; moc_cas_* use objective-space radius
    if method.startswith("moc_cas"):
        kwargs["radius"] = cfg.obj_radius
    return BASELINES[method](
        models=models, cand=cand, h=h, batch_size=batch_size, **kwargs,
    )


def _select_itcas(*, models, bounds, h, batch_size, cfg: ExperimentConfig, context_dims, seed):
    """Continuous C-MO-CAS selection (returns points in the original domain)."""
    return itcas_select_batch(
        models=models, bounds=bounds, tau=h, batch_size=batch_size,
        context_dims=context_dims, quality=cfg.quality, gamma=cfg.gamma,
        n_restarts=cfg.n_restarts, n_opt_steps=cfg.n_opt_steps, opt_lr=cfg.opt_lr,
        n_ts_samples=cfg.n_ts_samples, dpp_lambda=cfg.dpp_lambda,
        dpp_lambda_ctx=cfg.dpp_lambda_ctx, rng_seed=seed,
    )


def effective_batch_size(method: str, batch_size: int) -> int:
    """Per-iteration batch size for a method.

    Only the proposed algorithm (``itcas``) runs in a batch setting; every
    baseline performs a single evaluation per iteration. Total budget ``T`` is
    held identical across methods, so baselines simply run ``T`` sequential
    iterations of one point each.
    """
    return batch_size if method == "itcas" else 1


def _fallback_indices(models, cand, h, batch_size, rng_seed=0) -> list[int]:
    """Exploration fallback when the acquisition returns no points.

    Ranks candidates by joint feasibility probability p(z) = prod_i P(f_i > h_i)
    and returns the top ``batch_size`` distinct indices. If every probability is
    numerically zero (e.g. far from any predicted feasible region) it falls back
    to a uniform-random draw so the loop never stalls and the total budget T is
    always consumed.
    """
    from ..algorithms.roi_mi import feasibility_probabilities, joint_feasibility_probability

    n = cand.shape[0]
    k = min(batch_size, n)
    if k <= 0:
        return []
    mu, sigma = posterior_mean_std(models, cand)
    p = joint_feasibility_probability(feasibility_probabilities(mu, sigma, h))
    if bool((p > 0).any()):
        return torch.topk(p, k).indices.tolist()
    g = torch.Generator(device="cpu").manual_seed(int(rng_seed))
    perm = torch.randperm(n, generator=g)[:k]
    return perm.tolist()


def run_experiment(problem: Problem, cfg: ExperimentConfig) -> dict:
    set_global_seed(cfg.seed)
    device = resolve_device(cfg.device)
    h = problem.thresholds.to(device=device, dtype=torch.double)
    bounds = problem.bounds.to(device=device, dtype=torch.double)

    # Only the proposed algorithm (itcas) selects a batch per iteration. All
    # baselines run a single evaluation per iteration (eff_batch == 1), so the
    # comparison is "batch active search (ours)" vs. "sequential (baselines)"
    # under an identical total budget T.
    is_batch_method = cfg.method == "itcas"
    eff_batch = effective_batch_size(cfg.method, cfg.batch_size)

    # Init dataset
    X = problem.sample_uniform(cfg.n_init, seed=cfg.seed, device=device, dtype=torch.double)
    Y = problem.evaluate(X).to(torch.double)
    init_feasible = is_feasible(Y, h).tolist()
    init_positive_samples = int(sum(bool(f) for f in init_feasible))
    init_X = X.detach().cpu().tolist()
    init_Y = Y.detach().cpu().tolist()

    logger = RunLogger(cfg.out_dir, cfg.run_name)
    per_iter_feasible: list[bool] = []

    n_iters = (cfg.budget + eff_batch - 1) // eff_batch
    n_iters_run = 0
    for t in range(n_iters):
        models = build_independent_gps(X, Y, bounds=bounds)

        if cfg.method == "itcas":
            # Continuous C-MO-CAS: optimize the acquisition directly over the
            # joint design-context domain X x C; returns points (not pool idx).
            X_new, info = _select_itcas(
                models=models, bounds=bounds, h=h, batch_size=eff_batch, cfg=cfg,
                context_dims=problem.context_dims, seed=cfg.seed + 1000 + t,
            )
            selected_idx: list[int] = []
        else:
            cand = problem.sample_uniform(
                cfg.n_candidates, seed=cfg.seed + 1000 + t, device=device, dtype=torch.double
            )
            idx, info = _select_baseline(
                cfg.method, models=models, cand=cand, h=h,
                batch_size=eff_batch, cfg=cfg, X_obs=X, Y_obs=Y,
            )
            if not idx:
                # The acquisition produced no points (e.g. a degenerate ROI under
                # a tight threshold). Fall back to an exploration pick so the
                # search keeps moving toward the predicted feasible region.
                idx = _fallback_indices(
                    models, cand, h, eff_batch, rng_seed=cfg.seed + 1000 + t
                )
                info = {**info, "fallback": True}
            if not idx:
                break
            selected_idx = idx
            X_new = cand[idx]

        if X_new.numel() == 0:
            break
        Y_new = problem.evaluate(X_new).to(torch.double)

        feas = is_feasible(Y_new, h).tolist()
        per_iter_feasible.extend([bool(f) for f in feas])

        X = torch.cat([X, X_new], dim=0)
        Y = torch.cat([Y, Y_new], dim=0)

        logger.log_iter({
            "iter": t,
            "step": t + 1,
            "selected_idx": selected_idx,
            "x": X_new,
            "y": Y_new,
            "feasible": feas,
            "n_eval_this_iter": int(X_new.shape[0]),
            "n_eval_total": int(X.shape[0]),
            "info": {k: v for k, v in info.items() if k != "score"},
            "n_positives_so_far": positive_samples(Y, h),
        })
        n_iters_run += 1

    summary = {
        "config": {**asdict(cfg), "eps_archive": cfg.eps_archive if cfg.eps_archive is not None else problem.eps_archive},
        "problem": problem.name,
        "device": str(device),
        "batch_method": is_batch_method,
        "eff_batch_size": eff_batch,
        "n_iters": n_iters_run,
        "n_iters_planned": n_iters,
        "n_init": int(cfg.n_init),
        "n_init_positives": init_positive_samples,
        "init_X": init_X,
        "init_Y": init_Y,
        "init_feasible": [bool(f) for f in init_feasible],
        "n_total": int(X.shape[0]),
        "thresholds": h.detach().cpu().tolist(),
        "context_dims": list(problem.context_dims),
        "positive_samples": positive_samples(Y, h),
        "aup": aup(per_iter_feasible),
        "cumulative_positives": cumulative_positives(per_iter_feasible),
    }

    # Fill-distance metrics against problem-specific reference sets.
    ref = build_reference_data(problem, seed=cfg.seed, thresholds=h.cpu())
    feas_mask = is_feasible(Y, h)
    cdims = list(problem.context_dims)
    if cdims and ref.context_ref is not None:
        ctx_ref = ref.context_ref.to(device=device, dtype=torch.double)
        feas_ctx = X[feas_mask][:, cdims] if bool(feas_mask.any()) else X[:0, cdims]
        summary["feasible_context_fill_distance"] = feasible_context_fill_distance(
            feas_ctx, ctx_ref, penalty=ref.context_penalty
        )
    disc_Y = Y[feas_mask] if bool(feas_mask.any()) else Y[:0]
    summary["feasible_convex_hull_volume"] = feasible_convex_hull_volume(disc_Y)
    summary["epsilon_archive_size"] = epsilon_archive_size(
        disc_Y,
        eps=cfg.eps_archive if cfg.eps_archive is not None else problem.eps_archive,
    )

    scatter_paths = _plot_run_scatter(
        X=X,
        Y=Y,
        thresholds=h,
        n_init=cfg.n_init,
        context_dims=tuple(problem.context_dims),
        out_dir=cfg.out_dir,
        run_name=cfg.run_name,
    )
    if scatter_paths:
        summary["run_scatter_plots"] = scatter_paths

    logger.finalize(summary)
    return summary
