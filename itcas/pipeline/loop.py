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
configurable). Acquisition is one of {itcas, itcas_seq, ndig, cr_ndig,
random, one_step, ez, eisr, straddle, ...} from the registries. `cr_ndig`
(contexts/sequential_ndig.md) is the purely-sequential context-repulsive
NDIG variant used when no QD-DPP batch diversity is available.
"""
from __future__ import annotations

import re
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


# Family-B "naive cartographer" baselines that reuse the continuous
# multistart / QD-DPP machinery (select_batch_continuous) via a *fixed*
# quality variant of the same name, rather than the pipeline's configurable
# `--quality` flag (which only applies to `method == "itcas"`). Each of these
# is also selectable in DPP-batch form via the `<name>_batch` suffix (Part C
# uses a different, discrete-pool DPP path for its own four baselines, but
# the *_batch naming convention is shared -- see `effective_batch_size`).
#
# `cr_ndig` (contexts/sequential_ndig.md) also lives here purely to reuse the
# same continuous dispatch/lookup plumbing, even though it is not a Family-B
# baseline -- it is the proposed method's purely-sequential context-repulsive
# NDIG variant. It has NO `_batch` sibling: the whole point of CR-NDIG is
# that no QD-DPP batch diversity is available, so it is always forced to
# batch_size=1 via `effective_batch_size` (its name is neither `"itcas"` nor
# `_batch`-suffixed).
#
# `ndig_no_kobj`/`ndig_no_kctx` are QD-DPP diversity-kernel ablations of the
# proposed method's own `itcas`+`quality="ndig"` batch acquisition (the
# `itcas_ndig` in reporting's family tables): both use the plain `ndig`
# quality, unchanged, but their `_batch` sibling drops one half of the joint
# diversity kernel `k_obj(mu_i, mu_j) * k_ctx(c_i, c_j)` from the QD-DPP
# L-ensemble (see `NDIG_KERNEL_ABLATION` / `continuous._qd_marginal_gain`'s
# `disable_kernel`) -- `ndig_no_kobj_batch` keeps only `k_ctx` (diversity
# driven by context alone), `ndig_no_kctx_batch` keeps only `k_obj`
# (diversity driven by the objective-space prediction alone, even when
# context dimensions are present). The bare (non-`_batch`) forms are harmless
# but pointless duplicates of `itcas_seq`+`quality="ndig"`: with
# `batch_size=1` the QD-DPP cross-kernel never engages regardless of which
# half is ablated.
CONTINUOUS_BASELINE_QUALITY = {
    "c2lse": "c2lse",
    "bes": "bes",
    "cr_ndig": "cr_ndig",
    "ndig_no_kobj": "ndig",
    "ndig_no_kctx": "ndig",
}

# Maps the two NDIG QD-DPP kernel-ablation method names (see
# `CONTINUOUS_BASELINE_QUALITY` above) to the `disable_kernel` value
# `continuous.select_batch_continuous`/`_qd_marginal_gain` expects. Every
# other method (including plain `ndig` via `itcas`/`itcas_seq`) is absent
# here and so gets `None` (full joint kernel) from the `.get(...)` lookup in
# `_select_continuous`.
NDIG_KERNEL_ABLATION: dict[str, str] = {
    "ndig_no_kobj": "obj",
    "ndig_no_kctx": "ctx",
}


def _is_continuous_method(method: str) -> bool:
    """True if `method` should be routed through select_batch_continuous.

    Covers the proposed algorithm (`itcas`, quality selected via cfg.quality),
    its forced-sequential sibling `itcas_seq` (same cfg.quality machinery, but
    always batch_size=1 -- see `effective_batch_size`), the Family-B
    continuous baselines (`c2lse`/`bes`, each with a fixed quality variant of
    the same name), including their `_batch` suffix forms (batch_size>1 greedy
    QD-DPP instead of sequential argmax), `cr_ndig` (contexts/
    sequential_ndig.md), the purely-sequential context-repulsive NDIG variant
    with a fixed quality of the same name and no `_batch` sibling, and
    `ndig_no_kobj`/`ndig_no_kctx`, the QD-DPP diversity-kernel ablations of
    the proposed method's own batch NDIG acquisition (fixed quality `"ndig"`,
    see `NDIG_KERNEL_ABLATION`), including their `_batch` suffix forms.

    Family-C two-stage methods are handled separately (see
    ``TWO_STAGE_BASE`` / ``_stage_for`` / the dispatch branch in
    ``run_experiment``) since their Stage 1 vs Stage 2 continuity depends on
    which base method they wrap, not on their own name.
    """
    base = method[:-len("_batch")] if method.endswith("_batch") else method
    return base in ("itcas", "itcas_seq") or base in CONTINUOUS_BASELINE_QUALITY


# ---------------------------------------------------------------------------
# Family C: Two-Stage LSE-then-Sample ("Hybrid Cartographers")
# ---------------------------------------------------------------------------
# Maps each two-stage `--method` name (Stage-1 half of the schedule) to the
# pre-existing base acquisition it should call for `t < stage1_fraction *
# n_iters`. Stage 1 must be bit-identical to calling that base method
# directly (Phase 1 behavior, unmodified) -- see
# `_select_baseline`/`_select_continuous` reuse in `run_experiment`.
# `straddle` is the sole discrete-pool base (Family B's only baseline still
# on the candidate-pool path); `c2lse`/`bes` are continuous (see
# CONTINUOUS_BASELINE_QUALITY).
TWO_STAGE_BASE = {
    "straddle_then_sample": "straddle",
    "c2lse_then_sample": "c2lse",
    "bes_then_sample": "bes",
}

# The default 50/50 Stage-1(LSE)/Stage-2(interior) split. Every Family-C
# two-stage `--method` string must carry an explicit `_lseNN` infix -- there
# is no bare (unsuffixed) spelling -- so the default split is requested via
# `_lse50`, e.g. `straddle_then_sample_lse50`, exactly like `_lse10`/`_lse25`
# request non-default splits. This keeps every two-stage method distinguishable
# purely by its name (see `_parse_two_stage`), which is what every downstream
# consumer (results directory layout, `itcas/reporting/metrics.py`'s
# `_load_run`, etc.) groups runs by.
DEFAULT_STAGE1_FRACTION = 0.5

# Matches a *mandatory* `_lse<NN>` infix appended to one of the
# `TWO_STAGE_BASE` keys, *before* any trailing `_batch` suffix (which callers
# strip first). `pct` is captured greedily as "whatever follows `_lse`" (not
# just `\d+`) so that malformed suffixes (non-digits, out-of-range integers)
# are caught by `_parse_two_stage` and reported with a clear error instead of
# silently falling through to a generic "Unknown method" failure downstream.
_LSE_INFIX_RE = re.compile(
    r"^(?P<base>" + "|".join(re.escape(b) for b in TWO_STAGE_BASE) + r")"
    r"_lse(?P<pct>[^_]*)$"
)

# Matches one of the `TWO_STAGE_BASE` keys with NO `_lseNN` infix at all --
# the now-unsupported bare spelling (e.g. plain `straddle_then_sample`).
# Matched separately from `_LSE_INFIX_RE` purely so `_parse_two_stage` can
# raise a specific, actionable error pointing at `_lse50` instead of the
# generic "not a two-stage method" `None` return or a downstream "Unknown
# method" error several call frames away.
_BARE_TWO_STAGE_RE = re.compile(
    r"^(?P<base>" + "|".join(re.escape(b) for b in TWO_STAGE_BASE) + r")$"
)


@dataclass(frozen=True)
class TwoStageSpec:
    """Parsed Family-C two-stage method: Stage-1 base acquisition name plus
    the Stage-1 ("LSE") budget fraction of the total ``n_iters``."""

    base: str
    stage1_fraction: float


def _parse_two_stage(method: str) -> Optional[TwoStageSpec]:
    """Parse a (possibly `_batch`-suffixed) Family-C two-stage `--method`
    string. The `_lseNN` proportion infix is mandatory.

    Recognized shapes (``<base>`` in ``TWO_STAGE_BASE``, ``NN`` an integer
    percentage in ``[1, 99]``):

        <base>_lseNN         -> TwoStageSpec(base, NN/100)
        <base>_lseNN_batch   -> TwoStageSpec(base, NN/100)

    e.g. ``straddle_then_sample_lse50`` (the default 50/50 split),
    ``straddle_then_sample_lse10`` (10% LSE / 90% interior sampling).

    Returns ``None`` if ``method`` (after stripping a trailing ``_batch``) is
    not related to ``TWO_STAGE_BASE`` at all -- i.e. it is not a Family-C
    method. Raises ``ValueError`` (matching the "Unknown method: ..." style
    used elsewhere in this module) if it names a ``TWO_STAGE_BASE`` key but
    is missing the now-mandatory `_lseNN` suffix entirely, or if the suffix
    is malformed or out of the sane `[1, 99]` range (e.g. `_lse0`, `_lse100`,
    `_lse5x`) -- these must fail loudly, not be silently accepted as garbage
    or silently misinterpreted as some other method.
    """
    core = method[:-len("_batch")] if method.endswith("_batch") else method
    bare = _BARE_TWO_STAGE_RE.match(core)
    if bare is not None:
        raise ValueError(
            f"Method {method!r} is missing its mandatory LSE-proportion "
            f"suffix. Every Family-C two-stage method must carry an "
            f"explicit '_lseNN' infix -- there is no bare spelling anymore. "
            f"Use {bare.group('base') + '_lse50'!r} for the (previously "
            f"implicit) default 50/50 split."
        )
    m = _LSE_INFIX_RE.match(core)
    if m is None:
        return None
    # `m.group("base")` is the matched Family-C method-name key (e.g.
    # "straddle_then_sample"); look it up in TWO_STAGE_BASE to get the actual
    # Stage-1 acquisition it wraps (e.g. "straddle").
    method_key = m.group("base")
    base = TWO_STAGE_BASE[method_key]
    pct_str = m.group("pct")
    if not pct_str.isdigit():
        raise ValueError(
            f"Malformed two-stage LSE-proportion suffix in method {method!r}: "
            f"'_lse{pct_str}' must be an integer percentage in [1, 99] "
            f"(e.g. '_lse10', '_lse25', '_lse50')."
        )
    pct = int(pct_str)
    if not (1 <= pct <= 99):
        raise ValueError(
            f"Malformed two-stage LSE-proportion suffix in method {method!r}: "
            f"'_lse{pct_str}' = {pct}% is out of the sane [1, 99] range."
        )
    return TwoStageSpec(base=base, stage1_fraction=pct / 100.0)


def _two_stage_base(method: str) -> Optional[str]:
    """Return the Stage-1 base method name if `method` is a Family-C
    two-stage method (stripping any `_batch`/`_lseNN` suffix first), else
    None. Thin convenience wrapper around `_parse_two_stage` for call sites
    that only need the base name, not the Stage-1 fraction."""
    spec = _parse_two_stage(method)
    return spec.base if spec is not None else None


def _stage_for(method: str, t: int, n_iters: int) -> str:
    """Two-stage budget controller: 'search' (Stage 1) or 'interior' (Stage 2).

    Non-two-stage methods (everything outside ``TWO_STAGE_BASE``) always
    return 'search' -- i.e. the schedule is a no-op for them, matching the
    plan doc's ``BudgetController.get_mode`` (``if not is_two_stage: return
    'Search'``). For two-stage methods the switch point is the iteration
    index ``t`` (0-based) against ``stage1_fraction * n_iters`` -- NOT the
    raw evaluation count -- per the caller's contract: ``t < frac*T ->
    'search'``, ``t >= frac*T -> 'interior'``. ``frac`` comes from the
    method's mandatory `_lseNN` infix (see `_parse_two_stage`), e.g. 0.5 for
    `_lse50`, 0.1 for `_lse10`.
    """
    spec = _parse_two_stage(method)
    if spec is None:
        return "search"
    return "search" if t < spec.stage1_fraction * n_iters else "interior"


@dataclass
class ExperimentConfig:
    method: str = "itcas"             # itcas | itcas_seq (forced-sequential sibling
                                      # of itcas -- same cfg.quality machinery,
                                      # but always batch_size=1, unlike bare
                                      # itcas which honors cfg.batch_size) |
                                      # random | one_step | ez | eisr | straddle |
                                      # cas_eci | moc_cas_hard | moc_cas_soft |
                                      # c2lse | bes (Family-B continuous LSE
                                      # baselines, each with a fixed --quality of the
                                      # same name -- see CONTINUOUS_BASELINE_QUALITY) |
                                      # cr_ndig (contexts/sequential_ndig.md: purely-
                                      # sequential context-repulsive NDIG variant, fixed
                                      # --quality of the same name, no _batch sibling --
                                      # see CONTINUOUS_BASELINE_QUALITY) |
                                      # ndig_no_kobj | ndig_no_kctx (QD-DPP diversity-
                                      # kernel ablations of the proposed method's own
                                      # batch NDIG acquisition, fixed --quality "ndig";
                                      # the _batch sibling drops k_obj / k_ctx
                                      # respectively from the L-ensemble -- see
                                      # NDIG_KERNEL_ABLATION) |
                                      # random_batch | straddle_batch | cas_eci_batch |
                                      # moc_cas_hard_batch (discrete-pool + QD-DPP
                                      # batch siblings of the sequential baselines) |
                                      # c2lse_batch | bes_batch (continuous
                                      # + QD-DPP batch siblings) |
                                      # <base>_lseNN / <base>_lseNN_batch where
                                      # <base> in {straddle_then_sample,
                                      # c2lse_then_sample, bes_then_sample}
                                      # (Family-C two-stage: Stage-1 base
                                      # acquisition for t < (NN/100)*n_iters,
                                      # then Stage-2 penalized interior-sampling
                                      # for t >= (NN/100)*n_iters -- see
                                      # TWO_STAGE_BASE / _stage_for). The
                                      # `_lseNN` infix is MANDATORY (NN an
                                      # integer in [1, 99]; there is no bare
                                      # spelling) -- e.g.
                                      # straddle_then_sample_lse50 (the
                                      # default 50/50 split),
                                      # straddle_then_sample_lse10 (10% LSE /
                                      # 90% interior sampling),
                                      # c2lse_then_sample_lse25_batch (25% LSE,
                                      # batch sibling) -- see
                                      # _parse_two_stage / TwoStageSpec | ...
                                      # (RMILE was implemented and removed: its
                                      # per-candidate joint-posterior-covariance
                                      # solve over a reference pool was too
                                      # expensive to keep in the benchmark matrix.)
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
                     X_obs, Y_obs, context_dims=(), t: int = 0):
    """Discrete-pool selection for the baselines (returns indices into ``cand``).

    ``t`` is the 0-based pipeline iteration index. It is forwarded to every
    baseline via ``kwargs``; only ``straddle``/``straddle_batch`` consume it
    (to alternate which objective drives the score, ``active_obj = t % m``),
    every other baseline already swallows unknown kwargs via ``**kwargs``.
    """
    if method not in BASELINES:
        raise ValueError(f"Unknown method: {method}")
    kwargs = dict(
        rng_seed=cfg.seed,
        X_obs=X_obs,
        Y_obs=Y_obs,
        radius=cfg.radius,
        beta=cfg.beta,
        lam=cfg.soft_lambda,
        context_dims=context_dims,
        dpp_lambda=cfg.dpp_lambda,
        dpp_lambda_ctx=cfg.dpp_lambda_ctx,
        t=t,
    )
    # cas_eci uses input-space radius; moc_cas_* use objective-space radius
    if method.startswith("moc_cas"):
        kwargs["radius"] = cfg.obj_radius
    return BASELINES[method](
        models=models, cand=cand, h=h, batch_size=batch_size, **kwargs,
    )


def _select_continuous(
    *, models, bounds, h, batch_size, cfg: ExperimentConfig, context_dims, seed,
    method: Optional[str] = None, quality_override: Optional[str] = None, t: int = 0,
    X_obs=None,
):
    """Continuous multistart / QD-DPP selection (returns points in the original domain).

    Dispatches ``method == "itcas"`` (and its forced-sequential sibling
    ``"itcas_seq"``) to the configurable ``cfg.quality`` variant (unchanged
    behavior), and each Family-B continuous baseline (``c2lse``/``bes``,
    optionally ``_batch``-suffixed) to its own fixed quality variant of the
    same name via ``CONTINUOUS_BASELINE_QUALITY``.

    ``method`` defaults to ``cfg.method`` but can be overridden -- used by the
    Family-C two-stage Stage-1 dispatch, which needs to call this with the
    *base* method name (e.g. ``"c2lse"``) while ``cfg.method`` is still
    ``"c2lse_then_sample"``. ``quality_override``, if given, bypasses the
    method-name lookup entirely (used by the Family-C Stage-2 "interior"
    dispatch, which always wants the ``"interior_sampling"`` quality
    regardless of which base method precedes it).

    ``t`` is the 0-based pipeline iteration index (the global loop counter in
    ``run_experiment``, NOT a stage-relative counter). It is forwarded through
    ``itcas_select_batch`` -> ``build_quality_fn`` to the ``c2lse``/``bes``
    builders so they can alternate ``active_obj = t % m``; other quality
    variants ignore it via ``**_``.

    ``X_obs`` is the full accumulated observed dataset ``D_{t-1}`` (before
    this iteration's new points are appended), forwarded through
    ``itcas_select_batch`` -> ``build_quality_fn`` to the ``cr_ndig`` builder
    so it can build its context-repulsion penalty against acquisition
    history; every other quality variant ignores it via ``**_``.

    ``base``'s presence in ``NDIG_KERNEL_ABLATION`` (``ndig_no_kobj``/
    ``ndig_no_kctx``) forwards the corresponding ``disable_kernel`` ablation
    of the QD-DPP diversity kernel through to ``itcas_select_batch``; every
    other method gets ``None`` (the full joint kernel), unchanged behavior.
    """
    method = cfg.method if method is None else method
    base = method[:-len("_batch")] if method.endswith("_batch") else method
    if quality_override is not None:
        quality = quality_override
    else:
        quality = cfg.quality if base in ("itcas", "itcas_seq") else CONTINUOUS_BASELINE_QUALITY[base]
    return itcas_select_batch(
        models=models, bounds=bounds, tau=h, batch_size=batch_size,
        context_dims=context_dims, quality=quality, gamma=cfg.gamma,
        n_restarts=cfg.n_restarts, n_opt_steps=cfg.n_opt_steps, opt_lr=cfg.opt_lr,
        n_ts_samples=cfg.n_ts_samples, dpp_lambda=cfg.dpp_lambda,
        dpp_lambda_ctx=cfg.dpp_lambda_ctx, rng_seed=seed, t=t, X_obs=X_obs,
        disable_kernel=NDIG_KERNEL_ABLATION.get(base),
    )


def effective_batch_size(method: str, batch_size: int) -> int:
    """Per-iteration batch size for a method.

    A method is a "batch" method iff it is the proposed algorithm (``itcas``)
    or its name ends with ``_batch`` (the DPP-batch siblings of the
    sequential baselines, e.g. ``random_batch``, ``straddle_batch``,
    ``c2lse_batch``, and the Family-C two-stage siblings
    ``straddle_then_sample_batch``/``c2lse_then_sample_batch``/
    ``bes_then_sample_batch``). Every other
    (sequential) method -- including the non-``_batch`` Family-C two-stage
    methods, and ``itcas_seq`` (the forced-sequential sibling of ``itcas``,
    same ``cfg.quality`` machinery but always one evaluation per iteration
    regardless of ``batch_size``) -- performs a single evaluation per
    iteration, exactly matching their Stage-1 base method's batch behavior.
    Total budget ``T`` is held identical across methods, so sequential
    methods simply run ``T`` iterations of one point each.
    """
    return batch_size if (method == "itcas" or method.endswith("_batch")) else 1


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

    # The proposed algorithm (itcas) and any `_batch`-suffixed sibling select a
    # batch per iteration (via QD-DPP); every other (sequential) method --
    # including `itcas_seq`, the forced-sequential sibling of `itcas` used to
    # benchmark a quality variant (e.g. ndig) one point at a time -- runs a
    # single evaluation per iteration (eff_batch == 1), so the comparison is
    # "batch active search" vs. "sequential" under an identical total budget T.
    is_batch_method = cfg.method == "itcas" or cfg.method.endswith("_batch")
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

    two_stage_base = _two_stage_base(cfg.method)  # None for non-Family-C methods
    is_two_stage = two_stage_base is not None
    two_stage_batch = cfg.method.endswith("_batch")  # `<base>_then_sample_batch`

    n_iters = (cfg.budget + eff_batch - 1) // eff_batch
    n_iters_run = 0
    for t in range(n_iters):
        models = build_independent_gps(X, Y, bounds=bounds)
        stage = _stage_for(cfg.method, t, n_iters)

        if is_two_stage:
            if stage == "search":
                # Stage 1 ("Search"): run the wrapped base method's existing
                # selection path exactly as Phase 1 wired it (bit-identical to
                # `--method <base>` for t < 0.5*n_iters). Dispatch discrete
                # (straddle) vs. continuous (c2lse/bes) by reusing
                # `_select_baseline`/`_select_continuous` with the base name.
                if _is_continuous_method(two_stage_base):
                    X_new, info = _select_continuous(
                        models=models, bounds=bounds, h=h, batch_size=eff_batch, cfg=cfg,
                        context_dims=problem.context_dims, seed=cfg.seed + 1000 + t,
                        method=two_stage_base, t=t, X_obs=X,
                    )
                    selected_idx: list[int] = []
                else:
                    cand = problem.sample_uniform(
                        cfg.n_candidates, seed=cfg.seed + 1000 + t, device=device, dtype=torch.double
                    )
                    method_for_select = two_stage_base + "_batch" if two_stage_batch else two_stage_base
                    idx, info = _select_baseline(
                        method_for_select, models=models, cand=cand, h=h,
                        batch_size=eff_batch, cfg=cfg, X_obs=X, Y_obs=Y,
                        context_dims=problem.context_dims, t=t,
                    )
                    if not idx:
                        idx = _fallback_indices(
                            models, cand, h, eff_batch, rng_seed=cfg.seed + 1000 + t
                        )
                        info = {**info, "fallback": True}
                    if not idx:
                        break
                    selected_idx = idx
                    X_new = cand[idx]
            else:
                # Stage 2 ("Interior"): pure combined-uncertainty exploration
                # soft-penalized to stay inside the predicted feasible
                # interior (joint PoF >= 0.95). Discrete-pool path for
                # straddle_then_sample (PoF-filtered pool + fallback), the
                # continuous `interior_sampling` quality variant otherwise
                # (identical Stage-2 objective regardless of the Stage-1 base).
                if two_stage_base == "straddle":
                    cand = problem.sample_uniform(
                        cfg.n_candidates, seed=cfg.seed + 1000 + t, device=device, dtype=torch.double
                    )
                    interior_method = "interior_sampling_batch" if two_stage_batch else "interior_sampling"
                    idx, info = _select_baseline(
                        interior_method, models=models, cand=cand, h=h,
                        batch_size=eff_batch, cfg=cfg, X_obs=X, Y_obs=Y,
                        context_dims=problem.context_dims, t=t,
                    )
                    if not idx:
                        # No candidate cleared PoF>=0.95 (plausible early / on
                        # hard problems): reuse the same exploration fallback
                        # every other baseline's empty-index case triggers.
                        idx = _fallback_indices(
                            models, cand, h, eff_batch, rng_seed=cfg.seed + 1000 + t
                        )
                        info = {**info, "fallback": True}
                    if not idx:
                        break
                    selected_idx = idx
                    X_new = cand[idx]
                else:
                    X_new, info = _select_continuous(
                        models=models, bounds=bounds, h=h, batch_size=eff_batch, cfg=cfg,
                        context_dims=problem.context_dims, seed=cfg.seed + 1000 + t,
                        method=two_stage_base, quality_override="interior_sampling", t=t,
                    )
                    selected_idx = []
            info = {**info, "stage": stage}
        elif _is_continuous_method(cfg.method):
            # Continuous multistart / QD-DPP: optimize the acquisition directly
            # over the joint design-context domain X x C; returns points (not
            # pool indices). Covers "itcas" (cfg.quality) and the Family-B
            # continuous baselines c2lse/bes (+ their _batch siblings).
            X_new, info = _select_continuous(
                models=models, bounds=bounds, h=h, batch_size=eff_batch, cfg=cfg,
                context_dims=problem.context_dims, seed=cfg.seed + 1000 + t, t=t,
                X_obs=X,
            )
            selected_idx: list[int] = []
            info = {**info, "stage": stage}
        else:
            cand = problem.sample_uniform(
                cfg.n_candidates, seed=cfg.seed + 1000 + t, device=device, dtype=torch.double
            )
            idx, info = _select_baseline(
                cfg.method, models=models, cand=cand, h=h,
                batch_size=eff_batch, cfg=cfg, X_obs=X, Y_obs=Y,
                context_dims=problem.context_dims, t=t,
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
            info = {**info, "stage": stage}

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
        thresholds=h,
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
