"""Landscape visualization for the synthetic test problems.

A purely diagnostic, read-only tool: for a handful of problems registered in
:mod:`itcas.pipeline.problems`, it evaluates each problem's *true* objective
function ``Problem.evaluate_true`` (never touching the GP/BO pipeline or any
run logs) and renders one PDF per problem with:

    - one row per calibrated difficulty level (target joint-feasible
      fraction ``p`` in ``{0.2, 0.1, 0.05, 0.01}``, loaded from
      ``configs/thresholds.json`` via
      :func:`itcas.pipeline.thresholds.load_thresholds`, easiest ``p`` at the
      top), and, within each row, ``m + 2`` panels:

    1. ``m`` per-objective panels: a continuous heatmap over the first two
       design dims ``x_a, x_b`` of ``P(f_i(x_a, x_b, w) >= tau_i)``, i.e. the
       probability (estimated by Monte-Carlo averaging over the remaining
       dims ``w``) that objective ``i`` alone is feasible at that
       ``(x_a, x_b)`` location.
    2. one "marginalized joint-grid" panel: the analogous joint-feasibility
       probability ``P(all objectives feasible | x_a, x_b)``, displayed as a
       binary green/white mask via grid-quantile (top-k) matching so the
       displayed green area is, by construction, very close to the row's
       target fraction ``p``.
    3. one "random-projection scatter" panel: a direct Monte-Carlo sample of
       the true full-dimensional feasible set, projected onto a fixed random
       2D orthonormal basis, colored green/gray by the same joint-feasibility
       test -- no matching needed, because it is a direct sample rather than
       a probability estimate.

Why marginalize instead of pinning the non-plotted dims (the previous
approach)
------------------------------------------------------------------------
The calibrated thresholds in ``configs/thresholds.json`` are chosen (see
:mod:`itcas.pipeline.thresholds`) so that a target fraction ``p`` of points
``Z`` sampled *uniformly over the full domain* (``problem.bounds``, all
design + context dims) are jointly feasible. Pinning every non-plotted
dimension at a single point -- whether the bounds midpoint or one arbitrary
``sample_uniform`` draw -- makes the displayed 2D feasible fraction an
artifact of that one point's neighborhood, with no reliable relationship to
``p``: e.g. a midpoint pin can be systematically favorable (valley-shaped
objectives are near-minimal at the origin) while a random pin can land
anywhere, favorable or not. Neither is representative.

Instead, because ``problem.bounds`` is an axis-aligned hyperrectangle,
``Z ~ Uniform(bounds)`` has independent uniform marginals per dimension.
Writing ``Z = (x, w)`` with ``x = (x_a, x_b)`` the two plotted design dims
and ``w`` every other dim, the calibration condition is
``p = Pr_Z[feasible(Z)] = E_x[ g(x) ]`` where
``g(x) := Pr_w[feasible(x, w) | x]`` (law of total probability). So the
*grid panel* renders exactly ``g(x_a, x_b)``, estimated by Monte-Carlo
averaging ``feasible(x, w)`` over many draws of ``w`` per grid cell (common
random numbers ``W`` shared across all cells for variance reduction). Its
spatial average over the grid is, up to MC/grid discretization noise,
guaranteed to equal ``p`` -- this is reported per row as a sanity check.
However the *displayed* green/white area still needs quantile (top-k)
matching, because ``g`` is a smoothly-varying *probability field*, not a
0/1 indicator -- a naive single global cutoff (e.g. ``g >= 0.5``) would not
generally select an area fraction equal to ``p``. Top-k matching picks
exactly the top ``round(p * grid_n^2)`` grid cells by ``g``-value, so the
displayed green area fraction equals ``p`` by construction (up to the
``1/grid_n^2`` rounding granularity), independent of how ``g`` is
distributed.

The *random-projection scatter panel* needs no such matching: it is a direct
Monte-Carlo sample ``X ~ Uniform(bounds)`` (full dimension, nothing pinned or
marginalized), evaluated and classified feasible/infeasible exactly as the
calibration itself was computed, then projected onto a fixed random 2D
orthonormal basis ``Q`` (via QR of a fixed-seed Gaussian matrix) purely for
visualization -- the projection does not affect which points are feasible,
only where they are drawn. Its achieved feasible-point fraction is therefore
an unbiased direct sample of ``p`` and should match the target tightly
(limited only by sampling noise, i.e. tighter than the grid panel's
discretization-limited match). It is direction-agnostic (no two dims singled
out) and gives an honest view of the true feasible set's overall shape and
proportion, complementing the other panels' axis-aligned two-design-dim
slice/marginal.

The underlying ``(grid_n, grid_n, mc_samples, m)`` objective tensor and the
``(n_scatter, m)`` scatter evaluation are each computed once per problem and
reused across all four difficulty rows -- only the ``tau`` vector (and hence
the feasibility masks/probabilities/matching) changes per row.

Intended primarily for the five problems used in the "two schools of
thought" (LSE-then-sample vs CAS) comparison, see
``configs/two_schools_of_thought.json``, but works for any problem in
:data:`itcas.pipeline.problems.REGISTRY` with at least 2 design dims and
calibrated thresholds for all four difficulty levels in
``configs/thresholds.json``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Sequence

import torch

from ..pipeline.problems import REGISTRY, Problem
from ..pipeline.thresholds import DEFAULT_THRESHOLDS_PATH, load_thresholds

_DEFAULT_PROBLEMS = [
    "all_valley_8d",
    "alpine_12d",
    "dtlz1_12d",
    "dtlz2_6d",
    "dtlz3_8d",
]

_DEFAULT_CONFIG = "configs/two_schools_of_thought.json"
_DEFAULT_OUTPUT_DIR = "results/landscapes"

# Difficulty levels, easiest (largest target feasible fraction) first, so
# rows read top-to-bottom as "easiest -> hardest".
_DIFFICULTY_LEVELS = (0.2, 0.1, 0.05, 0.01)

# Fixed, hardcoded seeds -- keep the figure (and the diagnostics printed
# alongside it) exactly reproducible across runs.
_MC_SEED = 0        # shared common-random-numbers batch W for the grid panel
_SCATTER_SEED = 1   # full-dimensional uniform sample X for the scatter panel
_PROJECTION_SEED = 2  # random 2D projection basis Q

# Defaults chosen to keep this a fast diagnostic (well under a minute for
# all 5 problems combined; see module-level smoke test / CLI run notes).
_DEFAULT_GRID_N = 60
_DEFAULT_MC_SAMPLES = 300
_DEFAULT_N_SCATTER = 8000
# Cap on (cells * mc_samples) evaluated per `problem.evaluate_true` call, to
# bound peak memory for large grid_n/mc_samples/d combinations.
_MAX_CHUNK_ROWS = 500_000


def _load_default_problems(config_path: str | Path = _DEFAULT_CONFIG) -> list[str]:
    """Read the ``"problems"`` list from the two-schools config, if present.

    Falls back to :data:`_DEFAULT_PROBLEMS` (the same five problems, hardcoded)
    if the config file is missing or has no ``"problems"`` key, so this module
    stays runnable standalone even without the config on disk.
    """
    path = Path(config_path)
    if path.exists():
        with path.open() as f:
            cfg = json.load(f)
        problems = cfg.get("problems")
        if problems:
            return list(problems)
    return list(_DEFAULT_PROBLEMS)


def _build_problem(problem_name: str) -> Problem:
    if problem_name not in REGISTRY:
        raise KeyError(
            f"Unknown problem '{problem_name}'; available: {sorted(REGISTRY)}"
        )
    return REGISTRY[problem_name]()


def _build_marginalized_grid(
    problem: Problem,
    grid_n: int,
    mc_samples: int,
    mc_seed: int = _MC_SEED,
    max_chunk_rows: int = _MAX_CHUNK_ROWS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    """Monte-Carlo estimate of the objective distribution over the first two
    design dims, marginalizing (averaging) over every other dimension.

    Returns ``(Ygrid, xs, ys, (dim_a, dim_b))`` where ``Ygrid`` has shape
    ``(grid_n, grid_n, mc_samples, m)``: for each of the ``grid_n * grid_n``
    ``(x_a, x_b)`` grid cells, ``mc_samples`` draws of the noiseless
    objective vector with the remaining dims ``w`` resampled from
    ``problem.sample_uniform``. The same batch ``W`` of ``w`` draws (common
    random numbers) is reused across every grid cell, both for variance
    reduction and so a single call to ``problem.evaluate_true`` per chunk
    suffices (no per-cell Python loop). ``xs``/``ys`` are the 1-D coordinate
    vectors for the two varying axes (``xs`` indexes the grid's first axis),
    and ``(dim_a, dim_b)`` are their column indices (the first two entries
    of ``problem.design_dims``).
    """
    design_dims = problem.design_dims
    if len(design_dims) < 2:
        raise ValueError(
            f"{problem.name}: need >= 2 design dims for a 2D slice, "
            f"got design_dims={design_dims}"
        )
    dim_a, dim_b = design_dims[0], design_dims[1]
    lo, hi = problem.bounds[0].double(), problem.bounds[1].double()
    d = problem.d
    m = problem.m

    xs = torch.linspace(float(lo[dim_a]), float(hi[dim_a]), grid_n, dtype=torch.double)
    ys = torch.linspace(float(lo[dim_b]), float(hi[dim_b]), grid_n, dtype=torch.double)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    gx_flat = gx.reshape(-1)
    gy_flat = gy.reshape(-1)
    n_cells = grid_n * grid_n

    # One shared batch of "the rest of the dims", reused (broadcast) across
    # every grid cell -- common random numbers, drawn once per problem.
    W = problem.sample_uniform(mc_samples, seed=mc_seed).double()  # (mc_samples, d)

    Ygrid = torch.empty(grid_n, grid_n, mc_samples, m, dtype=torch.double)
    Ygrid_flat = Ygrid.reshape(n_cells, mc_samples, m)  # view, same storage

    chunk = max(1, min(n_cells, max_chunk_rows // max(1, mc_samples)))
    for start in range(0, n_cells, chunk):
        end = min(start + chunk, n_cells)
        c = end - start
        # (c, mc_samples, d): broadcast the shared W batch across this
        # chunk's cells, then overwrite the two plotted columns per-cell.
        Z = W.unsqueeze(0).expand(c, mc_samples, d).clone()
        Z[..., dim_a] = gx_flat[start:end].unsqueeze(-1).expand(c, mc_samples)
        Z[..., dim_b] = gy_flat[start:end].unsqueeze(-1).expand(c, mc_samples)
        Y = problem.evaluate_true(Z.reshape(c * mc_samples, d))
        Ygrid_flat[start:end] = Y.reshape(c, mc_samples, m).to(torch.double)

    return Ygrid, xs, ys, (dim_a, dim_b)


def _random_projection_basis(d: int, seed: int = _PROJECTION_SEED) -> torch.Tensor:
    """Fixed random ``(d, 2)`` orthonormal basis via QR of a Gaussian matrix."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    G = torch.randn(d, 2, dtype=torch.double, generator=gen)
    Q, _ = torch.linalg.qr(G)
    return Q


def _build_scatter_sample(
    problem: Problem, n_scatter: int, scatter_seed: int = _SCATTER_SEED
) -> tuple[torch.Tensor, torch.Tensor]:
    """Direct full-dimensional uniform sample, projected to a random 2D basis.

    Returns ``(Y, P)``: ``Y`` is ``(n_scatter, m)``, the noiseless objective
    values at a fresh ``problem.sample_uniform(n_scatter, seed=scatter_seed)``
    draw over the *full* domain (nothing pinned or marginalized); ``P`` is
    ``(n_scatter, 2)``, the mean-centered inputs projected onto a fixed
    random orthonormal basis (see :func:`_random_projection_basis`) -- purely
    a visualization choice, it does not affect feasibility.
    """
    X = problem.sample_uniform(n_scatter, seed=scatter_seed).double()
    Y = problem.evaluate_true(X).to(torch.double)
    Q = _random_projection_basis(problem.d)
    P = (X - X.mean(dim=0, keepdim=True)) @ Q
    return Y, P


def _topk_match_mask(prob_joint: torch.Tensor, p: float) -> tuple[torch.Tensor, int]:
    """Binary mask selecting the top ``round(p * n_cells)`` cells by value.

    Equivalent in spirit to thresholding at the ``(1 - p)``-quantile of
    ``prob_joint``, but robust to ties/degenerate quantiles (e.g. many cells
    at exactly 0 when ``p`` is small): it always selects exactly ``k`` cells
    (the ``k`` largest, ties broken arbitrarily), so the achieved green-area
    fraction is exactly ``k / n_cells`` regardless of the value distribution.
    """
    n_cells = prob_joint.numel()
    k = max(1, min(n_cells, round(p * n_cells)))
    flat = prob_joint.reshape(-1)
    _, top_idx = torch.topk(flat, k, largest=True)
    mask_flat = torch.zeros(n_cells, dtype=torch.bool)
    mask_flat[top_idx] = True
    return mask_flat.reshape(prob_joint.shape), k


def plot_problem_landscape(
    problem_name: str,
    out_path: str | Path,
    grid_n: int = _DEFAULT_GRID_N,
    mc_samples: int = _DEFAULT_MC_SAMPLES,
    n_scatter: int = _DEFAULT_N_SCATTER,
    thresholds_path: str | Path = DEFAULT_THRESHOLDS_PATH,
    difficulty_levels: Sequence[float] = _DIFFICULTY_LEVELS,
    verbose: bool = True,
) -> Path:
    """Render one landscape PDF for ``problem_name`` and save it to ``out_path``.

    Builds two reusable, evaluated-once data sources (see module docstring):

    - a ``(grid_n, grid_n, mc_samples, m)`` marginalized-grid tensor over the
      first two design dims (:func:`_build_marginalized_grid`);
    - an ``(n_scatter, m)`` + ``(n_scatter, 2)`` full-dimensional scatter
      sample and its random 2D projection (:func:`_build_scatter_sample`).

    The figure stacks one row per calibrated difficulty level in
    ``difficulty_levels`` (default: ``0.2, 0.1, 0.05, 0.01``, easiest first),
    each row re-thresholding the same two tensors with that level's
    calibrated ``tau`` (via :func:`itcas.pipeline.thresholds.load_thresholds`).
    Each row has ``m + 2`` panels: ``m`` per-objective feasibility-probability
    heatmaps, one marginalized joint-feasibility grid panel (top-k matched to
    the target area fraction), and one random-projection scatter panel (a
    direct, unmatched sample).

    If ``verbose``, prints one diagnostic line per row to stdout comparing
    target ``p`` against the achieved grid-panel area fraction, the raw
    (unmatched) spatial mean of the joint-feasibility probability field
    (sanity check per the law-of-total-probability argument in the module
    docstring), and the achieved scatter-panel feasible fraction.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    problem = _build_problem(problem_name)
    m = problem.m

    Ygrid, xs, ys, (dim_a, dim_b) = _build_marginalized_grid(problem, grid_n, mc_samples)
    Yscatter, Pscatter = _build_scatter_sample(problem, n_scatter)

    taus = [
        load_thresholds(problem_name, p, path=str(thresholds_path)).to(dtype=torch.double)
        for p in difficulty_levels
    ]

    n_rows = len(difficulty_levels)
    n_cols = m + 2  # m objective-probability panels + joint grid + scatter
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.3 * n_cols, 4.3 * n_rows), squeeze=False
    )

    x_np = xs.numpy()
    y_np = ys.numpy()
    kind_a = "design" if dim_a in problem.design_dims else "context"
    kind_b = "design" if dim_b in problem.design_dims else "context"

    for row, (p, th) in enumerate(zip(difficulty_levels, taus)):
        feas_per_obj_mc = Ygrid >= th.view(1, 1, 1, -1)      # (gn, gn, mc, m)
        prob_per_obj = feas_per_obj_mc.to(torch.double).mean(dim=2)  # (gn, gn, m)
        prob_joint = feas_per_obj_mc.all(dim=-1).to(torch.double).mean(dim=2)  # (gn, gn)

        for i in range(m):
            ax = axes[row][i]
            Pi = prob_per_obj[..., i].numpy()
            cf = ax.pcolormesh(
                x_np, y_np, Pi.T, cmap="viridis", vmin=0.0, vmax=1.0, shading="auto"
            )
            fig.colorbar(cf, ax=ax, shrink=0.8)
            title = f"f{i + 1}: P(f{i + 1} >= tau={float(th[i]):.3g})"
            if row == 0:
                title += f"\nvarying: x{dim_a} ({kind_a}), x{dim_b} ({kind_b})"
            ax.set_title(title, fontsize=9)
            ax.set_xlabel(f"x{dim_a}")
            ax.set_ylabel(f"x{dim_b}")

        # --- marginalized joint-grid panel: top-k (quantile) matched ---
        mask, k = _topk_match_mask(prob_joint, p)
        n_cells = prob_joint.numel()
        grid_achieved = k / n_cells
        raw_mean = float(prob_joint.mean())

        ax = axes[row][m]
        ax.pcolormesh(
            x_np, y_np, mask.numpy().T.astype(float), cmap="Greens",
            vmin=0.0, vmax=1.0, shading="auto",
        )
        ax.set_title(
            f"Joint feasible (marginalized grid)\ntarget p={p:g} "
            f"({p * 100:.1f}%), displayed green={grid_achieved * 100:.2f}%\n"
            f"(raw unthresholded mean={raw_mean * 100:.2f}%)",
            fontsize=9,
        )
        ax.set_xlabel(f"x{dim_a}")
        ax.set_ylabel(f"x{dim_b}")

        # --- random-projection scatter panel: direct sample, no matching ---
        feas_scatter = (Yscatter >= th.view(1, -1)).all(dim=-1)
        scatter_achieved = float(feas_scatter.to(torch.double).mean())
        ax = axes[row][m + 1]
        P_np = Pscatter.numpy()
        infeas = ~feas_scatter.numpy()
        feas = feas_scatter.numpy()
        ax.scatter(
            P_np[infeas, 0], P_np[infeas, 1], s=3, c="lightgray", alpha=0.35,
            linewidths=0, zorder=1,
        )
        ax.scatter(
            P_np[feas, 0], P_np[feas, 1], s=3, c="green", alpha=0.9,
            linewidths=0, zorder=2,
        )
        ax.set_title(
            f"Random-projection scatter (direct sample)\ntarget p={p:g} "
            f"({p * 100:.1f}%), scatter feasible={scatter_achieved * 100:.2f}%",
            fontsize=9,
        )
        ax.set_xlabel("proj0")
        ax.set_ylabel("proj1")

        if verbose:
            print(
                f"[{problem_name:18s}] p={p:<5g} grid_achieved="
                f"{grid_achieved * 100:6.2f}%  raw_mean={raw_mean * 100:6.2f}%  "
                f"scatter_achieved={scatter_achieved * 100:6.2f}%"
            )

        # Row label on the leftmost axis showing the difficulty level.
        axes[row][0].annotate(
            f"p={p:g} ({p * 100:.0f}%)",
            xy=(0, 0.5),
            xycoords="axes fraction",
            xytext=(-axes[row][0].yaxis.labelpad - 45, 0),
            textcoords="offset points",
            size=11,
            ha="right",
            va="center",
            rotation=90,
            fontweight="bold",
        )

    fig.suptitle(
        f"{problem.name}  --  2D slice varying design dims x{dim_a}, x{dim_b} "
        f"(d={problem.d}, m={problem.m})\n"
        f"per-objective + joint panels: marginalized over the other "
        f"{problem.d - 2} dims via {mc_samples} MC draws/cell "
        f"(grid {grid_n}x{grid_n}); joint panel top-k matched to target area\n"
        f"last panel: {n_scatter} full-dimensional uniform samples, random 2D "
        f"projection, direct (unmatched) feasibility sample\n"
        f"rows: calibrated difficulty levels (target global joint-feasible "
        f"fraction p), easiest at top",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.03, 0.0, 1.0, 0.88))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def summarize_landscapes(
    problems: Sequence[str],
    output_dir: str | Path = _DEFAULT_OUTPUT_DIR,
    grid_n: int = _DEFAULT_GRID_N,
    mc_samples: int = _DEFAULT_MC_SAMPLES,
    n_scatter: int = _DEFAULT_N_SCATTER,
    thresholds_path: str | Path = DEFAULT_THRESHOLDS_PATH,
) -> list[str]:
    """Render one ``{problem_name}_landscape.pdf`` per entry in ``problems``.

    Creates ``output_dir`` if missing. Returns the list of written PDF paths
    (as strings), in the same order as ``problems``.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for name in problems:
        out_path = out_dir / f"{name}_landscape.pdf"
        result = plot_problem_landscape(
            name,
            out_path,
            grid_n=grid_n,
            mc_samples=mc_samples,
            n_scatter=n_scatter,
            thresholds_path=thresholds_path,
        )
        paths.append(str(result))
    return paths


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser("itcas.landscape")
    parser.add_argument(
        "--problems",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Problem names from itcas.pipeline.problems.REGISTRY to plot "
            "(default: the 'problems' list in --config, e.g. "
            "configs/two_schools_of_thought.json)."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=_DEFAULT_CONFIG,
        help="Config file to read the default problem list from when --problems is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory to write '{problem}_landscape.pdf' files into.",
    )
    parser.add_argument(
        "--grid-n",
        type=int,
        default=_DEFAULT_GRID_N,
        help="Grid resolution per axis for the marginalized grid panels (grid-n x grid-n cells).",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=_DEFAULT_MC_SAMPLES,
        help="Monte-Carlo draws of the non-plotted dims averaged per grid cell.",
    )
    parser.add_argument(
        "--n-scatter",
        type=int,
        default=_DEFAULT_N_SCATTER,
        help="Number of full-dimensional uniform samples for the random-projection scatter panel.",
    )
    parser.add_argument(
        "--thresholds-path",
        type=str,
        default=DEFAULT_THRESHOLDS_PATH,
        help=(
            "JSON store of calibrated thresholds keyed by [problem][percentage] "
            "(default: itcas.pipeline.thresholds.DEFAULT_THRESHOLDS_PATH). All "
            "four difficulty levels (0.2, 0.1, 0.05, 0.01) must be present for "
            "each requested problem."
        ),
    )
    args = parser.parse_args(argv)

    problems = args.problems if args.problems else _load_default_problems(args.config)
    t0 = time.time()
    paths = summarize_landscapes(
        problems,
        args.output_dir,
        grid_n=args.grid_n,
        mc_samples=args.mc_samples,
        n_scatter=args.n_scatter,
        thresholds_path=args.thresholds_path,
    )
    elapsed = time.time() - t0
    for p in paths:
        print(p)
    print(f"[landscape] total wall-clock: {elapsed:.1f}s for {len(problems)} problem(s)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
