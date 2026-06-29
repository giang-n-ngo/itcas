"""Pure-Python sanity tests for math that does not need BoTorch.

Run: `python -m tests.test_smoke`. Heavier integration tests live in the
Slurm-launched smoke-run via `python -m itcas.cli --config configs/smoke.yaml`.
"""
from __future__ import annotations

import math


def test_qd_dpp_greedy_prefers_diverse_high_quality():
    import torch

    from itcas.algorithms.qd_dpp import build_qd_l_ensemble, greedy_dpp_batch

    # Three candidates: A and B are nearly identical in objective space;
    # C is far away. All have equal quality. A diverse batch should pick
    # one of {A, B} plus C, not {A, B}.
    mu = torch.tensor([[0.0, 0.0], [0.01, 0.0], [3.0, 3.0]])
    q = torch.tensor([1.0, 1.0, 1.0])
    L = build_qd_l_ensemble(q, mu, lam=1.0)
    sel = greedy_dpp_batch(L, batch_size=2)
    assert 2 in sel, f"expected diverse pick to include C; got {sel}"


def test_qd_dpp_context_kernel_enforces_context_diversity():
    import torch

    from itcas.algorithms.qd_dpp import build_qd_l_ensemble, greedy_dpp_batch

    # A and B are identical in objective space but live in different contexts;
    # C shares A's context but differs in objective space. With the joint
    # kernel, the batch of 2 should mix contexts rather than pick A and C
    # (same context). Indices: A=0, B=1, C=2.
    mu = torch.tensor([[0.0, 0.0], [0.0, 0.0], [3.0, 3.0]])
    ctx = torch.tensor([[0.0], [5.0], [0.0]])
    q = torch.tensor([1.0, 1.0, 1.0])
    L = build_qd_l_ensemble(q, mu, ctx=ctx, lam=1.0, lam_ctx=1.0)
    sel = greedy_dpp_batch(L, batch_size=2)
    assert 1 in sel, f"expected the distinct-context point B; got {sel}"


def test_qd_dpp_joint_kernel_reduces_to_objective_without_ctx():
    import torch

    from itcas.algorithms.qd_dpp import build_qd_l_ensemble

    mu = torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 3.0]])
    q = torch.tensor([1.0, 0.5, 0.8])
    L_no_ctx = build_qd_l_ensemble(q, mu, lam=1.0)
    L_none = build_qd_l_ensemble(q, mu, ctx=None, lam=1.0)
    assert torch.allclose(L_no_ctx, L_none)


def test_metrics_basic():
    import torch
    from itcas.metrics import (
        cumulative_positives, aup, positive_samples, is_feasible,
    )

    Y = torch.tensor([[1.0, 1.0], [-1.0, 1.0], [2.0, 2.0]])
    h = torch.tensor([0.0, 0.0])
    assert positive_samples(Y, h) == 2
    feas = is_feasible(Y, h).tolist()
    assert feas == [True, False, True]
    assert cumulative_positives(feas) == [1, 1, 2]
    # AUP = sum_t P(t) = 1 + 1 + 2 = 4 (a single number, not plotted).
    assert aup(feas) == 4


def test_fill_distance_metrics():
    import math

    import torch
    from itcas.metrics import (
        context_fill_distance,
        feasible_context_fill_distance,
    )

    # Context fill distance: a single sample at the box centre leaves the
    # corners of the unit square as the largest empty sphere (radius sqrt(2)/2).
    ref = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    samples = torch.tensor([[0.5, 0.5]])
    cfd = context_fill_distance(samples, ref, penalty=99.0)
    assert math.isclose(cfd, math.sqrt(0.5), rel_tol=1e-6)

    # Empty feasible set returns the penalty (e.g. context-space diagonal).
    empty = samples[:0]
    assert feasible_context_fill_distance(empty, ref, penalty=7.0) == 7.0



def test_objective_diversity_metrics():
    import math

    import torch
    from itcas.metrics import epsilon_archive_size, feasible_convex_hull_volume

    Y = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 5.0]], dtype=torch.double)

    # Raw-space points (0,0), (10,0), (0,5): 2D hull area is 0.5 * 10 * 5 = 25.0.
    fchv = feasible_convex_hull_volume(Y)
    assert math.isclose(fchv, 25.0, rel_tol=1e-6)
    assert feasible_convex_hull_volume(Y[:2]) == 0.0

    # ε-Archive Size: spread points (raw dist=sqrt(125) >> eps) → 2; duplicates → 1.
    spread = torch.tensor([[0.0, 0.0], [10.0, 5.0]], dtype=torch.double)
    duplicate = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.double)
    assert epsilon_archive_size(spread, eps=0.05) == 2
    assert epsilon_archive_size(duplicate, eps=0.05) == 1
    # Spread has strictly more diversity than duplicate set.
    assert epsilon_archive_size(spread) > epsilon_archive_size(duplicate)
    assert epsilon_archive_size(Y[:0]) == 0


def test_binary_entropy_bounds():
    from itcas.algorithms.roi_mi import _binary_entropy
    import torch

    p = torch.tensor([0.0, 0.5, 1.0])
    H = _binary_entropy(p)
    assert math.isclose(H[1].item(), 1.0, abs_tol=1e-6)
    assert H[0].item() < 1e-4 and H[2].item() < 1e-4


def test_synthetic_benchmarks_shapes_and_feasibility():
    import torch

    from itcas.pipeline.problems import REGISTRY

    specs = {
        "sphere2_6d": (6, 2, (3, 4, 5)),
        "rosenbrock_sphere_6d": (6, 2, (4, 5)),
        "multimodal_trap_20d": (20, 3, tuple(range(10, 20))),
        "dtlz2_6d": (6, 4, (3, 4, 5)),
        "zdt3_6d": (6, 2, (1, 2, 3, 4, 5)),
        "levy_16d": (16, 3, tuple(range(8, 16))),
        "dtlz3_8d": (8, 4, (3, 4, 5, 6, 7)),
        "vlmop2_6d": (6, 2, (3, 4, 5)),
        "dtlz4_12d": (12, 3, tuple(range(2, 12))),
        "dixon_price_10d": (10, 2, (5, 6, 7, 8, 9)),
        "griewank_16d": (16, 4, tuple(range(8, 16))),
        "alpine_12d": (12, 3, tuple(range(6, 12))),
        "ackley_rosenbrock_6d": (6, 2, (3, 4, 5)),
        "rastrigin_griewank_sphere_20d": (20, 3, tuple(range(10, 20))),
        "styblinski_tang_levy_10d": (10, 2, (5, 6, 7, 8, 9)),
        "heterogeneous_quadrants_6d": (6, 4, (3, 4, 5)),
        "ellipsoid_rastrigin_20d": (20, 2, tuple(range(10, 20))),
        "dixon_rosenbrock_sphere_12d": (12, 3, tuple(range(6, 12))),
        "zakharov_ackley_10d": (10, 2, (5, 6, 7, 8, 9)),
        "all_valley_8d": (8, 4, (4, 5, 6, 7)),
        "schwefel_styblinski_sphere_20d": (20, 3, tuple(range(10, 20))),
        "griewank_vlmop2_16d": (16, 3, tuple(range(8, 16))),
    }
    for name, (d, m, ctx) in specs.items():
        p = REGISTRY[name]()
        assert p.d == d and p.m == m
        assert p.context_dims == ctx
        assert p.bounds.shape == (2, d)
        X = p.sample_uniform(64, seed=0)
        Y = p.evaluate(X)
        assert Y.shape == (64, m)
        assert torch.isfinite(Y).all()
        # Negated (maximization) form: feasible region non-empty over a sample.
        big = p.sample_uniform(50000, seed=1)
        feas = (p.fn(big) >= p.thresholds).all(dim=-1)
        assert feas.any(), f"{name} has empty feasible region at default tau"


def test_dtlz2_pareto_front_identity():
    import torch

    from itcas.pipeline.problems import dtlz2_6d

    # With w = 0.5 -> g = 0, the negated DTLZ2 objectives satisfy
    # sum_i f_i^2 == 1 exactly (unit sphere Pareto front).
    p = dtlz2_6d()
    Z = torch.rand(100, 6)
    Z[:, 3:] = 0.5
    Y = p.fn(Z)
    assert torch.allclose((Y ** 2).sum(dim=-1), torch.ones(100), atol=1e-5)


def test_sphere2_optima_locations():
    import torch

    from itcas.pipeline.problems import sphere2_6d

    p = sphere2_6d()
    z0 = torch.zeros(1, 6)
    z2 = torch.full((1, 6), 2.0)
    assert torch.allclose(p.fn(z0)[0, 0], torch.tensor(0.0))
    assert torch.allclose(p.fn(z2)[0, 1], torch.tensor(0.0))


def test_only_itcas_runs_in_batch():
    from itcas.pipeline.loop import effective_batch_size

    # The proposed algorithm uses the configured batch size...
    assert effective_batch_size("itcas", 4) == 4
    assert effective_batch_size("itcas", 1) == 1
    # ...every baseline is forced to a single evaluation per iteration.
    for m in [
        "random", "one_step", "ez", "eisr", "straddle",
        "cas_eci", "moc_cas_hard", "moc_cas_soft",
        "eps_constraint", "moo_cluster",
    ]:
        assert effective_batch_size(m, 4) == 1, m


def test_visualization_writes_per_metric_pdfs():
    import json
    import tempfile
    from pathlib import Path

    from itcas.reporting.visualize import visualize_benchmark
    from itcas.reporting.metrics import REGISTRY as METRIC_REGISTRY

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        def write_run(run_name, method, init_pos, feas_rows):
            (root / f"{run_name}.jsonl").write_text(
                "\n".join(json.dumps(row) for row in feas_rows) + "\n"
            )
            (root / f"{run_name}.summary.json").write_text(json.dumps({
                "config": {"method": method, "n_init": 2, "seed": 0, "radius": 0.1},
                "problem": "sphere2_6d",
                "n_init": 2,
                "n_init_positives": init_pos,
                "thresholds": [-30.0, -30.0],
                "context_dims": [3, 4, 5],
                "init_X": [[0.0] * 6, [0.1] * 6],
                "init_Y": [[-1.0, -1.0], [-2.0, -2.0]],
                "init_feasible": [True, True],
            }))

        write_run(
            "itcas_run",
            "itcas",
            1,
            [
                {"step": 1, "n_eval_total": 6, "n_eval_this_iter": 4,
                 "feasible": [True, False, True, False],
                 "x": [[0.2] * 6, [0.3] * 6, [0.4] * 6, [0.5] * 6],
                 "y": [[-1.0, -1.0], [-100.0, -1.0], [-1.0, -1.0], [-100.0, -1.0]]},
                {"step": 2, "n_eval_total": 10, "n_eval_this_iter": 4,
                 "feasible": [False, False, True, False],
                 "x": [[0.6] * 6, [0.7] * 6, [0.8] * 6, [0.9] * 6],
                 "y": [[-100.0, -1.0], [-100.0, -1.0], [-1.0, -1.0], [-100.0, -1.0]]},
            ],
        )
        write_run(
            "random_run",
            "random",
            0,
            [
                {"step": 1, "n_eval_total": 3, "n_eval_this_iter": 1,
                 "feasible": [False],
                 "x": [[0.15] * 6], "y": [[-100.0, -1.0]]},
                {"step": 2, "n_eval_total": 4, "n_eval_this_iter": 1,
                 "feasible": [True],
                 "x": [[0.25] * 6], "y": [[-1.0, -1.0]]},
            ],
        )

        out = visualize_benchmark(root, benchmark="sphere2_6d")
        # Two PDFs (vs_evaluations + vs_steps) per metric in REGISTRY.
        assert len(out) >= 2 * len(METRIC_REGISTRY) - 4, out  # allow grid metrics to skip
        for p in out:
            assert Path(p).exists()
            assert Path(p).suffix == ".pdf"
        # Per-metric file naming sanity check.
        names = {Path(p).name for p in out}
        assert any("cumulative_positives_vs_evaluations.pdf" in n for n in names)
        assert any("cumulative_positives_vs_steps.pdf" in n for n in names)


def test_run_scatter_plots_generated_for_2d_spaces():
    import tempfile
    from pathlib import Path

    import torch

    from itcas.pipeline.loop import _plot_run_scatter

    with tempfile.TemporaryDirectory() as tmp:
        X = torch.rand(12, 5, dtype=torch.double)
        Y = torch.rand(12, 2, dtype=torch.double)
        out = _plot_run_scatter(
            X=X,
            Y=Y,
            thresholds=torch.tensor([0.5, 0.5], dtype=torch.double),
            n_init=4,
            context_dims=(1, 4),
            out_dir=tmp,
            run_name="plot2d",
        )
        assert len(out) == 1
        assert Path(tmp, "plot2d.spaces2d.png").exists()


def test_qd_dpp_selects_under_small_uniform_quality():
    """Regression: the p(z) fallback yields tiny but positive quality. The DPP
    must still return a full batch (not break early) so the search does not
    stall and terminate the experiment prematurely."""
    import torch

    from itcas.algorithms.qd_dpp import build_qd_l_ensemble, greedy_dpp_batch

    mu = torch.randn(8, 2)
    q = torch.full((8,), 1e-3)  # small positive feasibility-probability proxy
    L = build_qd_l_ensemble(q, mu, lam=1.0)
    sel = greedy_dpp_batch(L, batch_size=4)
    assert len(sel) == 4, f"expected a full batch under small quality; got {sel}"


def test_roi_mi_degenerate_falls_back_to_feasibility_prob():
    """When the Thompson reference set is empty (q == 0 everywhere), quality
    must fall back to the joint feasibility probability rather than stay zero."""
    import torch

    from itcas.algorithms import roi_mi

    mu = torch.tensor([[2.0, 2.0], [-2.0, -2.0]])
    sigma = torch.ones(2, 2)
    h = torch.zeros(2)
    expected = roi_mi.joint_feasibility_probability(
        roi_mi.feasibility_probabilities(mu, sigma, h)
    )

    orig_mean_std = roi_mi.posterior_mean_std
    orig_ref = roi_mi.thompson_reference_set
    roi_mi.posterior_mean_std = lambda models, cand: (mu, sigma)
    roi_mi.thompson_reference_set = lambda *a, **k: torch.zeros(2, dtype=torch.bool)
    try:
        q, _ = roi_mi.roi_mi_quality(models=[None, None], cand=torch.zeros(2, 3), h=h)
    finally:
        roi_mi.posterior_mean_std = orig_mean_std
        roi_mi.thompson_reference_set = orig_ref

    assert bool((q > 0).any()), "degenerate ROI should fall back to p(z) > 0"
    assert torch.allclose(q, expected)


def test_efig_quality_is_pof_weighted_info_gain_and_differentiable():
    """EFIG must equal PoF(z) * sum_i 0.5 log(1 + sigma_i^2/sigma_eps_i^2),
    be non-negative, and be differentiable w.r.t. z."""
    import torch

    from itcas.algorithms import efig_quality
    from itcas.algorithms.roi_mi import (
        feasibility_probabilities,
        joint_feasibility_probability,
    )
    from itcas.utils.gp import build_independent_gps, observation_noise

    torch.manual_seed(0)
    X = torch.rand(14, 2, dtype=torch.double)
    Y = torch.rand(14, 2, dtype=torch.double)
    bounds = torch.stack([torch.zeros(2), torch.ones(2)]).double()
    models = build_independent_gps(X, Y, bounds=bounds)
    h = torch.tensor([0.4, 0.4], dtype=torch.double)

    z = torch.rand(6, 2, dtype=torch.double, requires_grad=True)
    q = efig_quality(models, z, h)

    # Closed form: q(z) = PoF(z) * sum_i 0.5 log(1 + sigma_i^2 / sigma_eps_i^2).
    mus, sigmas, info_gain = [], [], torch.zeros(6, dtype=torch.double)
    for gp in models:
        post = gp.posterior(z)
        mu_i = post.mean.squeeze(-1)
        var_i = post.variance.clamp_min(1e-12).squeeze(-1)
        mus.append(mu_i)
        sigmas.append(var_i.sqrt())
        info_gain = info_gain + 0.5 * torch.log1p(var_i / observation_noise(gp))
    mu = torch.stack(mus, dim=-1)
    sigma = torch.stack(sigmas, dim=-1)
    pof = joint_feasibility_probability(feasibility_probabilities(mu, sigma, h))
    expected = pof * info_gain
    assert torch.allclose(q, expected)

    # PoF in [0, 1] and info gain >= 0, so EFIG is non-negative.
    assert float(q.min().detach()) >= 0.0

    # Differentiable w.r.t. z (required by the continuous QD-DPP optimizer).
    g = torch.autograd.grad(q.sum(), z)[0]
    assert g.shape == z.shape and torch.isfinite(g).all()


def test_quality_registry_exposes_roi_and_efig_variants():
    from itcas.algorithms import QUALITY_REGISTRY, available_qualities

    names = available_qualities()
    assert "roi_mi" in names and "efig" in names
    assert set(names) == set(QUALITY_REGISTRY)


def test_smooth_margin_recovers_min_as_gamma_to_zero():
    import torch

    from itcas.algorithms.continuous import smooth_margin

    f = torch.tensor([[1.0, 3.0], [-1.0, 2.0], [0.5, 0.5]])
    tau = torch.tensor([0.0, 0.0])
    hard_min = (f - tau).min(dim=-1).values
    soft = smooth_margin(f, tau, gamma=1e-3)
    # As gamma -> 0 the softmin margin converges to the hard min margin.
    assert torch.allclose(soft, hard_min, atol=1e-2)
    # The softmin always lower-bounds the hard min (conservative feasibility).
    assert torch.all(smooth_margin(f, tau, gamma=1.0) <= hard_min + 1e-9)
    # M > 0 iff strictly feasible under the (gamma->0) limit.
    feasible = (f > tau).all(dim=-1)
    assert torch.equal(soft > 0, feasible)


def test_smooth_margin_rejects_nonpositive_gamma():
    import torch

    from itcas.algorithms.continuous import smooth_margin

    f = torch.zeros(2, 2)
    tau = torch.zeros(2)
    for g in (0.0, -1.0):
        try:
            smooth_margin(f, tau, gamma=g)
            raise AssertionError("expected ValueError for non-positive gamma")
        except ValueError:
            pass


def test_qd_marginal_gain_penalizes_duplicates():
    import torch

    from itcas.algorithms.continuous import _qd_marginal_gain

    # One point already selected with quality q=0.7 at objective mu=[0,0].
    qB = torch.tensor([0.7])
    muB = torch.tensor([[0.0, 0.0]])
    # Candidate A duplicates the selected point; candidate B is far away in
    # objective space. Both share the same quality.
    qN = torch.tensor([0.7, 0.7])
    muN = torch.tensor([[0.0, 0.0], [5.0, 5.0]])
    gain = _qd_marginal_gain(qN, muN, None, qB, muB, None, lam_obj=1.0, lam_ctx=None)
    # The far-away candidate must have a strictly larger marginal gain.
    assert gain[1] > gain[0]
    # And the empty-batch gain equals log(1 + q^2).
    g0 = _qd_marginal_gain(qN, muN, None, None, None, None, lam_obj=1.0, lam_ctx=None)
    assert torch.allclose(g0, torch.log1p(qN.pow(2)))


def test_multistart_ascent_finds_quadratic_peak():
    import torch

    from itcas.algorithms.continuous import multistart_ascent

    bounds = torch.tensor([[-2.0, -2.0], [2.0, 2.0]], dtype=torch.double)
    target = torch.tensor([0.5, -0.5], dtype=torch.double)

    def obj(z):
        return -((z - target) ** 2).sum(dim=-1)

    Z, vals = multistart_ascent(obj, bounds, n_restarts=8, n_steps=200, lr=0.1, seed=0)
    best = Z[vals.argmax()]
    assert torch.allclose(best, target, atol=1e-2)


def test_resolve_comparison_root_walks_up_to_method_lca():
    import json
    import tempfile
    from pathlib import Path

    from itcas.reporting.visualize import resolve_comparison_root

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Cluster-style layout: <root>/<problem>/<difficulty>/<method>/<run>.summary.json
        for method in ("itcas", "random"):
            d = root / "sphere2_6d" / "p0_05" / method
            d.mkdir(parents=True)
            (d / f"run_{method}.summary.json").write_text(json.dumps({
                "config": {"method": method}, "problem": "sphere2_6d",
            }))

        # From a per-method leaf the comparison root is the difficulty dir,
        # whether one or many methods are already present alongside.
        leaf = root / "sphere2_6d" / "p0_05" / "itcas"
        got = resolve_comparison_root(leaf, problem="sphere2_6d")
        assert got == (root / "sphere2_6d" / "p0_05").resolve(), got

        # Flat layout: start already aggregates >=2 methods -> stay.
        flat = root / "flat"
        flat.mkdir()
        for method in ("itcas", "random"):
            (flat / f"r_{method}.summary.json").write_text(json.dumps({
                "config": {"method": method}, "problem": "two_circles_2d",
            }))
        got = resolve_comparison_root(flat, problem="two_circles_2d")
        assert got == flat.resolve(), got

        # Orphan (no runs anywhere upstream) -> fall back to start.
        orphan = root / "nope" / "p" / "itcas"
        orphan.mkdir(parents=True)
        got = resolve_comparison_root(orphan, problem="x")
        assert got == orphan.resolve(), got


def test_threshold_calibration_hits_target_fraction():
    import torch

    from itcas.pipeline.problems import sphere2_6d
    from itcas.pipeline.thresholds import calibrate_thresholds

    problem = sphere2_6d()
    for target in (0.20, 0.10, 0.05):
        res = calibrate_thresholds(
            problem, target_fraction=target, n_samples=40000, seed=0
        )
        # Achieved joint-feasible fraction is close to the requested target.
        assert abs(res.achieved_fraction - target) < 0.01, res
        # Thresholds lie below the per-objective maxima (feasible set non-empty).
        tau = torch.tensor(res.thresholds)
        mx = torch.tensor(res.maxima)
        assert torch.all(tau <= mx)
        # Smaller target => harder => higher thresholds (monotone difficulty).
    r10 = calibrate_thresholds(problem, target_fraction=0.10, n_samples=40000, seed=0)
    r05 = calibrate_thresholds(problem, target_fraction=0.05, n_samples=40000, seed=0)
    assert all(a <= b for a, b in zip(r10.thresholds, r05.thresholds))


def test_threshold_save_load_roundtrip():
    import tempfile
    from pathlib import Path

    import torch

    from itcas.pipeline.problems import sphere2_6d
    from itcas.pipeline.thresholds import (
        calibrate_thresholds, save_calibration, load_thresholds,
    )

    problem = sphere2_6d()
    res = calibrate_thresholds(problem, target_fraction=0.10, n_samples=20000, seed=0)
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "thresholds.json")
        save_calibration(res, path=path)
        tau = load_thresholds("sphere2_6d", percentage=0.10, path=path)
        assert torch.allclose(tau, torch.tensor(res.thresholds, dtype=torch.double))
        try:
            load_thresholds("sphere2_6d", percentage=0.999, path=path)
            raise AssertionError("expected KeyError for missing percentage")
        except KeyError:
            pass


def test_resolve_device_specs():
    import torch

    from itcas.utils.device import resolve_device

    assert resolve_device("cpu").type == "cpu"
    # "auto" falls back to CPU when CUDA is absent.
    auto = resolve_device("auto")
    assert auto.type in ("cpu", "cuda")
    if not torch.cuda.is_available():
        assert auto.type == "cpu"
        # Explicit CUDA request errors clearly when no GPU is visible.
        try:
            resolve_device("cuda")
            raise AssertionError("expected RuntimeError for unavailable CUDA")
        except RuntimeError:
            pass
    # Passing through a concrete device is a no-op.
    dev = torch.device("cpu")
    assert resolve_device(dev) is dev


def test_sample_uniform_device_and_reproducibility():
    import torch

    from itcas.pipeline.problems import sphere2_6d

    p = sphere2_6d()
    a = p.sample_uniform(16, seed=0, device="cpu", dtype=torch.double)
    b = p.sample_uniform(16, seed=0, device="cpu", dtype=torch.double)
    assert a.dtype == torch.double and a.device.type == "cpu"
    # CPU-side draw is deterministic regardless of target device.
    assert torch.allclose(a, b)


def test_evaluate_true_matches_input_device_and_dtype():
    import torch

    from itcas.pipeline.problems import sphere2_6d

    p = sphere2_6d()
    X = p.sample_uniform(8, seed=1, device="cpu", dtype=torch.double)
    Y = p.evaluate_true(X)
    assert Y.device == X.device and Y.dtype == X.dtype
    assert torch.allclose(Y, p.fn(X.cpu()).to(X))


def test_parse_seed_spec_ranges_and_singletons():
    from itcas.utils.seeds import parse_seed_spec

    assert parse_seed_spec("1-10") == list(range(1, 11))
    assert parse_seed_spec("1,3,6-8") == [1, 3, 6, 7, 8]
    assert parse_seed_spec("0-2, 5") == [0, 1, 2, 5]
    # De-duplication and sorting.
    assert parse_seed_spec("5,1,5,3-4") == [1, 3, 4, 5]
    # Iterable input passthrough.
    assert parse_seed_spec([4, 2, 2, 1]) == [1, 2, 4]
    # Descending range is an error.
    try:
        parse_seed_spec("5-1")
        raise AssertionError("expected ValueError for descending range")
    except ValueError:
        pass


def test_run_name_for_seed_template_and_suffix():
    from itcas.utils.seeds import run_name_for_seed

    assert run_name_for_seed("demo", 4) == "demo_seed4"
    assert run_name_for_seed("p_{seed}_run", 4) == "p_4_run"


def test_pending_seeds_skips_completed():
    import json
    import tempfile
    from pathlib import Path

    from itcas.utils.seeds import (
        pending_seeds, run_name_for_seed, is_run_complete, summary_path,
    )

    with tempfile.TemporaryDirectory() as tmp:
        out = str(Path(tmp))
        # Mark seeds 1, 3, 6, 7 as completed by writing valid summary files.
        for s in (1, 3, 6, 7):
            rn = run_name_for_seed("demo", s)
            Path(summary_path(out, rn)).write_text(json.dumps({"ok": True}))
            assert is_run_complete(out, rn)

        todo = pending_seeds(range(1, 11), out, "demo")
        assert todo == [2, 4, 5, 8, 9, 10]
        # --force ignores existing summaries.
        assert pending_seeds(range(1, 11), out, "demo", force=True) == list(range(1, 11))

        # A corrupt summary counts as incomplete.
        bad = run_name_for_seed("demo", 1)
        Path(summary_path(out, bad)).write_text("{ not json")
        assert not is_run_complete(out, bad)
        assert 1 in pending_seeds(range(1, 11), out, "demo")


def test_run_lock_prevents_concurrent_runs(tmp_path=None):
    """A second RunLock on the same (out_dir, run_name) must fail fast."""
    import tempfile
    from itcas.utils.seeds import RunBusyError, RunLock

    with tempfile.TemporaryDirectory() as out:
        with RunLock(out, "demo"):
            try:
                with RunLock(out, "demo"):
                    raise AssertionError("second RunLock unexpectedly succeeded")
            except RunBusyError:
                pass
        # After release, a new lock should be acquirable.
        with RunLock(out, "demo"):
            pass


if __name__ == "__main__":
    test_qd_dpp_greedy_prefers_diverse_high_quality()
    test_qd_dpp_context_kernel_enforces_context_diversity()
    test_qd_dpp_joint_kernel_reduces_to_objective_without_ctx()
    test_metrics_basic()
    test_fill_distance_metrics()
    test_binary_entropy_bounds()
    test_synthetic_benchmarks_shapes_and_feasibility()
    test_dtlz2_pareto_front_identity()
    test_sphere2_optima_locations()
    test_only_itcas_runs_in_batch()
    test_visualization_writes_per_metric_pdfs()
    test_qd_dpp_selects_under_small_uniform_quality()
    test_roi_mi_degenerate_falls_back_to_feasibility_prob()
    test_smooth_margin_recovers_min_as_gamma_to_zero()
    test_smooth_margin_rejects_nonpositive_gamma()
    test_qd_marginal_gain_penalizes_duplicates()
    test_multistart_ascent_finds_quadratic_peak()
    test_resolve_comparison_root_walks_up_to_method_lca()
    test_threshold_calibration_hits_target_fraction()
    test_threshold_save_load_roundtrip()
    test_resolve_device_specs()
    test_sample_uniform_device_and_reproducibility()
    test_evaluate_true_matches_input_device_and_dtype()
    test_parse_seed_spec_ranges_and_singletons()
    test_run_name_for_seed_template_and_suffix()
    test_pending_seeds_skips_completed()
    print("OK: smoke unit tests passed.")
