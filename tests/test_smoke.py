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

    # ε-Archive Size operates on y' = log1p(y - thresholds) (contexts/metrics.md
    # §4), not raw y. With thresholds=0, log1p(y) still separates these two
    # points by far more than eps=0.05, so this still nets 2 vs. 1.
    tau0 = torch.zeros(2, dtype=torch.double)
    spread = torch.tensor([[0.0, 0.0], [10.0, 5.0]], dtype=torch.double)
    duplicate = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.double)
    assert epsilon_archive_size(spread, thresholds=tau0, eps=0.05) == 2
    assert epsilon_archive_size(duplicate, thresholds=tau0, eps=0.05) == 1
    # Spread has strictly more diversity than duplicate set.
    assert epsilon_archive_size(spread, thresholds=tau0) > epsilon_archive_size(
        duplicate, thresholds=tau0
    )
    assert epsilon_archive_size(Y[:0], thresholds=tau0) == 0


def test_epsilon_archive_size_uses_transformed_not_raw_distance():
    """Regression test for the unit-consistency bug: skipping the log1p(y -
    thresholds) transform makes eps-Archive Size collapse onto Number of
    Positives whenever the feasible margin (y - tau) is large, because a raw
    Euclidean distance that looks large is actually a near-duplicate once
    compressed by log1p.

    tau = [100, 100]; p1 = [200, 100], p2 = [200.5, 100].
    Raw distance ||p1 - p2|| = 0.5, comfortably >= eps=0.05 -- a raw-space
    archive would (wrongly) admit both as distinct.
    Transformed: y'_1 = [log1p(100), log1p(0)] ~= [4.6151, 0.0],
                 y'_2 = [log1p(100.5), log1p(0)] ~= [4.6201, 0.0].
    Transformed distance ~= 0.005 < eps=0.05 -- the correct (spec-following)
    archive collapses these to a single entry.
    """
    import math

    import torch
    from itcas.metrics import epsilon_archive_size, transform_feasible_for_archive

    tau = torch.tensor([100.0, 100.0], dtype=torch.double)
    p1 = torch.tensor([200.0, 100.0], dtype=torch.double)
    p2 = torch.tensor([200.5, 100.0], dtype=torch.double)
    Y = torch.stack([p1, p2])

    raw_dist = float(torch.linalg.norm(p1 - p2).item())
    assert raw_dist >= 0.05, "fixture must exercise the large-raw-distance regime"

    transformed = transform_feasible_for_archive(Y, tau)
    transformed_dist = float(torch.linalg.norm(transformed[0] - transformed[1]).item())
    assert transformed_dist < 0.05, "log1p(y - tau) must compress this pair below eps"
    assert math.isclose(transformed_dist, 0.005, abs_tol=5e-4)

    # The production function must follow the transformed distance (archive
    # size 1, i.e. p2 recognised as a near-duplicate of p1), not the raw one
    # (which would wrongly report 2).
    assert epsilon_archive_size(Y, thresholds=tau, eps=0.05) == 1


def test_tune_eps_archive_percentile_hand_checked():
    """eps_from_pool: 10 collinear points spaced by 1 -> pairwise distances
    are the integers 1..9 (distance d occurs 10-d times, 45 pairs total).
    ``np.percentile(..., 5)`` on that multiset is exactly 1.0 (the smallest
    distance value, since it already covers > 5% of the 45-pair mass)."""
    import torch
    from itcas.reporting.tune_eps_archive import eps_from_pool

    pooled = torch.tensor([[float(i)] for i in range(10)], dtype=torch.double)
    eps = eps_from_pool(pooled, percentile=5.0, min_points=10)
    assert eps == 1.0

    # Fewer than min_points pooled points -> skip (None), even though pdist
    # itself would happily run on 2+ points.
    assert eps_from_pool(pooled[:5], percentile=5.0, min_points=10) is None
    assert eps_from_pool(pooled[:1], percentile=5.0, min_points=1) is None  # pdist needs >= 2


def test_tune_eps_archive_caps_pdist_via_subsampling():
    """eps_from_pool: pools larger than max_pool_size are randomly subsampled
    (fixed seed) before pdist, so pairwise-distance count stays bounded by
    max_pool_size instead of growing O(n^2) with the full pool."""
    import torch
    from itcas.reporting.tune_eps_archive import eps_from_pool

    # 200 collinear points spaced by 1; full pdist would have 200*199/2=19900
    # pairs, but capping at max_pool_size=20 subsamples down to <= 20*19/2=190
    # pairs. The subsampled 5th percentile should still land near the true
    # nearest-neighbour spacing (1.0) since the points are evenly spaced.
    pooled = torch.tensor([[float(i)] for i in range(200)], dtype=torch.double)
    eps = eps_from_pool(pooled, percentile=5.0, min_points=10, max_pool_size=20)
    assert eps is not None and eps > 0.0
    # Reproducible: same inputs -> same subsample -> same eps.
    eps2 = eps_from_pool(pooled, percentile=5.0, min_points=10, max_pool_size=20)
    assert eps == eps2


def test_tune_eps_archive_pools_log_transforms_and_filters_infeasible():
    """pooled_transformed_feasible: only strictly-feasible rows (y_i >= tau_i for
    every objective) are pooled, and each is transformed via
    y' = log(1 + (y - tau)) using its own run's own thresholds.

    ``final_states`` is the lightweight ``(thresholds, Y_final)`` tuple that
    :func:`load_run_final` produces — no ``RunSeries``/per-step history
    needed for calibration (see the module docstring: this avoids the O(T^2)
    per-step reconstruction that ``visualize._load_run`` does for plotting)."""
    import math

    import torch
    from itcas.reporting.tune_eps_archive import (
        compute_eps_for_group,
        pooled_transformed_feasible,
    )

    tau = torch.tensor([1.0, 1.0], dtype=torch.double)
    # Row 0 feasible (2,2); row 1 infeasible (0,0) — must be dropped from the pool.
    run_a = (tau, torch.tensor([[2.0, 2.0], [0.0, 0.0]], dtype=torch.double))
    run_b = (tau, torch.tensor([[3.0, 1.0]], dtype=torch.double))

    pooled = pooled_transformed_feasible([run_a, run_b])
    assert pooled is not None
    assert pooled.shape == (2, 2)
    expected = torch.tensor(
        [[math.log1p(1.0), math.log1p(1.0)], [math.log1p(2.0), math.log1p(0.0)]],
        dtype=torch.double,
    )
    assert torch.allclose(pooled, expected)

    # Too few pooled feasible points (2 < default min_points=10) -> skip.
    assert compute_eps_for_group([run_a, run_b]) is None

    # No feasible points anywhere in the group -> None.
    run_c = (tau, torch.tensor([[0.0, 0.0]], dtype=torch.double))
    assert pooled_transformed_feasible([run_c]) is None


def test_tune_eps_archive_load_run_final_dedupes_steps_no_per_step_history(tmp_path=None):
    """load_run_final: parses a run's .jsonl + .summary.json into just the
    final (thresholds, Y) pair, deduping repeated step writes (keeping the
    last one seen) and truncating at the first gap in the step sequence —
    matching visualize._load_run's contiguous-prefix rule — without building
    any per-step snapshot list."""
    import json
    import tempfile
    from pathlib import Path

    import torch
    from itcas.reporting.tune_eps_archive import load_run_final

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        jsonl_path = root / "run.jsonl"
        summary_path = root / "run.summary.json"
        summary_path.write_text(json.dumps({
            "thresholds": [0.0, 0.0],
            "init_Y": [[1.0, 1.0]],
        }))
        with jsonl_path.open("w") as f:
            f.write(json.dumps({"step": 1, "y": [[2.0, 2.0]]}) + "\n")
            # Duplicate write of step 1 (restarted writer) — last one wins.
            f.write(json.dumps({"step": 1, "y": [[2.5, 2.5]]}) + "\n")
            f.write(json.dumps({"step": 2, "y": [[3.0, 3.0]]}) + "\n")
            # Gap at step 4 (step 3 missing) — everything from here on is
            # truncated, matching the "contiguous prefix from step 1" rule.
            f.write(json.dumps({"step": 4, "y": [[9.0, 9.0]]}) + "\n")

        loaded = load_run_final(jsonl_path)
        assert loaded is not None
        thresholds, Y = loaded
        assert torch.allclose(thresholds, torch.tensor([0.0, 0.0], dtype=torch.double))
        expected_Y = torch.tensor(
            [[1.0, 1.0], [2.5, 2.5], [3.0, 3.0]], dtype=torch.double,
        )
        assert torch.allclose(Y, expected_Y)

    # Missing summary.json -> None.
    with tempfile.TemporaryDirectory() as tmp:
        missing = Path(tmp) / "orphan.jsonl"
        missing.write_text("")
        assert load_run_final(missing) is None


def test_tune_eps_archive_difficulty_dir_to_config_key():
    from itcas.reporting.tune_eps_archive import config_key_from_difficulty_dir as key_of

    assert key_of("p0_01") == "0.01"
    assert key_of("p0_05") == "0.05"
    assert key_of("p0_1") == "0.1"
    assert key_of("p0_2") == "0.2"
    assert key_of("p1") == "1"
    assert key_of("p10") == "10"


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
    # ...every sequential baseline (including the new Family-B continuous
    # baselines, and itcas_seq -- the forced-sequential sibling of itcas) is
    # forced to a single evaluation per iteration...
    for m in [
        "random", "one_step", "ez", "eisr", "straddle",
        "cas_eci", "moc_cas_hard", "moc_cas_soft",
        "eps_constraint", "moo_cluster",
        "c2lse", "bes", "itcas_seq",
    ]:
        assert effective_batch_size(m, 4) == 1, m
    # ...while any "_batch"-suffixed sibling (discrete-pool DPP or continuous
    # DPP) uses the configured batch size, just like itcas.
    for m in [
        "random_batch", "straddle_batch", "cas_eci_batch", "moc_cas_hard_batch",
        "c2lse_batch", "bes_batch",
    ]:
        assert effective_batch_size(m, 4) == 4, m
        assert effective_batch_size(m, 1) == 1, m


def test_cr_ndig_is_continuous_and_forced_sequential():
    from itcas.pipeline.loop import _is_continuous_method, effective_batch_size

    assert _is_continuous_method("cr_ndig")
    assert effective_batch_size("cr_ndig", 4) == 1


def test_cr_ndig_quality_repels_previously_evaluated_contexts():
    """A candidate whose context sits exactly on top of a previously evaluated
    context must score strictly lower under cr_ndig than an otherwise-identical
    candidate (same design coordinate) whose context is far from every prior
    context -- the core repulsion behavior of contexts/sequential_ndig.md."""
    import torch
    from itcas.algorithms.quality import build_quality_fn

    models, bounds, tau = _tiny_gp_setup(m=2, d=2)
    context_dims = (1,)

    # Two previously evaluated points (design, context) pairs.
    X_obs = torch.tensor([[0.5, 0.30], [0.2, 0.70]], dtype=torch.double)

    q, info = build_quality_fn(
        "cr_ndig", models, bounds, tau, context_dims=context_dims, X_obs=X_obs,
    )
    assert info["quality"] == "cr_ndig"
    assert info["n_prev_contexts"] == 2

    # Same design coordinate (x=0.5) for both candidates; only the context
    # coordinate differs: one lands exactly on a previously evaluated context
    # (0.30), the other is far from both prior contexts.
    z_close = torch.tensor([[0.5, 0.30]], dtype=torch.double)
    z_far = torch.tensor([[0.5, 0.99]], dtype=torch.double)

    q_close = q(z_close)
    q_far = q(z_far)
    assert float(q_close.item()) < float(q_far.item())


def test_cr_ndig_degenerate_paths_equal_plain_ndig():
    """With no context dims, or no prior observations, cr_ndig must reduce
    exactly to plain ndig_quality (the repulsion concept is inapplicable, not
    computed against a trivially-constant context)."""
    import torch
    from itcas.algorithms.quality import build_quality_fn, ndig_quality

    models, bounds, tau = _tiny_gp_setup(m=2, d=2)
    z = torch.rand(5, 2, dtype=torch.double)
    expected = ndig_quality(models, z, tau)

    X_obs = torch.tensor([[0.5, 0.30], [0.2, 0.70]], dtype=torch.double)

    # No context dims -> degenerate, exact ndig.
    q1, _ = build_quality_fn("cr_ndig", models, bounds, tau, context_dims=(), X_obs=X_obs)
    assert torch.allclose(q1(z), expected)

    # No prior observations -> degenerate, exact ndig.
    q2, _ = build_quality_fn("cr_ndig", models, bounds, tau, context_dims=(1,), X_obs=None)
    assert torch.allclose(q2(z), expected)

    q3, _ = build_quality_fn(
        "cr_ndig", models, bounds, tau, context_dims=(1,),
        X_obs=torch.empty(0, 2, dtype=torch.double),
    )
    assert torch.allclose(q3(z), expected)


def test_itcas_seq_is_continuous_and_uses_cfg_quality():
    """itcas_seq must route through the same continuous QD-DPP machinery as
    itcas (so --quality ndig etc. still applies) but never be treated as a
    batch method, regardless of cfg.batch_size."""
    import torch
    from itcas.pipeline.loop import _is_continuous_method, _select_continuous, ExperimentConfig
    from itcas.pipeline import PROBLEM_REGISTRY
    from itcas.utils.device import resolve_device
    from itcas.utils.gp import build_independent_gps

    assert _is_continuous_method("itcas_seq")

    problem = PROBLEM_REGISTRY["two_circles_2d"]()
    device = resolve_device("cpu")
    bounds = problem.bounds.to(device=device, dtype=torch.double)
    h = problem.thresholds.to(device=device, dtype=torch.double)
    X = problem.sample_uniform(6, seed=0, device=device, dtype=torch.double)
    Y = problem.evaluate(X).to(torch.double)
    models = build_independent_gps(X, Y, bounds=bounds)

    cfg = ExperimentConfig(method="itcas_seq", quality="ndig", batch_size=1)
    X_new, info = _select_continuous(
        models=models, bounds=bounds, h=h, batch_size=1, cfg=cfg,
        context_dims=problem.context_dims, seed=0,
    )
    assert X_new.shape[0] == 1
    assert info.get("quality") == "ndig"


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


def test_batch_vs_sequential_variant_pair_and_plots():
    import json
    import tempfile
    from pathlib import Path

    from itcas.reporting import batch_vs_sequential as bvs

    families_cfg = {
        "eci": ["cas_eci", "cas_eci_batch"],
        "moc_cas_hard": ["moc_cas_hard", "moc_cas_hard_batch"],
    }

    seq, batch = bvs.variant_pair(families_cfg, "eci")
    assert (seq, batch) == ("cas_eci", "cas_eci_batch")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        problem_dir = root / "sphere2_6d" / "p0_05"

        def write_run(method, seed, feasible_first_step):
            run_dir = problem_dir / method
            run_dir.mkdir(parents=True, exist_ok=True)
            name = f"sphere2_6d__{method}__p0_05_seed{seed}"
            (run_dir / f"{name}.jsonl").write_text(
                json.dumps({
                    "step": 1, "n_eval_total": 3, "n_eval_this_iter": 1,
                    "feasible": [feasible_first_step],
                    "x": [[0.1] * 6],
                    # Off-diagonal (non-collinear with the two init points below) so a
                    # feasible step actually grows the feasible convex hull volume from
                    # zero -- otherwise every metric curve here would trivially stay at
                    # its initial value regardless of feasibility, making the AUC-based
                    # Wilcoxon test below vacuous (all-zero diffs, no signal to detect).
                    "y": [[-1.0, -1.5] if feasible_first_step else [-100.0, -1.0]],
                }) + "\n"
            )
            (run_dir / f"{name}.summary.json").write_text(json.dumps({
                "config": {"method": method, "n_init": 2, "seed": seed,
                           "extra": {"threshold_pct": 0.05}},
                "problem": "sphere2_6d",
                "n_init": 2,
                "thresholds": [-30.0, -30.0],
                "context_dims": [3, 4, 5],
                "init_X": [[0.0] * 6, [0.1] * 6],
                "init_Y": [[-1.0, -1.0], [-2.0, -2.0]],
                "init_feasible": [True, True],
            }))

        # Batch variants of BOTH families always find a feasible point;
        # sequential variants never do -- a clean separation across every
        # seed so the one-sided Wilcoxon below has a real, consistent signal
        # to detect (6 paired seeds all favoring batch).
        for seed in range(6):
            write_run("cas_eci", seed, feasible_first_step=False)
            write_run("cas_eci_batch", seed, feasible_first_step=True)
            write_run("moc_cas_hard", seed, feasible_first_step=False)
            write_run("moc_cas_hard_batch", seed, feasible_first_step=True)

        out_dir = root / "_summary"

        # --- Plotting: eci + moc_cas_hard are combined into one "cas" group
        # (4 lines per panel); this problem/difficulty has no spacecraft data,
        # so exactly one standard-difficulty PDF should be produced.
        paths, runs_by_problem, caches_by_problem = bvs.summarize_group(
            "cas", ["eci", "moc_cas_hard"], root, ["sphere2_6d"], families_cfg,
            output_dir=out_dir,
        )
        assert paths
        pdfs = [p for p in paths if p.endswith(".pdf")]
        assert pdfs, paths
        for p in pdfs:
            assert Path(p).exists()
        assert len(pdfs) == 1, pdfs
        assert Path(pdfs[0]).name == "cas_p0_05_vs_evaluations.pdf", pdfs

        methods, styles = bvs._group_method_styles(["eci", "moc_cas_hard"], families_cfg)
        assert methods == ["cas_eci", "cas_eci_batch", "moc_cas_hard", "moc_cas_hard_batch"]
        # Same family shares one hue; sequential solid, batch dashed.
        assert styles["cas_eci"]["color"] == styles["cas_eci_batch"]["color"]
        assert styles["cas_eci"]["linestyle"] == "-"
        assert styles["cas_eci_batch"]["linestyle"] == "--"
        assert styles["moc_cas_hard"]["color"] == styles["moc_cas_hard_batch"]["color"]
        assert styles["moc_cas_hard"]["linestyle"] == "-"
        assert styles["moc_cas_hard_batch"]["linestyle"] == "--"
        assert styles["cas_eci"]["color"] != styles["moc_cas_hard"]["color"]

        # --- Statistics: one-sided Wilcoxon on area-under-product-curve,
        # per family, reusing the runs/caches already discovered above.
        family_reports = {}
        family_pairs = {}
        for fam in ("eci", "moc_cas_hard"):
            sequential, batch_m, auc_data = bvs._family_auc_data(
                fam, families_cfg, runs_by_problem, caches_by_problem,
            )
            report = bvs.build_family_stats_report(fam, sequential, batch_m, auc_data)
            assert report.groups
            g = next(
                g for g in report.groups
                if g.problem == "sphere2_6d" and g.difficulty == "p0_05"
            )
            assert g.pairwise, g.note
            assert g.pairwise[0].significant, g.pairwise[0]
            assert g.pairwise[0].effect_median_diff > 0  # batch AUC > sequential AUC
            assert g.friedman_significant is False  # Friedman gate intentionally skipped

            json_p, md_p = bvs.write_family_stats_report(report, fam, sequential, batch_m, out_dir)
            assert json_p.exists() and md_p.exists()
            md_text = md_p.read_text()
            assert "one-sided" in md_text.lower()
            assert "friedman" in md_text.lower()

            family_reports[fam] = report
            family_pairs[fam] = (sequential, batch_m)

        # --- SUMMARY.md synthesis
        summary_path = bvs.write_overall_summary(
            family_reports, family_pairs, ["sphere2_6d"], out_dir,
        )
        assert summary_path.exists()
        summary_text = summary_path.read_text()
        assert "Overall conclusion" in summary_text
        assert "sphere2_6d" in summary_text
        assert "eci" in summary_text and "moc_cas_hard" in summary_text
        assert "batch significantly outperforms sequential" in summary_text


def test_forward_fill_at_carries_last_known_value():
    from itcas.reporting.summary import _forward_fill_at

    xs = [0.0, 5.0, 10.0]
    ys = [1.0, 2.0, 3.0]
    assert _forward_fill_at(xs, ys, -1.0) != _forward_fill_at(xs, ys, -1.0)  # NaN, before first point
    assert _forward_fill_at(xs, ys, 0.0) == 1.0
    assert _forward_fill_at(xs, ys, 3.0) == 1.0  # forward-filled from x=0
    assert _forward_fill_at(xs, ys, 5.0) == 2.0
    assert _forward_fill_at(xs, ys, 9.9) == 2.0
    assert _forward_fill_at(xs, ys, 10.0) == 3.0
    assert _forward_fill_at(xs, ys, 100.0) == 3.0  # carries past its own last point
    assert _forward_fill_at([], ys, 1.0) != _forward_fill_at([], ys, 1.0)  # NaN, empty xs


def test_rank_curves_on_union_grid_handles_batch_vs_sequential_length_mismatch():
    """Reproduces the exact bug scenario: a dense sequential method's array
    (many points, one per evaluation) vs a sparse batch method's array (few
    points, one per algorithmic step) spanning the SAME total-evals range.

    Before the fix, positionally truncating every method to the shortest
    array length collapsed the shared x-axis down to the batch method's
    array length while mislabeling it with the dense method's *early*
    x-values -- e.g. a batch method that legitimately covers evals 0..10
    would get plotted as if it only covered evals 0..2. The fix instead
    builds the union of every method's own x-values as the grid and
    forward-fills each method's product curve onto it.
    """
    from itcas.reporting.summary import _rank_curves_on_union_grid

    # Sequential method: one point per evaluation, 0..10 (11 points).
    seq_x = list(range(11))
    seq_med = [float(v) for v in seq_x]  # e.g. product grows 0..10

    # Batch method (q=5-ish): only 3 checkpoints, but spanning the SAME
    # 0..10 range as the sequential method -- far fewer points, not a
    # shorter range.
    batch_x = [0.0, 5.0, 10.0]
    batch_med = [1.0, 20.0, 3.0]  # batch briefly leads mid-range, then trails

    method_x = {"batch": batch_x, "seq": seq_x}
    method_med = {"batch": batch_med, "seq": seq_med}

    grid, ranks = _rank_curves_on_union_grid(method_x, method_med)

    # The grid must cover the FULL range every method actually reaches, not
    # be truncated to the shorter array's length (old bug: len 3).
    assert len(grid) == 11, grid
    assert grid[0] == 0.0 and grid[-1] == 10.0

    # At x=0: batch=1.0 vs seq=0.0 -> batch ranks 1st (larger is better).
    idx0 = grid.index(0.0)
    assert ranks["batch"][idx0] == 1
    assert ranks["seq"][idx0] == 2

    # At x=3 (no batch checkpoint there): batch forward-fills its x=0 value
    # (1.0) while seq has its own real value (3.0) -> seq now ranks 1st.
    idx3 = grid.index(3.0)
    assert ranks["seq"][idx3] == 1
    assert ranks["batch"][idx3] == 2

    # At x=7 (forward-filled from batch's x=5 checkpoint, value 20.0): batch
    # is way ahead of seq's real value of 7.0 -> batch ranks 1st again.
    idx7 = grid.index(7.0)
    assert ranks["batch"][idx7] == 1
    assert ranks["seq"][idx7] == 2

    # At x=10 both have real checkpoints: seq=10.0 > batch=3.0.
    idx10 = grid.index(10.0)
    assert ranks["seq"][idx10] == 1
    assert ranks["batch"][idx10] == 2


def test_per_method_product_curves_keeps_each_methods_own_x_and_length():
    """Batch vs sequential runs of very different array lengths, same range.

    ``_per_method_product_curves`` must return each method's own x-values at
    their own full length -- never truncated to the shortest method's array
    length (that was the raw-product-column half of the bug).
    """
    import torch

    from itcas.reporting.metrics import REGISTRY as METRIC_REGISTRY, RunSeries
    from itcas.reporting.summary import _per_method_product_curves

    def make_run(run_name, method, x_evals):
        n = len(x_evals)
        return RunSeries(
            problem="p",
            method=method,
            run_name=run_name,
            seed=0,
            thresholds=torch.tensor([]),
            context_dims=(),
            config={},
            x_evals=list(x_evals),
            x_steps=list(range(n)),
            feasible_per_step=[[] for _ in range(n)],
            X_per_step=[torch.zeros(0, 1) for _ in range(n)],
            Y_per_step=[torch.zeros(0, 1) for _ in range(n)],
        )

    # Sequential: 11 points, one per evaluation, 0..10.
    seq_run = make_run("seq_run", "seq_method", list(range(11)))
    # Batch: 3 points (one per step), but spanning the SAME 0..10 range.
    batch_run = make_run("batch_run", "batch_method", [0, 5, 10])

    metrics_present = [METRIC_REGISTRY["cumulative_positives"]]
    cache = {
        "seq_run": {"cumulative_positives": [float(v) for v in range(11)]},
        "batch_run": {"cumulative_positives": [1.0, 2.0, 3.0]},
    }
    method_runs = {"seq_method": [seq_run], "batch_method": [batch_run]}

    method_x, method_med, method_lo, method_hi = _per_method_product_curves(
        method_runs, ["seq_method", "batch_method"], "evals", cache, metrics_present,
    )

    assert len(method_x["seq_method"]) == 11
    assert method_x["seq_method"][-1] == 10
    assert len(method_x["batch_method"]) == 3
    assert method_x["batch_method"] == [0, 5, 10]
    # The batch method's own full range/length is preserved -- not collapsed
    # down to match the sequential method's positional length.
    assert len(method_med["batch_method"]) == 3


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


def _tiny_gp_setup(n_train=12, d=2, m=2, seed=0):
    import torch
    from itcas.utils.gp import build_independent_gps

    torch.manual_seed(seed)
    X = torch.rand(n_train, d, dtype=torch.double)
    Y = torch.rand(n_train, m, dtype=torch.double)
    bounds = torch.stack([torch.zeros(d), torch.ones(d)]).double()
    models = build_independent_gps(X, Y, bounds=bounds)
    tau = torch.tensor([0.4] * m, dtype=torch.double)
    return models, bounds, tau


def test_c2lse_quality_matches_closed_form_nonneg_differentiable():
    """C2LSE: q(z, active_obj=k) = sigma_k(z) / max(eps, |mu_k(z)-tau_k|).

    Round-robin objective alternation: check both active_obj=0 and
    active_obj=1 against their single-objective closed forms, and confirm
    the two differ (alternation actually changes which GP drives the score).
    """
    import torch

    from itcas.algorithms.quality import c2lse_quality, QUALITY_REGISTRY, available_qualities

    models, bounds, tau = _tiny_gp_setup(m=2)
    z = torch.rand(6, 2, dtype=torch.double, requires_grad=True)
    eps = 1e-2

    def closed_form(k):
        gp, tau_i = models[k], tau[k]
        post = gp.posterior(z)
        mu_i = post.mean.squeeze(-1)
        sigma_i = post.variance.clamp_min(1e-12).sqrt().squeeze(-1)
        return sigma_i / (mu_i - tau_i).abs().clamp_min(eps)

    q0 = c2lse_quality(models, z, tau, active_obj=0)
    q1 = c2lse_quality(models, z, tau, active_obj=1)
    assert torch.allclose(q0, closed_form(0))
    assert torch.allclose(q1, closed_form(1))
    assert not torch.allclose(q0, q1), "active_obj=0 and active_obj=1 should differ"

    assert float(q0.min().detach()) >= 0.0
    assert float(q1.min().detach()) >= 0.0

    g = torch.autograd.grad(q0.sum(), z)[0]
    assert g.shape == z.shape and torch.isfinite(g).all()

    assert "c2lse" in available_qualities()
    assert "c2lse" in QUALITY_REGISTRY


def test_bes_quality_nonneg_and_differentiable():
    """BES: binary entropy search quality is finite, and differentiable w.r.t. z.

    (No simple closed form to check against; verify shape/finiteness/autograd
    and registry membership, matching the style of the other quality tests.)
    Also confirm round-robin alternation: active_obj=0 vs active_obj=1 give
    different scores (proves it's not silently still aggregating over m).
    """
    import torch

    from itcas.algorithms.quality import bes_quality, QUALITY_REGISTRY, available_qualities

    models, bounds, tau = _tiny_gp_setup(m=2)
    z = torch.rand(6, 2, dtype=torch.double, requires_grad=True)
    q0 = bes_quality(models, z, tau, active_obj=0)
    q1 = bes_quality(models, z, tau, active_obj=1)

    assert q0.shape == (6,)
    assert torch.isfinite(q0).all()
    assert torch.isfinite(q1).all()
    assert not torch.allclose(q0, q1), "active_obj=0 and active_obj=1 should differ"

    g = torch.autograd.grad(q0.sum(), z)[0]
    assert g.shape == z.shape and torch.isfinite(g).all()

    assert "bes" in available_qualities()
    assert "bes" in QUALITY_REGISTRY


def test_straddle_score_depends_only_on_active_objective():
    """STRADDLE's score at pipeline iteration t uses only objective t % m.

    Build a 2-objective GP setup and confirm straddle(t=0) vs straddle(t=1)
    produce different per-candidate scores, each matching the single-objective
    closed form alpha_i = beta*sigma_i - |mu_i - h_i| for the corresponding
    active objective.
    """
    import torch

    from itcas.baselines.baselines import straddle
    from itcas.utils.gp import posterior_mean_std

    models, bounds, tau = _tiny_gp_setup(m=2)
    cand = torch.rand(10, 2, dtype=torch.double)
    beta = 1.96

    mu, sigma = posterior_mean_std(models, cand)
    expected0 = beta * sigma[:, 0] - (mu[:, 0] - tau[0]).abs()
    expected1 = beta * sigma[:, 1] - (mu[:, 1] - tau[1]).abs()

    idx0, info0 = straddle(models, cand, tau, batch_size=cand.shape[0], beta=beta, t=0)
    idx1, info1 = straddle(models, cand, tau, batch_size=cand.shape[0], beta=beta, t=1)

    assert torch.allclose(info0["score"], expected0)
    assert torch.allclose(info1["score"], expected1)
    assert not torch.allclose(info0["score"], info1["score"])
    assert info0["active_obj"] == 0
    assert info1["active_obj"] == 1

    # t=2 wraps back around to objective 0 (m=2).
    _, info2 = straddle(models, cand, tau, batch_size=cand.shape[0], beta=beta, t=2)
    assert info2["active_obj"] == 0
    assert torch.allclose(info2["score"], expected0)


def test_build_quality_fn_t_propagation_c2lse():
    """t correctly propagates end-to-end through build_quality_fn for c2lse:
    t=0 and t=1 (m=2) evaluate differently on the same z."""
    import torch

    from itcas.algorithms.quality import build_quality_fn

    models, bounds, tau = _tiny_gp_setup(m=2)
    z = torch.rand(6, 2, dtype=torch.double)

    q_fn0, info0 = build_quality_fn("c2lse", models, bounds, tau, t=0)
    q_fn1, info1 = build_quality_fn("c2lse", models, bounds, tau, t=1)

    assert info0["active_obj"] == 0
    assert info1["active_obj"] == 1

    q0 = q_fn0(z)
    q1 = q_fn1(z)
    assert not torch.allclose(q0, q1)

    # t=2 wraps back to active_obj=0 and matches the t=0 evaluation.
    q_fn2, info2 = build_quality_fn("c2lse", models, bounds, tau, t=2)
    assert info2["active_obj"] == 0
    assert torch.allclose(q_fn2(z), q0)


def test_interior_sampling_quality_registered_and_differentiable():
    """Family-C Stage 2 quality: registered, finite, differentiable w.r.t. z,
    and matches the closed-form sigma_combined - lambda*relu(0.95 - PoF)."""
    import torch

    from itcas.algorithms.quality import (
        interior_sampling_quality, sigma_combined, QUALITY_REGISTRY,
        available_qualities, _INTERIOR_LAMBDA_PENALTY_DEFAULT, _INTERIOR_POF_TARGET,
    )
    from itcas.algorithms.roi_mi import feasibility_probabilities, joint_feasibility_probability

    models, bounds, tau = _tiny_gp_setup()
    z = torch.rand(8, 2, dtype=torch.double, requires_grad=True)
    q = interior_sampling_quality(models, z, tau)

    assert q.shape == (8,)
    assert torch.isfinite(q).all()

    # Differentiable w.r.t. z.
    g = torch.autograd.grad(q.sum(), z)[0]
    assert g.shape == z.shape and torch.isfinite(g).all()

    # Closed-form check: q(z) = sigma_combined(z) - lambda*relu(pof_target - PoF(z)).
    with torch.no_grad():
        mus, sigmas = [], []
        for gp in models:
            post = gp.posterior(z)
            mus.append(post.mean.squeeze(-1))
            sigmas.append(post.variance.clamp_min(1e-12).sqrt().squeeze(-1))
        mu = torch.stack(mus, dim=-1)
        sigma = torch.stack(sigmas, dim=-1)
        sc = sigma_combined(models, z)
        assert torch.allclose(sc, sigma.pow(2).sum(dim=-1).sqrt())
        pof = joint_feasibility_probability(feasibility_probabilities(mu, sigma, tau))
        raw = sc - _INTERIOR_LAMBDA_PENALTY_DEFAULT * (_INTERIOR_POF_TARGET - pof).clamp_min(0.0)
        expected = torch.nn.functional.softplus(raw)
        assert torch.allclose(q, expected)
        # softplus keeps quality strictly positive (and hence gradient-bearing)
        # even deep outside the feasible interior -- this is the fix for the
        # vanishing-gradient bug in the QD-DPP marginal-gain machinery's
        # clamp_min(0) (see interior_sampling_quality docstring).
        assert float(q.min()) > 0.0

    assert "interior_sampling" in available_qualities()
    assert "interior_sampling" in QUALITY_REGISTRY

    # sigma_combined itself: non-negative and reduces to a single sigma for m=1.
    assert float(sc.min()) >= 0.0
    from itcas.utils.gp import build_independent_gps
    X1 = torch.rand(10, 2, dtype=torch.double)
    Y1 = torch.rand(10, 1, dtype=torch.double)
    models1 = build_independent_gps(X1, Y1, bounds=bounds)
    sc1 = sigma_combined(models1, z.detach())
    post1 = models1[0].posterior(z.detach())
    sigma1 = post1.variance.clamp_min(1e-12).sqrt().squeeze(-1)
    assert torch.allclose(sc1, sigma1)


def test_interior_sampling_ascent_satisfies_pof_constraint():
    """Regression: the *best-scoring* multistart-ascent restart(s) on
    interior_sampling_quality must satisfy the soft PoF>=0.95 constraint
    whenever the feasible interior is non-empty. This is the property that
    actually matters operationally -- `select_batch_continuous` always picks
    the highest-quality / highest-marginal-gain candidate(s), not a random
    restart, so not every one of the 16 random restarts needs to converge
    into the feasible interior in a fixed step budget, only the winner(s).

    (Earlier draft of this test asserted >=90% of *all* restarts satisfy the
    gate; that is too strong -- with only 60 Adam steps, restarts that start
    deep in infeasible territory can fail to fully climb back before the
    ascent budget runs out, even though the softplus reparameterization
    keeps their gradient nonzero throughout. The best-of-16 restart is what
    the pipeline actually selects, and it reliably clears the gate; see also
    the DPP-driven `select_batch_continuous` empirical check in the Phase-2
    final report.)
    """
    import torch

    from itcas.algorithms.continuous import multistart_ascent
    from itcas.algorithms.quality import interior_sampling_quality
    from itcas.algorithms.roi_mi import feasibility_probabilities, joint_feasibility_probability
    from itcas.pipeline.problems import two_circles_2d
    from itcas.utils.gp import build_independent_gps, posterior_mean_std

    torch.manual_seed(0)
    problem = two_circles_2d()
    bounds = problem.bounds.to(torch.double)
    X = problem.sample_uniform(30, seed=0, dtype=torch.double)
    Y = problem.evaluate(X).to(torch.double)
    h = problem.thresholds.to(torch.double)
    models = build_independent_gps(X, Y, bounds=bounds)

    def obj(z):
        return interior_sampling_quality(models, z, h)

    Z, vals = multistart_ascent(obj, bounds, n_restarts=16, n_steps=60, lr=0.05, seed=1)
    mu, sigma = posterior_mean_std(models, Z)
    pof = joint_feasibility_probability(feasibility_probabilities(mu, sigma, h))
    best = int(torch.argmax(vals))
    assert float(pof[best]) >= 0.95, (
        f"expected the best-scoring restart to satisfy PoF>=0.95, got {float(pof[best])}"
    )


def test_stage_for_switches_at_half_budget():
    from itcas.pipeline.loop import _stage_for

    # Non-two-stage methods always stay in "search".
    for m in ["itcas", "random", "straddle", "c2lse", "c2lse_batch"]:
        assert _stage_for(m, 0, 10) == "search"
        assert _stage_for(m, 9, 10) == "search"

    # Two-stage methods (and their _batch siblings) switch at t >= 0.5*n_iters
    # under the explicit _lse50 (default) infix.
    for m in [
        "straddle_then_sample_lse50", "straddle_then_sample_lse50_batch",
        "c2lse_then_sample_lse50", "c2lse_then_sample_lse50_batch",
        "bes_then_sample_lse50", "bes_then_sample_lse50_batch",
    ]:
        assert _stage_for(m, 4, 10) == "search", m
        assert _stage_for(m, 5, 10) == "interior", m
        assert _stage_for(m, 0, 10) == "search", m
        assert _stage_for(m, 9, 10) == "interior", m


def test_parse_two_stage_lse_proportion_variants():
    """`_lseNN` infix parsing: the infix is mandatory (no bare spelling
    anymore), `_lse50` is the explicit default, `_lseNN` and its `_batch`
    sibling override the Stage-1 fraction, and bare/malformed/out-of-range
    suffixes all raise ValueError."""
    from itcas.pipeline.loop import _parse_two_stage, TwoStageSpec

    # Explicit _lse50 is the (formerly implicit) default 50% split. `.base`
    # is the resolved Stage-1 acquisition name (TWO_STAGE_BASE's *value*,
    # e.g. "straddle"), not the two-stage method name itself.
    assert _parse_two_stage("straddle_then_sample_lse50") == TwoStageSpec("straddle", 0.5)
    assert _parse_two_stage("c2lse_then_sample_lse50_batch") == TwoStageSpec("c2lse", 0.5)

    # Explicit _lseNN infix overrides the Stage-1 fraction; _batch sibling
    # ordering is <base>_lseNN_batch (infix before the batch suffix).
    assert _parse_two_stage("straddle_then_sample_lse10") == TwoStageSpec("straddle", 0.10)
    assert _parse_two_stage("straddle_then_sample_lse25") == TwoStageSpec("straddle", 0.25)
    assert _parse_two_stage("bes_then_sample_lse10_batch") == TwoStageSpec("bes", 0.10)
    assert _parse_two_stage("c2lse_then_sample_lse25_batch") == TwoStageSpec("c2lse", 0.25)

    # Non-Family-C methods (including bare itcas/random) are not two-stage.
    assert _parse_two_stage("itcas") is None
    assert _parse_two_stage("random") is None
    assert _parse_two_stage("straddle") is None

    # Bare two-stage names (missing the now-mandatory _lseNN infix) raise,
    # as do malformed / out-of-range suffixes.
    for bad in (
        "straddle_then_sample",          # bare -- no longer valid
        "c2lse_then_sample_batch",       # bare + _batch -- no longer valid
        "straddle_then_sample_lse0",
        "straddle_then_sample_lse100",
        "bes_then_sample_lse5x",
    ):
        try:
            _parse_two_stage(bad)
            raise AssertionError(f"expected ValueError for {bad}")
        except ValueError:
            pass


def test_stage_for_respects_lse_proportion_override():
    from itcas.pipeline.loop import _stage_for

    # Explicit _lse50: switches at the 50% mark (the formerly-implicit default).
    assert _stage_for("straddle_then_sample_lse50", 4, 10) == "search"
    assert _stage_for("straddle_then_sample_lse50", 5, 10) == "interior"

    # _lse10: switches at 10% of n_iters (t=0 is already >= 1 for n_iters=10,
    # so only t=0 stays "search").
    assert _stage_for("straddle_then_sample_lse10", 0, 10) == "search"
    assert _stage_for("straddle_then_sample_lse10", 1, 10) == "interior"
    assert _stage_for("straddle_then_sample_lse10_batch", 0, 10) == "search"
    assert _stage_for("straddle_then_sample_lse10_batch", 1, 10) == "interior"

    # _lse25: switches at 25% of n_iters.
    assert _stage_for("bes_then_sample_lse25", 2, 20) == "search"
    assert _stage_for("bes_then_sample_lse25", 5, 20) == "interior"
    assert _stage_for("c2lse_then_sample_lse25_batch", 4, 20) == "search"
    assert _stage_for("c2lse_then_sample_lse25_batch", 5, 20) == "interior"

    # Bare (missing infix) and malformed suffixes still raise through
    # _stage_for (not silently treated as a non-two-stage "search"-only method).
    for bad in ("straddle_then_sample", "straddle_then_sample_lse0", "bes_then_sample_lse150"):
        try:
            _stage_for(bad, 0, 10)
            raise AssertionError(f"expected ValueError for {bad}")
        except ValueError:
            pass


def test_two_stage_effective_batch_size_and_is_batch_method():
    from itcas.pipeline.loop import effective_batch_size

    seq_methods = [
        "straddle_then_sample_lse50", "c2lse_then_sample_lse50",
        "bes_then_sample_lse50",
    ]
    batch_methods = [m + "_batch" for m in seq_methods]

    for m in seq_methods:
        assert effective_batch_size(m, 4) == 1, m
        # is_batch_method mirrors run_experiment's own predicate.
        assert not (m == "itcas" or m.endswith("_batch")), m

    for m in batch_methods:
        assert effective_batch_size(m, 4) == 4, m
        assert effective_batch_size(m, 1) == 1, m
        assert (m == "itcas" or m.endswith("_batch")), m


def test_two_stage_baselines_registered_in_baseline_registry():
    """The discrete-pool interior-sampling helpers backing straddle_then_sample
    must be registered so `_select_baseline` can dispatch to them."""
    from itcas.baselines import REGISTRY

    assert "interior_sampling" in REGISTRY
    assert "interior_sampling_batch" in REGISTRY


def test_straddle_then_sample_stage_flips_in_jsonl_log():
    """Integration-style: run straddle_then_sample for a handful of iterations
    on two_circles_2d and confirm info.stage flips from 'search' to 'interior'
    partway through the logged JSONL."""
    import json
    import tempfile
    from pathlib import Path

    from itcas.pipeline.loop import ExperimentConfig, run_experiment
    from itcas.pipeline.problems import two_circles_2d

    with tempfile.TemporaryDirectory() as tmp:
        problem = two_circles_2d()
        cfg = ExperimentConfig(
            method="straddle_then_sample_lse50",
            budget=8,
            batch_size=1,
            n_init=4,
            n_candidates=32,
            seed=0,
            out_dir=tmp,
            run_name="stage_flip_test",
            device="cpu",
        )
        summary = run_experiment(problem, cfg)
        assert summary["n_iters"] == 8

        lines = Path(tmp, "stage_flip_test.jsonl").read_text().strip().split("\n")
        assert len(lines) == 8
        stages = [json.loads(line)["info"]["stage"] for line in lines]
        # First half (t=0..3, n_iters=8 -> 0.5*8=4) is "search", second half "interior".
        assert stages[:4] == ["search"] * 4, stages
        assert stages[4:] == ["interior"] * 4, stages


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


def test_synthetic_comparison_split_pipeline_matches_monolithic():
    """The per-problem+aggregate synthetic-comparison split must not change any numbers.

    Builds tiny fake run data for two "synthetic" problems (every method in
    ``SYNTHETIC_METHODS``, a handful of seeds, one difficulty level) and
    checks that:

    1. :func:`summarize_synthetic_comparison_problem` (called once per
       problem) + :func:`summarize_synthetic_comparison_aggregate` produce a
       byte-identical ``synthetic_comparison_stats_report.json`` to the
       monolithic :func:`summarize_synthetic_comparison`.
    2. The combined avg-rank / relative-AUC dicts (:func:`_combine_synthetic_summaries`)
       are numerically identical whether fed in-memory per-problem summaries
       (the monolithic path) or summaries reloaded from the intermediate
       JSON files the split per-problem jobs write to disk.
    """
    import json
    import tempfile
    from pathlib import Path

    from itcas.reporting import summary as smry

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        problems = ["fake_problem_a", "fake_problem_b"]
        diff = "p0_05"

        def write_run(problem, method, seed, bias):
            run_dir = root / problem / diff / method
            run_dir.mkdir(parents=True, exist_ok=True)
            name = f"{problem}__{method}__{diff}_seed{seed}"
            # A little per-(method, seed) variation so ranks/ratios aren't
            # all tied -- exact values don't matter, only that both
            # pipelines compute the *same* values from the same inputs.
            feasible = (seed + bias) % 3 != 0
            y0 = -1.0 - 0.1 * bias - 0.01 * seed
            (run_dir / f"{name}.jsonl").write_text(
                json.dumps({
                    "step": 1, "n_eval_total": 3, "n_eval_this_iter": 1,
                    "feasible": [feasible],
                    "x": [[0.1] * 6],
                    "y": [[y0, y0 - 0.5] if feasible else [-100.0, -1.0]],
                }) + "\n"
            )
            (run_dir / f"{name}.summary.json").write_text(json.dumps({
                "config": {"method": method, "n_init": 2, "seed": seed,
                           "extra": {"threshold_pct": 0.05}},
                "problem": problem,
                "n_init": 2,
                "thresholds": [-30.0, -30.0],
                "context_dims": [3, 4, 5],
                "init_X": [[0.0] * 6, [0.1] * 6],
                "init_Y": [[-1.0, -1.0], [-2.0, -2.0]],
                "init_feasible": [True, True],
            }))

        for problem in problems:
            for m_idx, method in enumerate(smry.SYNTHETIC_METHODS):
                for seed in range(4):
                    write_run(problem, method, seed, bias=m_idx)

        problems_config = root / "problems_config.json"
        problems_config.write_text(json.dumps({"problems": problems, "temporary": []}))

        mono_out = root / "mono_out"
        split_out = root / "split_out"
        metrics_dir = root / "split_metrics"

        mono_paths = smry.summarize_synthetic_comparison(
            root, problems_config=problems_config, output_dir=mono_out, alpha=0.05,
        )
        assert mono_paths

        split_paths = []
        for problem in problems:
            split_paths.extend(smry.summarize_synthetic_comparison_problem(
                root, problem, problems_config=problems_config,
                output_dir=split_out, save_metrics_dir=metrics_dir, alpha=0.05,
            ))
        split_paths.extend(smry.summarize_synthetic_comparison_aggregate(
            metrics_dir, output_dir=split_out, alpha=0.05,
        ))
        assert split_paths

        mono_stats = (mono_out / diff / "synthetic_comparison_stats_report.json").read_text()
        split_stats = (split_out / diff / "synthetic_comparison_stats_report.json").read_text()
        assert json.loads(mono_stats) == json.loads(split_stats)

        # Per-problem PDF must exist under both pipelines with the same name.
        for problem in problems:
            assert (mono_out / diff / f"synthetic_comparison_{problem}_vs_evaluations.pdf").exists()
            assert (split_out / diff / f"synthetic_comparison_{problem}_vs_evaluations.pdf").exists()
        assert (mono_out / diff / "synthetic_comparison_avg_rank_vs_evaluations.pdf").exists()
        assert (split_out / diff / "synthetic_comparison_avg_rank_vs_evaluations.pdf").exists()
        assert (mono_out / diff / "synthetic_comparison_relative_auc_vs_evaluations.pdf").exists()
        assert (split_out / diff / "synthetic_comparison_relative_auc_vs_evaluations.pdf").exists()

        # Recompute the combined avg-rank/relative-AUC dicts two ways and
        # confirm they agree exactly: (a) in-memory per-problem summaries
        # (what the monolithic function uses internally) vs (b) summaries
        # reloaded from the on-disk JSON the split per-problem jobs wrote.
        from itcas.reporting.batch_vs_sequential import _collect_family_runs

        runs_by_problem = _collect_family_runs(root, problems, list(smry.SYNTHETIC_METHODS))
        caches_by_problem = {p: smry._precompute(rs) for p, rs in runs_by_problem.items() if rs}
        summaries_in_memory = {}
        for problem in problems:
            rs = runs_by_problem.get(problem, [])
            if not rs:
                continue
            summary, _ = smry._synthetic_problem_report(
                problem, rs, caches_by_problem.get(problem, {}), out_dir=None,
            )
            summaries_in_memory[problem] = summary
        combined_in_memory = smry._combine_synthetic_summaries(summaries_in_memory)

        summaries_from_disk = {}
        for f in sorted(metrics_dir.glob("*_synthetic_metrics.json")):
            problem, diffs = smry.load_synthetic_problem_metrics(f)
            summaries_from_disk[problem] = diffs
        combined_from_disk = smry._combine_synthetic_summaries(summaries_from_disk)

        assert set(combined_in_memory) == set(combined_from_disk)
        for d in combined_in_memory:
            a, b = combined_in_memory[d], combined_from_disk[d]
            assert a["n_rows"] == b["n_rows"]
            assert [s.key for s in a["metrics_present"]] == [s.key for s in b["metrics_present"]]
            assert a["avg_rank"] == b["avg_rank"]
            assert a["relative_auc"] == b["relative_auc"]
            assert a["auc_by_problem"] == b["auc_by_problem"]


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
    test_objective_diversity_metrics()
    test_epsilon_archive_size_uses_transformed_not_raw_distance()
    test_tune_eps_archive_percentile_hand_checked()
    test_tune_eps_archive_caps_pdist_via_subsampling()
    test_tune_eps_archive_pools_log_transforms_and_filters_infeasible()
    test_tune_eps_archive_load_run_final_dedupes_steps_no_per_step_history()
    test_tune_eps_archive_difficulty_dir_to_config_key()
    test_binary_entropy_bounds()
    test_synthetic_benchmarks_shapes_and_feasibility()
    test_dtlz2_pareto_front_identity()
    test_sphere2_optima_locations()
    test_only_itcas_runs_in_batch()
    test_visualization_writes_per_metric_pdfs()
    test_forward_fill_at_carries_last_known_value()
    test_rank_curves_on_union_grid_handles_batch_vs_sequential_length_mismatch()
    test_per_method_product_curves_keeps_each_methods_own_x_and_length()
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
    test_synthetic_comparison_split_pipeline_matches_monolithic()
    test_run_lock_prevents_concurrent_runs()
    test_evaluate_true_matches_input_device_and_dtype()
    test_parse_seed_spec_ranges_and_singletons()
    test_run_name_for_seed_template_and_suffix()
    test_pending_seeds_skips_completed()
    test_c2lse_quality_matches_closed_form_nonneg_differentiable()
    test_bes_quality_nonneg_and_differentiable()
    test_straddle_score_depends_only_on_active_objective()
    test_build_quality_fn_t_propagation_c2lse()
    test_interior_sampling_quality_registered_and_differentiable()
    test_interior_sampling_ascent_satisfies_pof_constraint()
    test_stage_for_switches_at_half_budget()
    test_parse_two_stage_lse_proportion_variants()
    test_stage_for_respects_lse_proportion_override()
    test_two_stage_effective_batch_size_and_is_batch_method()
    test_two_stage_baselines_registered_in_baseline_registry()
    test_straddle_then_sample_stage_flips_in_jsonl_log()
    print("OK: smoke unit tests passed.")
