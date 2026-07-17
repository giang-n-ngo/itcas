"""Synthetic test problems for smoke / unit testing.

Each problem exposes:
    - bounds: (2, d) tensor
    - thresholds: (m,) tensor
    - evaluate(X) -> (N, m) tensor of noiseless objective values
    - feasible_grid(n) -> (K, d) discretization of the satisfactory region

These are intentionally small to keep smoke tests fast.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch


@dataclass
class Problem:
    name: str
    bounds: torch.Tensor   # (2, d)
    thresholds: torch.Tensor  # (m,)
    fn: Callable[[torch.Tensor], torch.Tensor]
    noise_std: float = 0.0
    context_dims: tuple[int, ...] = ()  # indices of context columns c in z=(x,c)
    eps_archive: float = 0.05           # ε-Archive Size distance threshold (normalised obj space)

    @property
    def d(self) -> int:
        return int(self.bounds.shape[-1])

    @property
    def m(self) -> int:
        return int(self.thresholds.numel())

    @property
    def design_dims(self) -> tuple[int, ...]:
        return tuple(i for i in range(self.d) if i not in self.context_dims)

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        """Noiseless objective values, evaluated device-safely.

        The synthetic objective functions build their constants on CPU, so we
        evaluate ``fn`` on CPU and move the result back to ``X``'s device. This
        keeps the (cheap) analytic objective free of device bookkeeping while
        letting the (expensive) GP modeling run on a GPU.
        """
        Y = self.fn(X.detach().to("cpu"))
        return Y.to(device=X.device, dtype=X.dtype)

    def reference_objectives(
        self,
        n: int,
        seed: int,
        thresholds: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Return pre-computed feasible reference objectives, or None to use MC sampling.

        Override this in problems where ``evaluate_true`` is expensive (e.g. runs
        external simulations) so that ``build_reference_data`` can use historical
        data instead of triggering new evaluations.
        """
        return None

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        Y = self.evaluate_true(X)
        if self.noise_std > 0:
            Y = Y + self.noise_std * torch.randn_like(Y)
        return Y

    def sample_uniform(
        self,
        n: int,
        seed: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        # Draw on CPU for cross-device reproducibility, then move to ``device``.
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        u = torch.rand(n, self.d, generator=g)
        lo, hi = self.bounds[0], self.bounds[1]
        X = lo + (hi - lo) * u
        if dtype is not None:
            X = X.to(dtype=dtype)
        if device is not None:
            X = X.to(device=device)
        return X


def two_circles_2d(noise_std: float = 0.05) -> Problem:
    """Two-objective problem on [0,1]^2; feasible region is the intersection
    of two disks. Used for fast smoke tests.
    """
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    thresholds = torch.tensor([0.0, 0.0])

    def fn(X: torch.Tensor) -> torch.Tensor:
        c1 = torch.tensor([0.4, 0.5])
        c2 = torch.tensor([0.6, 0.5])
        f1 = 0.25 - ((X - c1) ** 2).sum(dim=-1)
        f2 = 0.25 - ((X - c2) ** 2).sum(dim=-1)
        return torch.stack([f1, f2], dim=-1)

    return Problem("two_circles_2d", bounds, thresholds, fn, noise_std)


def contextual_circles_3d(noise_std: float = 0.05) -> Problem:
    """Contextual two-objective problem on [0,1]^3.

    z = (x1, x2, c) with the last column the context c. The two feasible
    disks are centred at context-dependent locations, so the feasible
    region S(c) sweeps across the design space as c varies. This exercises
    the joint (objective + context) QD-DPP diversity kernel.
    """
    bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    thresholds = torch.tensor([0.0, 0.0])

    def fn(Z: torch.Tensor) -> torch.Tensor:
        x = Z[..., :2]
        c = Z[..., 2]
        cx = 0.3 + 0.4 * c
        c1 = torch.stack([cx - 0.1, torch.full_like(cx, 0.5)], dim=-1)
        c2 = torch.stack([cx + 0.1, torch.full_like(cx, 0.5)], dim=-1)
        f1 = 0.25 - ((x - c1) ** 2).sum(dim=-1)
        f2 = 0.25 - ((x - c2) ** 2).sum(dim=-1)
        return torch.stack([f1, f2], dim=-1)

    return Problem(
        "contextual_circles_3d", bounds, thresholds, fn, noise_std,
        context_dims=(2,),
    )


# ---------------------------------------------------------------------------
# C-MO-CAS synthetic benchmark suite (see contexts/synthetic_benchmarks.md)
#
# All objective functions are *negated* so that feasibility is a super-level
# set constraint f_i(z) >= tau_i (maximization), as required by the C-MO-CAS
# problem formulation. z = [x, w] with the design variables x first and the
# context variables w last; ``context_dims`` records the context columns.
#
# Default thresholds were calibrated by Monte-Carlo so that the joint feasible
# region S = {z : f_i(z) >= tau_i for all i} is non-empty and occupies roughly
# 1-10% of the (uniformly sampled) joint domain. They can be overridden via the
# ``thresholds`` argument for custom difficulty.
# ---------------------------------------------------------------------------


def _default_thresholds(
    thresholds: Optional[Sequence[float]], defaults: Sequence[float]
) -> torch.Tensor:
    vals = defaults if thresholds is None else thresholds
    return torch.as_tensor(list(vals), dtype=torch.double)


def sphere2_6d(
    noise_std: float = 0.0, thresholds: Optional[Sequence[float]] = None
) -> Problem:
    """Benchmark 1 - 2-Objective Shifted Sphere (Low-D).

    D=6 (d_x=3, d_w=3) on [-5, 5]^6. A convex baseline to verify basic
    convergence and hypervolume expansion. f1 centres at the origin; f2 is
    shifted uniformly by 2.0.
    """
    bounds = torch.tensor([[-5.0] * 6, [5.0] * 6])
    th = _default_thresholds(thresholds, (-30.0, -30.0))

    def fn(Z: torch.Tensor) -> torch.Tensor:
        f1 = -(Z ** 2).sum(dim=-1)
        f2 = -((Z - 2.0) ** 2).sum(dim=-1)
        return torch.stack([f1, f2], dim=-1)

    return Problem("sphere2_6d", bounds, th, fn, noise_std, context_dims=(3, 4, 5))


def rosenbrock_sphere_6d(
    noise_std: float = 0.0, thresholds: Optional[Sequence[float]] = None
) -> Problem:
    """Benchmark 2 - Contextual Rosenbrock vs. Shifted Sphere (Low-D).

    D=6 (d_x=4, d_w=2) on [-2, 2]^6. The context variable w1 bends the optimal
    manifold of the design variable x1 inside the Rosenbrock valley, paired
    against a shifted sphere to force a multi-objective trade-off.
    """
    bounds = torch.tensor([[-2.0] * 6, [2.0] * 6])
    th = _default_thresholds(thresholds, (-200.0, -40.0))

    def fn(Z: torch.Tensor) -> torch.Tensor:
        x = Z[..., :4]
        w = Z[..., 4:6]
        x1, w1, w2 = x[..., 0], w[..., 0], w[..., 1]
        ros_ctx = 100.0 * (x1 - w1 ** 2) ** 2 + (1.0 - w1) ** 2
        inner = (
            100.0 * (x[..., 1:] - x[..., :-1] ** 2) ** 2 + (1.0 - x[..., :-1]) ** 2
        ).sum(dim=-1)
        f1 = -(ros_ctx + inner + w2 ** 2)
        f2 = -((x + 2.0) ** 2).sum(dim=-1) - ((w + 2.0) ** 2).sum(dim=-1)
        return torch.stack([f1, f2], dim=-1)

    return Problem(
        "rosenbrock_sphere_6d", bounds, th, fn, noise_std, context_dims=(4, 5)
    )


def multimodal_trap_20d(
    noise_std: float = 0.0, thresholds: Optional[Sequence[float]] = None
) -> Problem:
    """Benchmark 3 - 3-Objective Multi-Modal Trap (High-D).

    D=20 (d_x=10, d_w=10) on [-5, 5]^20. A rugged stress test combining a
    negated Ackley, a shifted Rastrigin, and a shifted Sphere. Probes the
    acquisition's robustness to local optima and high-dimensional volume
    scaling.
    """
    bounds = torch.tensor([[-5.0] * 20, [5.0] * 20])
    th = _default_thresholds(thresholds, (-9.0, -400.0, -200.0))

    def fn(Z: torch.Tensor) -> torch.Tensor:
        two_pi = 2.0 * math.pi
        # f1: negated Ackley
        ackley = (
            -20.0 * torch.exp(-0.2 * torch.sqrt((Z ** 2).mean(dim=-1)))
            - torch.exp(torch.cos(two_pi * Z).mean(dim=-1))
            + 20.0
            + math.e
        )
        f1 = -ackley
        # f2: negated shifted Rastrigin (A=10, n=20 -> offset 200)
        shifted = Z - 2.0
        rastrigin = 200.0 + (
            shifted ** 2 - 10.0 * torch.cos(two_pi * shifted)
        ).sum(dim=-1)
        f2 = -rastrigin
        # f3: negated shifted Sphere
        f3 = -((Z + 2.0) ** 2).sum(dim=-1)
        return torch.stack([f1, f2, f3], dim=-1)

    return Problem(
        "multimodal_trap_20d", bounds, th, fn, noise_std,
        context_dims=tuple(range(10, 20)),
    )


def dtlz2_6d(
    noise_std: float = 0.0, thresholds: Optional[Sequence[float]] = None
) -> Problem:
    """Benchmark 4 - 4-Objective Contextual DTLZ2 (Low-D).

    D=6 (d_x=3, d_w=3) on [0, 1]^6. The design variables x are the DTLZ2
    position parameters tracing the Pareto front, while the context variables w
    control the distance term g(w), dictating the difficulty of feasibility.
    """
    bounds = torch.tensor([[0.0] * 6, [1.0] * 6])
    th = _default_thresholds(thresholds, (-0.8, -0.8, -0.8, -0.8))

    def fn(Z: torch.Tensor) -> torch.Tensor:
        x = Z[..., :3]
        w = Z[..., 3:6]
        g = ((w - 0.5) ** 2).sum(dim=-1)
        a = x * (math.pi / 2.0)
        c1, c2, c3 = torch.cos(a[..., 0]), torch.cos(a[..., 1]), torch.cos(a[..., 2])
        s1, s2, s3 = torch.sin(a[..., 0]), torch.sin(a[..., 1]), torch.sin(a[..., 2])
        scale = 1.0 + g
        f1 = -scale * c1 * c2 * c3
        f2 = -scale * c1 * c2 * s3
        f3 = -scale * c1 * s2
        f4 = -scale * s1
        return torch.stack([f1, f2, f3, f4], dim=-1)

    return Problem("dtlz2_6d", bounds, th, fn, noise_std, context_dims=(3, 4, 5))


def styblinski_tang_6d(
    noise_std: float = 0.0, thresholds: Optional[Sequence[float]] = None
) -> Problem:
    """Benchmark 5 - 2-Objective Shifted Styblinski-Tang (Low-D).

    D=6 (d_x=3, d_w=3) on [-5, 5]^6. A highly non-convex, multi-modal trap:
    unlike Ackley (one deep global funnel) the Styblinski-Tang function has
    multiple equally compelling local pockets. Shifting f2 by 2.0 ensures that
    moving toward one objective's global optimum drags the algorithm through
    deceptive local traps of the opposing objective.

    Equations (z = [x, w]):
        f1(z) = -0.5 * sum_i( z_i^4 - 16*z_i^2 + 5*z_i )
        f2(z) = -0.5 * sum_i( (z_i-2)^4 - 16*(z_i-2)^2 + 5*(z_i-2) )
    """
    bounds = torch.tensor([[-5.0] * 6, [5.0] * 6])
    # tau=50 -> ~7% uniform feasibility (MC N=5000)
    th = _default_thresholds(thresholds, (50.0, 50.0))

    def fn(Z: torch.Tensor) -> torch.Tensor:
        f1 = -0.5 * (Z ** 4 - 16.0 * Z ** 2 + 5.0 * Z).sum(dim=-1)
        S = Z - 2.0
        f2 = -0.5 * (S ** 4 - 16.0 * S ** 2 + 5.0 * S).sum(dim=-1)
        return torch.stack([f1, f2], dim=-1)

    return Problem(
        "styblinski_tang_6d", bounds, th, fn, noise_std, context_dims=(3, 4, 5)
    )


def dtlz7_6d(
    noise_std: float = 0.0, thresholds: Optional[Sequence[float]] = None
) -> Problem:
    """Benchmark 6 - 3-Objective Contextual DTLZ7 (Low-D).

    D=6 (d_x=2, d_w=4) on [0, 1]^6. DTLZ7 has a discontinuous, disconnected
    Pareto front — a brutal test for the diversity mechanism, which must
    simultaneously discover and maintain coverage over multiple isolated
    islands of feasibility.

    Equations:
        g(w) = 1 + 9/4 * sum_j w_j
        f1(x, w) = -x1
        f2(x, w) = -x2
        f3(x, w) = -(1+g(w)) * [3 - sum_{i=1}^{2} x_i/(1+g(w)) * (1 + sin(3*pi*x_i))]
    """
    bounds = torch.tensor([[0.0] * 6, [1.0] * 6])
    # tau3=-15 -> ~3.7% uniform feasibility (MC N=5000)
    th = _default_thresholds(thresholds, (-0.5, -0.5, -15.0))

    def fn(Z: torch.Tensor) -> torch.Tensor:
        x = Z[..., :2]
        w = Z[..., 2:6]
        g = 1.0 + (9.0 / 4.0) * w.sum(dim=-1)           # (...,)
        f1 = -x[..., 0]
        f2 = -x[..., 1]
        h_terms = (x / (1.0 + g.unsqueeze(-1))) * (
            1.0 + torch.sin(3.0 * math.pi * x)
        )
        h = 3.0 - h_terms.sum(dim=-1)
        f3 = -(1.0 + g) * h
        return torch.stack([f1, f2, f3], dim=-1)

    return Problem("dtlz7_6d", bounds, th, fn, noise_std, context_dims=(2, 3, 4, 5))


def ellipsoid_20d(
    noise_std: float = 0.0, thresholds: Optional[Sequence[float]] = None
) -> Problem:
    """Benchmark 7 - 2-Objective Ill-Conditioned Ellipsoid (High-D).

    D=20 (d_x=10, d_w=10) on [-5.12, 5.12]^20. An ARD diagnostic: weights
    grow exponentially (1000^((i-1)/19)) so that only dimensions near i=20
    strongly affect the objective. Tests whether the acquisition function can
    ignore flat environmental dimensions and correctly allocate its budget to
    the highly sensitive ones.

    Equations:
        f1(z) = -sum_i 1000^((i-1)/19) * z_i^2
        f2(z) = -sum_i 1000^((i-1)/19) * (z_i - 2)^2
    """
    bounds = torch.tensor([[-5.12] * 20, [5.12] * 20])
    # tau=-20000 -> ~6.4% uniform feasibility (MC N=5000)
    th = _default_thresholds(thresholds, (-20000.0, -20000.0))

    # Pre-build the weight vector [1000^(0/19), ..., 1000^(19/19)] on CPU.
    _w = torch.tensor(
        [1000.0 ** (i / 19.0) for i in range(20)], dtype=torch.double
    )  # (20,)

    def fn(Z: torch.Tensor) -> torch.Tensor:
        w = _w.to(dtype=Z.dtype)
        f1 = -(Z ** 2 * w).sum(dim=-1)
        f2 = -((Z - 2.0) ** 2 * w).sum(dim=-1)
        return torch.stack([f1, f2], dim=-1)

    return Problem(
        "ellipsoid_20d", bounds, th, fn, noise_std,
        context_dims=tuple(range(10, 20)),
    )


def dtlz1_12d(
    noise_std: float = 0.0, thresholds: Optional[Sequence[float]] = None
) -> Problem:
    """Benchmark 8 - 4-Objective Contextual DTLZ1 (High-D).

    D=12 (d_x=3, d_w=9) on [0, 1]^12. DTLZ1 uses a highly multi-modal
    context/distance function g(w) based on the Rastrigin topology, creating
    thousands of local Pareto fronts. Tests whether the acquisition avoids
    getting trapped on sub-optimal feasible regions in a heavily-scaled
    4-objective space.

    Equations:
        g(w) = 100 * [9 + sum_j ((w_j-0.5)^2 - cos(20*pi*(w_j-0.5)))]
        f1 = -0.5 * x1 * x2 * x3       * (1+g(w))
        f2 = -0.5 * x1 * x2 * (1-x3)  * (1+g(w))
        f3 = -0.5 * x1 * (1-x2)        * (1+g(w))
        f4 = -0.5 * (1-x1)             * (1+g(w))
    """
    bounds = torch.tensor([[0.0] * 12, [1.0] * 12])
    th = _default_thresholds(thresholds, (-0.25, -0.25, -0.25, -0.25))

    def fn(Z: torch.Tensor) -> torch.Tensor:
        x = Z[..., :3]
        w = Z[..., 3:12]
        dw = w - 0.5
        g = 100.0 * (9.0 + (dw ** 2 - torch.cos(20.0 * math.pi * dw)).sum(dim=-1))
        scale = 0.5 * (1.0 + g)
        x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
        f1 = -scale * x1 * x2 * x3
        f2 = -scale * x1 * x2 * (1.0 - x3)
        f3 = -scale * x1 * (1.0 - x2)
        f4 = -scale * (1.0 - x1)
        return torch.stack([f1, f2, f3, f4], dim=-1)

    return Problem(
        "dtlz1_12d", bounds, th, fn, noise_std,
        context_dims=tuple(range(3, 12)),
    )


def zdt3_6d(
    noise_std: float = 0.0, thresholds: Optional[Sequence[float]] = None
) -> Problem:
    """Benchmark 9 - 2-Objective Contextual ZDT3 (Low-D).

    D=6 (d_x=1, d_w=5) on [0, 1]^6. ZDT3 has a disconnected Pareto front —
    the feasible region is multiple disjoint islands in objective space. Tests
    the algorithm's ability to simultaneously discover and maintain coverage
    over isolated feasible manifolds.
    """
    bounds = torch.tensor([[0.0] * 6, [1.0] * 6])
    # tau_f1=-0.5 → x1≥0.5 (50% of [0,1]); tau_f2=-5.0 → feasible for g near 1
    th = _default_thresholds(thresholds, (-0.5, -5.0))

    def fn(Z: torch.Tensor) -> torch.Tensor:
        x1 = Z[..., 0]
        w = Z[..., 1:6]
        g = 1.0 + (9.0 / 5.0) * w.sum(dim=-1)
        f1 = -x1
        ratio = x1 / g
        f2 = -g * (1.0 - ratio.sqrt() - ratio * torch.sin(10.0 * math.pi * x1))
        return torch.stack([f1, f2], dim=-1)

    return Problem("zdt3_6d", bounds, th, fn, noise_std, context_dims=(1, 2, 3, 4, 5))


def levy_16d(
    noise_std: float = 0.0, thresholds: Optional[Sequence[float]] = None
) -> Problem:
    """Benchmark 10 - 3-Objective Shifted Levy (High-D).

    D=16 (d_x=8, d_w=8) on [-10, 10]^16. Three shifted variants of the
    highly rugged Levy function compete in 16 dimensions, testing the
    algorithm's ability to resolve competing multimodal landscapes.
    """
    bounds = torch.tensor([[-10.0] * 16, [10.0] * 16])
    # p90 of each objective individually; joint ~5% (correlated shifts)
    th = _default_thresholds(thresholds, (-120.0, -140.0, -120.0))

    def _levy(V: torch.Tensor) -> torch.Tensor:
        pi = math.pi
        t1 = torch.sin(pi * V[..., 0]) ** 2
        mid = ((V[..., :-1] - 1.0) ** 2 * (1.0 + 10.0 * torch.sin(pi * V[..., 1:]) ** 2)).sum(dim=-1)
        t3 = (V[..., -1] - 1.0) ** 2 * (1.0 + torch.sin(2.0 * pi * V[..., -1]) ** 2)
        return t1 + mid + t3

    def fn(Z: torch.Tensor) -> torch.Tensor:
        v = lambda S: 1.0 + (S - 1.0) / 4.0  # noqa: E731
        f1 = -_levy(v(Z))
        f2 = -_levy(v(Z - 2.0))
        f3 = -_levy(v(Z + 2.0))
        return torch.stack([f1, f2, f3], dim=-1)

    return Problem(
        "levy_16d", bounds, th, fn, noise_std, context_dims=tuple(range(8, 16))
    )


def dtlz3_8d(
    noise_std: float = 0.0, thresholds: Optional[Sequence[float]] = None
) -> Problem:
    """Benchmark 11 - 4-Objective Contextual DTLZ3 (Low-D).

    D=8 (d_x=3, d_w=5) on [0, 1]^8. A Rastrigin-based distance function
    g(w) creates thousands of local Pareto fronts, testing whether the
    algorithm avoids getting trapped on sub-optimal feasible regions.
    """
    bounds = torch.tensor([[0.0] * 8, [1.0] * 8])
    # p80 per objective; thousands of local Pareto fronts make joint feasibility tight
    th = _default_thresholds(thresholds, (-20.0, -20.0, -55.0, -150.0))

    def fn(Z: torch.Tensor) -> torch.Tensor:
        x = Z[..., :3]
        w = Z[..., 3:8]
        dw = w - 0.5
        g = 100.0 * (5.0 + (dw ** 2 - torch.cos(20.0 * math.pi * dw)).sum(dim=-1))
        a = x * (math.pi / 2.0)
        c1, c2, c3 = torch.cos(a[..., 0]), torch.cos(a[..., 1]), torch.cos(a[..., 2])
        s1, s2, s3 = torch.sin(a[..., 0]), torch.sin(a[..., 1]), torch.sin(a[..., 2])
        scale = 1.0 + g
        f1 = -scale * c1 * c2 * c3
        f2 = -scale * c1 * c2 * s3
        f3 = -scale * c1 * s2
        f4 = -scale * s1
        return torch.stack([f1, f2, f3, f4], dim=-1)

    return Problem("dtlz3_8d", bounds, th, fn, noise_std, context_dims=(3, 4, 5, 6, 7))


def vlmop2_6d(
    noise_std: float = 0.0, thresholds: Optional[Sequence[float]] = None
) -> Problem:
    """Benchmark 12 - 2-Objective Exponential VLMOP2 (Low-D).

    D=6 (d_x=3, d_w=3) on [-2, 2]^6. The topology is flat almost everywhere
    except near the optima, testing the GP's ability to handle vanishing
    gradients during exploration.
    """
    bounds = torch.tensor([[-2.0] * 6, [2.0] * 6])
    # p99 per objective; both objectives have opposing centers so joint is narrow
    th = _default_thresholds(thresholds, (-0.9, -0.9))
    _inv_sqrt6 = 1.0 / math.sqrt(6.0)

    def fn(Z: torch.Tensor) -> torch.Tensor:
        f1 = -(1.0 - torch.exp(-((Z - _inv_sqrt6) ** 2).sum(dim=-1)))
        f2 = -(1.0 - torch.exp(-((Z + _inv_sqrt6) ** 2).sum(dim=-1)))
        return torch.stack([f1, f2], dim=-1)

    return Problem("vlmop2_6d", bounds, th, fn, noise_std, context_dims=(3, 4, 5))


def dtlz4_12d(
    noise_std: float = 0.0, thresholds: Optional[Sequence[float]] = None
) -> Problem:
    """Benchmark 13 - 3-Objective Contextual DTLZ4 (High-D).

    D=12 (d_x=2, d_w=10) on [0, 1]^12. The alpha=100 power mapping creates
    a highly biased density, clustering the feasible manifold into a skewed
    geometric corner. Tests whether active search operates on geometry vs
    true uncertainty.
    """
    bounds = torch.tensor([[0.0] * 12, [1.0] * 12])
    th = _default_thresholds(thresholds, (-2.0, -2.0, -2.0))

    def fn(Z: torch.Tensor) -> torch.Tensor:
        x = Z[..., :2]
        w = Z[..., 2:12]
        g = ((w - 0.5) ** 2).sum(dim=-1)
        xa = x ** 100
        a = xa * (math.pi / 2.0)
        c1, c2 = torch.cos(a[..., 0]), torch.cos(a[..., 1])
        s1 = torch.sin(a[..., 0])
        scale = 1.0 + g
        f1 = -scale * c1 * c2
        f2 = -scale * c1 * torch.sin(a[..., 1])
        f3 = -scale * s1
        return torch.stack([f1, f2, f3], dim=-1)

    return Problem(
        "dtlz4_12d", bounds, th, fn, noise_std, context_dims=tuple(range(2, 12))
    )


def dixon_price_10d(
    noise_std: float = 0.0, thresholds: Optional[Sequence[float]] = None
) -> Problem:
    """Benchmark 14 - 2-Objective Contextual Dixon-Price vs. Sphere (Medium-D).

    D=10 (d_x=5, d_w=5) on [-10, 10]^10. The ill-conditioned Dixon-Price
    function forms a deep, narrow valley that must be tracked against a
    conflicting smooth sphere.
    """
    bounds = torch.tensor([[-10.0] * 10, [10.0] * 10])
    # p90 per objective (f1 very spread: narrow valley rarely sampled uniformly)
    th = _default_thresholds(thresholds, (-250000.0, -270.0))

    def fn(Z: torch.Tensor) -> torch.Tensor:
        i = torch.arange(2, Z.shape[-1] + 1, dtype=Z.dtype)
        f1 = -((Z[..., 0] - 1.0) ** 2 + (i * (2.0 * Z[..., 1:] ** 2 - Z[..., :-1]) ** 2).sum(dim=-1))
        f2 = -((Z - 2.0) ** 2).sum(dim=-1)
        return torch.stack([f1, f2], dim=-1)

    return Problem(
        "dixon_price_10d", bounds, th, fn, noise_std, context_dims=(5, 6, 7, 8, 9)
    )


def griewank_16d(
    noise_std: float = 0.0, thresholds: Optional[Sequence[float]] = None
) -> Problem:
    """Benchmark 15 - 4-Objective Shifted Griewank (High-D).

    D=16 (d_x=8, d_w=8) on [-10, 10]^16. Macro-convex but micro-rugged
    everywhere due to product-cosine interference. Four shifted variants
    create conflicting directional quadrants.
    """
    bounds = torch.tensor([[-10.0] * 16, [10.0] * 16])
    # ~20th-percentile of each objective; all four are highly concentrated
    th = _default_thresholds(thresholds, (-1.18, -1.18, -1.18, -1.18))

    _sqrt_i = torch.tensor([math.sqrt(i) for i in range(1, 17)], dtype=torch.double)

    def _griewank(V: torch.Tensor) -> torch.Tensor:
        si = _sqrt_i.to(dtype=V.dtype)
        return 1.0 + (V ** 2 / 4000.0).sum(dim=-1) - torch.cos(V / si).prod(dim=-1)

    def fn(Z: torch.Tensor) -> torch.Tensor:
        f1 = -_griewank(Z - 2.0)
        f2 = -_griewank(Z + 2.0)
        mixed_f3 = torch.cat([Z[..., :8] - 2.0, Z[..., 8:] + 2.0], dim=-1)
        mixed_f4 = torch.cat([Z[..., :8] + 2.0, Z[..., 8:] - 2.0], dim=-1)
        f3 = -_griewank(mixed_f3)
        f4 = -_griewank(mixed_f4)
        return torch.stack([f1, f2, f3, f4], dim=-1)

    return Problem(
        "griewank_16d", bounds, th, fn, noise_std, context_dims=tuple(range(8, 16))
    )


def alpine_12d(
    noise_std: float = 0.0, thresholds: Optional[Sequence[float]] = None
) -> Problem:
    """Benchmark 16 - 3-Objective Shifted Alpine N.1 (Medium-D).

    D=12 (d_x=6, d_w=6) on [-10, 10]^12. Absolute value operations make
    derivatives non-differentiable at local minima, testing the robustness
    of the acquisition optimizer under non-smooth landscape kinks.
    """
    bounds = torch.tensor([[-10.0] * 12, [10.0] * 12])
    # p90 per objective; joint feasibility ~5% under correlated shifts
    th = _default_thresholds(thresholds, (-27.0, -31.0, -30.0))

    def _alpine(V: torch.Tensor) -> torch.Tensor:
        return (V * torch.sin(V) + 0.1 * V).abs().sum(dim=-1)

    def fn(Z: torch.Tensor) -> torch.Tensor:
        f1 = -_alpine(Z)
        f2 = -_alpine(Z - 2.0)
        f3 = -_alpine(Z + 2.0)
        return torch.stack([f1, f2, f3], dim=-1)

    return Problem(
        "alpine_12d", bounds, th, fn, noise_std, context_dims=tuple(range(6, 12))
    )


def ackley_rosenbrock_6d(
    noise_std: float = 0.0, thresholds: Optional[Sequence[float]] = None
) -> Problem:
    """Benchmark 17 - 2-Objective Ackley vs. Rosenbrock (Low-D).

    D=6 (d_x=3, d_w=3) on [-2, 2]^6. Pits a central, symmetric, multi-modal
    funnel (Ackley) directly against an asymmetric, flat, banana-shaped valley
    (Rosenbrock). Tests the acquisition's ability to balance a radially-symmetric
    trap against an ill-conditioned curving manifold.
    """
    bounds = torch.tensor([[-2.0] * 6, [2.0] * 6])
    th = _default_thresholds(thresholds, (-4.8, -640.0))

    def fn(Z: torch.Tensor) -> torch.Tensor:
        two_pi = 2.0 * math.pi
        ackley = (
            -20.0 * torch.exp(-0.2 * torch.sqrt((Z ** 2).mean(dim=-1)))
            - torch.exp(torch.cos(two_pi * Z).mean(dim=-1))
            + 20.0 + math.e
        )
        f1 = -ackley
        f2 = -(
            100.0 * (Z[..., 1:] - Z[..., :-1] ** 2) ** 2
            + (1.0 - Z[..., :-1]) ** 2
        ).sum(dim=-1)
        return torch.stack([f1, f2], dim=-1)

    return Problem(
        "ackley_rosenbrock_6d", bounds, th, fn, noise_std, context_dims=(3, 4, 5)
    )


def rastrigin_griewank_sphere_20d(
    noise_std: float = 0.0, thresholds: Optional[Sequence[float]] = None
) -> Problem:
    """Benchmark 18 - 3-Objective Rastrigin vs. Griewank vs. Sphere (High-D).

    D=20 (d_x=10, d_w=10) on [-5, 5]^20. A brutal stress test combining three
    different macroscopic structures. Rastrigin introduces massive independent
    local optima, Griewank adds micro-rugged cosine interference over a convex
    macro-structure, and the shifted Sphere acts as a smooth, conflicting anchor.
    """
    bounds = torch.tensor([[-5.0] * 20, [5.0] * 20])
    th = _default_thresholds(thresholds, (-328.0, -1.05, -194.0))

    _sqrt_i_20 = torch.tensor(
        [math.sqrt(i) for i in range(1, 21)], dtype=torch.double
    )

    def fn(Z: torch.Tensor) -> torch.Tensor:
        two_pi = 2.0 * math.pi
        f1 = -(200.0 + (Z ** 2 - 10.0 * torch.cos(two_pi * Z)).sum(dim=-1))
        S = Z - 2.0
        si = _sqrt_i_20.to(dtype=Z.dtype)
        f2 = -(1.0 + (S ** 2 / 4000.0).sum(dim=-1) - torch.cos(S / si).prod(dim=-1))
        f3 = -((Z + 2.0) ** 2).sum(dim=-1)
        return torch.stack([f1, f2, f3], dim=-1)

    return Problem(
        "rastrigin_griewank_sphere_20d", bounds, th, fn, noise_std,
        context_dims=tuple(range(10, 20)),
    )


def styblinski_tang_levy_10d(
    noise_std: float = 0.0, thresholds: Optional[Sequence[float]] = None
) -> Problem:
    """Benchmark 19 - 2-Objective Styblinski-Tang vs. Levy (Medium-D).

    D=10 (d_x=5, d_w=5) on [-5, 5]^10. Tests the algorithm's capability to
    escape two different styles of deceptive local minima. Levy uses a nested
    polynomial-trigonometric trap; Styblinski-Tang features massive, disjoint
    polynomial basins.
    """
    bounds = torch.tensor([[-5.0] * 10, [5.0] * 10])
    th = _default_thresholds(thresholds, (165.0, -17.0))

    def _levy10(V: torch.Tensor) -> torch.Tensor:
        pi = math.pi
        t1 = torch.sin(pi * V[..., 0]) ** 2
        mid = (
            (V[..., :-1] - 1.0) ** 2
            * (1.0 + 10.0 * torch.sin(pi * V[..., 1:]) ** 2)
        ).sum(dim=-1)
        t3 = (V[..., -1] - 1.0) ** 2 * (
            1.0 + torch.sin(2.0 * pi * V[..., -1]) ** 2
        )
        return t1 + mid + t3

    def fn(Z: torch.Tensor) -> torch.Tensor:
        f1 = -0.5 * (Z ** 4 - 16.0 * Z ** 2 + 5.0 * Z).sum(dim=-1)
        v = 1.0 + (Z - 1.0) / 4.0
        f2 = -_levy10(v)
        return torch.stack([f1, f2], dim=-1)

    return Problem(
        "styblinski_tang_levy_10d", bounds, th, fn, noise_std,
        context_dims=(5, 6, 7, 8, 9),
    )


def heterogeneous_quadrants_6d(
    noise_std: float = 0.0, thresholds: Optional[Sequence[float]] = None
) -> Problem:
    """Benchmark 20 - 4-Objective Heterogeneous Quadrants (Low-D).

    D=6 (d_x=3, d_w=3) on [-5, 5]^6. Shifts four entirely different landscape
    topologies (Sphere, Ackley, Rastrigin, Griewank) into four conflicting
    directional quadrants, forming a highly asymmetric feasible intersection.
    """
    bounds = torch.tensor([[-5.0] * 6, [5.0] * 6])
    th = _default_thresholds(thresholds, (-58.0, -10.95, -118.0, -1.0))

    _sqrt_i_6 = torch.tensor(
        [math.sqrt(i) for i in range(1, 7)], dtype=torch.double
    )

    def fn(Z: torch.Tensor) -> torch.Tensor:
        two_pi = 2.0 * math.pi
        zA = Z - 2.0
        zB = Z + 2.0
        zC = torch.cat([Z[..., :3] - 2.0, Z[..., 3:] + 2.0], dim=-1)
        zD = torch.cat([Z[..., :3] + 2.0, Z[..., 3:] - 2.0], dim=-1)
        f1 = -(zA ** 2).sum(dim=-1)
        ackley = (
            -20.0 * torch.exp(-0.2 * torch.sqrt((zB ** 2).mean(dim=-1)))
            - torch.exp(torch.cos(two_pi * zB).mean(dim=-1))
            + 20.0 + math.e
        )
        f2 = -ackley
        f3 = -(60.0 + (zC ** 2 - 10.0 * torch.cos(two_pi * zC)).sum(dim=-1))
        si = _sqrt_i_6.to(dtype=Z.dtype)
        f4 = -(1.0 + (zD ** 2 / 4000.0).sum(dim=-1) - torch.cos(zD / si).prod(dim=-1))
        return torch.stack([f1, f2, f3, f4], dim=-1)

    return Problem(
        "heterogeneous_quadrants_6d", bounds, th, fn, noise_std, context_dims=(3, 4, 5)
    )


def ellipsoid_rastrigin_20d(
    noise_std: float = 0.0, thresholds: Optional[Sequence[float]] = None
) -> Problem:
    """Benchmark 21 - 2-Objective Ellipsoid vs. Rastrigin (High-D).

    D=20 (d_x=10, d_w=10) on [-5.12, 5.12]^20. A critical ARD stress test.
    Objective 1 (Ellipsoid) has heavily ill-conditioned dimensional scaling;
    Objective 2 (Rastrigin) applies equal, heavy multi-modal variance to all dims.
    """
    bounds = torch.tensor([[-5.12] * 20, [5.12] * 20])
    th = _default_thresholds(thresholds, (-15000.0, -362.0))

    _w20 = torch.tensor(
        [1000.0 ** (i / 19.0) for i in range(20)], dtype=torch.double
    )

    def fn(Z: torch.Tensor) -> torch.Tensor:
        two_pi = 2.0 * math.pi
        w = _w20.to(dtype=Z.dtype)
        f1 = -(Z ** 2 * w).sum(dim=-1)
        S = Z - 2.0
        f2 = -(200.0 + (S ** 2 - 10.0 * torch.cos(two_pi * S)).sum(dim=-1))
        return torch.stack([f1, f2], dim=-1)

    return Problem(
        "ellipsoid_rastrigin_20d", bounds, th, fn, noise_std,
        context_dims=tuple(range(10, 20)),
    )


def dixon_rosenbrock_sphere_12d(
    noise_std: float = 0.0, thresholds: Optional[Sequence[float]] = None
) -> Problem:
    """Benchmark 22 - 3-Objective Dixon-Price vs. Rosenbrock vs. Sphere (Medium-D).

    D=12 (d_x=6, d_w=6) on [-5, 5]^12. Combines two narrow, curving valleys
    (Dixon-Price and shifted Rosenbrock) with different polynomial scalings,
    pitted against a smooth Sphere. Hyper-precise gradient tracking is required
    to find the feasible intersection.
    """
    bounds = torch.tensor([[-5.0] * 12, [5.0] * 12])
    th = _default_thresholds(thresholds, (-17700.0, -127400.0, -88.0))

    def fn(Z: torch.Tensor) -> torch.Tensor:
        i = torch.arange(2, Z.shape[-1] + 1, dtype=Z.dtype)
        f1 = -(
            (Z[..., 0] - 1.0) ** 2
            + (i * (2.0 * Z[..., 1:] ** 2 - Z[..., :-1]) ** 2).sum(dim=-1)
        )
        S = Z + 2.0
        f2 = -(
            100.0 * (S[..., 1:] - S[..., :-1] ** 2) ** 2
            + (1.0 - S[..., :-1]) ** 2
        ).sum(dim=-1)
        f3 = -((Z - 2.0) ** 2).sum(dim=-1)
        return torch.stack([f1, f2, f3], dim=-1)

    return Problem(
        "dixon_rosenbrock_sphere_12d", bounds, th, fn, noise_std,
        context_dims=tuple(range(6, 12)),
    )


def zakharov_ackley_10d(
    noise_std: float = 0.0, thresholds: Optional[Sequence[float]] = None
) -> Problem:
    """Benchmark 23 - 2-Objective Zakharov vs. Ackley (Medium-D).

    D=10 (d_x=5, d_w=5) on [-5, 5]^10. Zakharov operates as a massive, steep,
    asymmetrical plate with highly correlated dimension gradients, opposing
    Ackley's flat outer region and deep center. Tests the acquisition's capacity
    to balance extreme gradient magnitudes.
    """
    bounds = torch.tensor([[-5.0] * 10, [5.0] * 10])
    th = _default_thresholds(thresholds, (-990.0, -10.24))

    def fn(Z: torch.Tensor) -> torch.Tensor:
        two_pi = 2.0 * math.pi
        i = torch.arange(1, Z.shape[-1] + 1, dtype=Z.dtype)
        s1 = (Z ** 2).sum(dim=-1)
        s2 = (0.5 * i * Z).sum(dim=-1)
        f1 = -(s1 + s2 ** 2 + s2 ** 4)
        S = Z - 2.0
        ackley = (
            -20.0 * torch.exp(-0.2 * torch.sqrt((S ** 2).mean(dim=-1)))
            - torch.exp(torch.cos(two_pi * S).mean(dim=-1))
            + 20.0 + math.e
        )
        f2 = -ackley
        return torch.stack([f1, f2], dim=-1)

    return Problem(
        "zakharov_ackley_10d", bounds, th, fn, noise_std, context_dims=(5, 6, 7, 8, 9)
    )


def all_valley_8d(
    noise_std: float = 0.0, thresholds: Optional[Sequence[float]] = None
) -> Problem:
    """Benchmark 24 - 4-Objective "All-Valley" Trade-off (Medium-D).

    D=8 (d_x=4, d_w=4) on [-2, 2]^8. A severe 4-way trade-off composed
    entirely of shifted, narrow valleys (two Rosenbrock variants and two
    Dixon-Price variants), forcing the algorithm to balance between four
    competing ridge floors simultaneously.
    """
    bounds = torch.tensor([[-2.0] * 8, [2.0] * 8])
    th = _default_thresholds(thresholds, (-1650.0, -252.0, -6065.0, -585.0))

    def fn(Z: torch.Tensor) -> torch.Tensor:
        i = torch.arange(2, Z.shape[-1] + 1, dtype=Z.dtype)
        # f1: Rosenbrock(z)
        f1 = -(
            100.0 * (Z[..., 1:] - Z[..., :-1] ** 2) ** 2
            + (1.0 - Z[..., :-1]) ** 2
        ).sum(dim=-1)
        # f2: DixonPrice(z)
        f2 = -(
            (Z[..., 0] - 1.0) ** 2
            + (i * (2.0 * Z[..., 1:] ** 2 - Z[..., :-1]) ** 2).sum(dim=-1)
        )
        # f3: Rosenbrock(z - 1.0)
        S3 = Z - 1.0
        f3 = -(
            100.0 * (S3[..., 1:] - S3[..., :-1] ** 2) ** 2
            + (1.0 - S3[..., :-1]) ** 2
        ).sum(dim=-1)
        # f4: DixonPrice(z + 1.0)
        S4 = Z + 1.0
        f4 = -(
            (S4[..., 0] - 1.0) ** 2
            + (i * (2.0 * S4[..., 1:] ** 2 - S4[..., :-1]) ** 2).sum(dim=-1)
        )
        return torch.stack([f1, f2, f3, f4], dim=-1)

    return Problem(
        "all_valley_8d", bounds, th, fn, noise_std, context_dims=(4, 5, 6, 7)
    )


def schwefel_styblinski_sphere_20d(
    noise_std: float = 0.0, thresholds: Optional[Sequence[float]] = None
) -> Problem:
    """Benchmark 25 - 3-Objective Schwefel vs. Styblinski-Tang vs. Sphere (High-D).

    D=20 (d_x=10, d_w=10) on [-5, 5]^20. The Schwefel global optimum is hidden
    at the boundary of the search space (inputs scaled ×100 to map its canonical
    [-500, 500] topology into [-5, 5]), explicitly testing boundary exploration.
    Pitted against Styblinski-Tang's massive internal basins.
    """
    bounds = torch.tensor([[-5.0] * 20, [5.0] * 20])
    th = _default_thresholds(thresholds, (-7650.0, -625.0, -194.0))

    def fn(Z: torch.Tensor) -> torch.Tensor:
        u = 100.0 * Z
        schwefel = 8379.658 - (u * torch.sin(torch.sqrt(u.abs()))).sum(dim=-1)
        f1 = -schwefel
        S = Z - 2.0
        f2 = -0.5 * (S ** 4 - 16.0 * S ** 2 + 5.0 * S).sum(dim=-1)
        f3 = -((Z + 2.0) ** 2).sum(dim=-1)
        return torch.stack([f1, f2, f3], dim=-1)

    return Problem(
        "schwefel_styblinski_sphere_20d", bounds, th, fn, noise_std,
        context_dims=tuple(range(10, 20)),
    )


def griewank_vlmop2_16d(
    noise_std: float = 0.0, thresholds: Optional[Sequence[float]] = None
) -> Problem:
    """Benchmark 26 - 3-Objective "Micro-Rugged vs. Macro-Flat" (High-D).

    D=16 (d_x=8, d_w=8) on [-5, 5]^16. Griewank is jagged everywhere,
    providing constant but misleading local gradient information. The two VLMOP2
    objectives provide zero gradient information (exponentially flat) until the
    algorithm is directly on top of the tiny feasible region.
    """
    bounds = torch.tensor([[-5.0] * 16, [5.0] * 16])
    # f2/f3 evaluate to exactly -1.0 (IEEE underflow) over almost all of [-5,5]^16;
    # thresholds slightly below -1 keep those constraints always satisfied so that
    # f1 (Griewank) alone controls joint feasibility (~10%).
    th = _default_thresholds(thresholds, (-1.024, -1.1, -1.1))

    _sqrt_i_16 = torch.tensor(
        [math.sqrt(i) for i in range(1, 17)], dtype=torch.double
    )
    _inv_sqrt16 = 1.0 / math.sqrt(16.0)

    def fn(Z: torch.Tensor) -> torch.Tensor:
        si = _sqrt_i_16.to(dtype=Z.dtype)
        f1 = -(1.0 + (Z ** 2 / 4000.0).sum(dim=-1) - torch.cos(Z / si).prod(dim=-1))
        f2 = -(1.0 - torch.exp(-((Z - _inv_sqrt16) ** 2).sum(dim=-1)))
        f3 = -(1.0 - torch.exp(-((Z + _inv_sqrt16) ** 2).sum(dim=-1)))
        return torch.stack([f1, f2, f3], dim=-1)

    return Problem(
        "griewank_vlmop2_16d", bounds, th, fn, noise_std,
        context_dims=tuple(range(8, 16)),
    )


REGISTRY = {
    "two_circles_2d": two_circles_2d,
    "contextual_circles_3d": contextual_circles_3d,
    "sphere2_6d": sphere2_6d,
    "rosenbrock_sphere_6d": rosenbrock_sphere_6d,
    "multimodal_trap_20d": multimodal_trap_20d,
    "dtlz2_6d": dtlz2_6d,
    "styblinski_tang_6d": styblinski_tang_6d,
    "dtlz7_6d": dtlz7_6d,
    "ellipsoid_20d": ellipsoid_20d,
    "dtlz1_12d": dtlz1_12d,
    "zdt3_6d": zdt3_6d,
    "levy_16d": levy_16d,
    "dtlz3_8d": dtlz3_8d,
    "vlmop2_6d": vlmop2_6d,
    "dtlz4_12d": dtlz4_12d,
    "dixon_price_10d": dixon_price_10d,
    "griewank_16d": griewank_16d,
    "alpine_12d": alpine_12d,
    "ackley_rosenbrock_6d": ackley_rosenbrock_6d,
    "rastrigin_griewank_sphere_20d": rastrigin_griewank_sphere_20d,
    "styblinski_tang_levy_10d": styblinski_tang_levy_10d,
    "heterogeneous_quadrants_6d": heterogeneous_quadrants_6d,
    "ellipsoid_rastrigin_20d": ellipsoid_rastrigin_20d,
    "dixon_rosenbrock_sphere_12d": dixon_rosenbrock_sphere_12d,
    "zakharov_ackley_10d": zakharov_ackley_10d,
    "all_valley_8d": all_valley_8d,
    "schwefel_styblinski_sphere_20d": schwefel_styblinski_sphere_20d,
    "griewank_vlmop2_16d": griewank_vlmop2_16d,
}
# spacecraft_formation_flying_a1 is added to REGISTRY after its definition below.


# --------------------------------------------------------------------------- #
# Spacecraft Formation Flying — A1 (LQR) controller                            #
# --------------------------------------------------------------------------- #

import json as _json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .ff_sim_broker_client import (
    broker_is_available as _ff_broker_heartbeat_ok,
    default_queue_dir as _ff_broker_default_queue_dir,
    submit_via_broker as _ff_broker_submit,
)


def _ff_sim_use_broker() -> bool:
    """Opt-in switch for the Slurm-dispatch broker (see ff_sim_broker_client.py
    and scripts/ff_sim_broker/broker.py). Defaults to OFF: unset/false means
    ``_slurm_batch_fn`` behaves exactly as before (one direct ``sbatch`` call
    per evaluate_true() call), so this has zero effect on any run that
    doesn't explicitly set FF_SIM_USE_BROKER=1.
    """
    return os.environ.get("FF_SIM_USE_BROKER", "0").strip().lower() in ("1", "true", "yes")


# Deadband geometry (must match scarlet-gamma-deakin-dev/utils.py)
_DEADBAND_TARGET = [0.0, -245.0, 0.0]        # Hill frame, meters
_DEADBAND_HALF_WIDTHS = [52.5, 105.0, 125.0]  # meters
_POSITION_SCALE = 1.25                          # "hard" Monte Carlo profile

# Hyperparameter search bounds (widened to encompass the feasible region
# demonstrated in the A1_hard historical dataset).
_Q_LOG_LO, _Q_LOG_HI = -12.0, -6.0    # log10 of Q weights
_R_LOG_LO, _R_LOG_HI = -1.0, 0.0      # log10 of R weights
_DT_LO, _DT_HI = 1.0, 50.0          # control_update_interval [s]

# Context (initial condition) bounds
_POS_LO = [t - _POSITION_SCALE * h for t, h in zip(_DEADBAND_TARGET, _DEADBAND_HALF_WIDTHS)]
_POS_HI = [t + _POSITION_SCALE * h for t, h in zip(_DEADBAND_TARGET, _DEADBAND_HALF_WIDTHS)]
_VEL_LO = [-0.1, -0.1, -0.1]  # m/s
_VEL_HI = [0.1, 0.1, 0.1]     # m/s

# Constraint thresholds (raw metric values)
_RMSE_THRESHOLD = 150.0          # meters
_FUEL_THRESHOLD_G = 150.0        # grams
_PEAK_THRUST_THRESHOLD = 5.0     # Newtons (thruster saturates at 100 N)
_PEAK_THRUST_PENALTY = 200.0     # penalty value for failed/timed-out simulations

# Numerical stability transform breakpoints for large values
_RMSE_TRANSFORM_BREAK = 2000.0   # meters
_FUEL_TRANSFORM_BREAK = 2000.0   # grams

# Historical data filter thresholds (for initialization pool)
_HIST_RMSE_FILTER_M = 1000.0   # meters
_HIST_FUEL_FILTER_KG = 1.0    # kg = 1000 g

# D = 16: 6 Q (log10) + 3 R (log10) + 1 dt + 3 pos + 3 vel
# context_dims = (10, 11, 12, 13, 14, 15)
_FF_BOUNDS = torch.tensor(
    [
        [_Q_LOG_LO] * 6 + [_R_LOG_LO] * 3 + [_DT_LO] + _POS_LO + _VEL_LO,
        [_Q_LOG_HI] * 6 + [_R_LOG_HI] * 3 + [_DT_HI] + _POS_HI + _VEL_HI,
    ],
    dtype=torch.double,
)


def _transform_rmse(rmse_m: Optional[float]) -> float:
    """Compress large RMSE values for numerical stability."""
    if rmse_m is None or not math.isfinite(rmse_m):
        return _RMSE_TRANSFORM_BREAK + 1000.0
    if rmse_m <= _RMSE_TRANSFORM_BREAK:
        return rmse_m
    excess = rmse_m - _RMSE_TRANSFORM_BREAK
    return _RMSE_TRANSFORM_BREAK + (math.log10(excess) if excess > 0.0 else 0.0)


def _transform_fuel_g(fuel_g: Optional[float]) -> float:
    """Compress large fuel values for numerical stability."""
    if fuel_g is None or not math.isfinite(fuel_g):
        return _FUEL_TRANSFORM_BREAK + 1000.0
    if fuel_g <= _FUEL_TRANSFORM_BREAK:
        return fuel_g
    excess = fuel_g - _FUEL_TRANSFORM_BREAK
    return _FUEL_TRANSFORM_BREAK + (math.log10(excess) if excess > 0.0 else 0.0)


def _metrics_to_constraints(
    rmse_m: Optional[float],
    fuel_kg: Optional[float],
    peak_thrust_n: Optional[float],
    *,
    rmse_threshold: float = _RMSE_THRESHOLD,
    fuel_threshold_g: float = _FUEL_THRESHOLD_G,
    peak_thrust_threshold: float = _PEAK_THRUST_THRESHOLD,
) -> list:
    """Convert raw simulation metrics to constraint values.

    Returns a 3-element list where positive values indicate satisfied constraints:
        y[0] = rmse_threshold     - transform(RMSE [m])
        y[1] = fuel_threshold_g   - transform(fuel [g])
        y[2] = peak_thrust_threshold - peak_thrust_command [N]

    The ITCAS thresholds tensor is all zeros, so y[i] ≥ 0 ↔ constraint i satisfied.
    Keyword-only threshold overrides default to the module-level constants.
    """
    fuel_g = (fuel_kg * 1000.0) if fuel_kg is not None and math.isfinite(fuel_kg) else None
    y0 = rmse_threshold - _transform_rmse(rmse_m)
    y1 = fuel_threshold_g - _transform_fuel_g(fuel_g)
    y2 = peak_thrust_threshold - (peak_thrust_n if peak_thrust_n is not None else _PEAK_THRUST_PENALTY)
    return [y0, y1, y2]


def _penalty_constraints(
    *,
    rmse_threshold: float = _RMSE_THRESHOLD,
    fuel_threshold_g: float = _FUEL_THRESHOLD_G,
    peak_thrust_threshold: float = _PEAK_THRUST_THRESHOLD,
) -> list:
    """Worst-case constraint values for failed or timed-out simulations."""
    return [
        rmse_threshold - (_RMSE_TRANSFORM_BREAK + 1000.0),
        fuel_threshold_g - (_FUEL_TRANSFORM_BREAK + 1000.0),
        peak_thrust_threshold - _PEAK_THRUST_PENALTY,
    ]


def _resolve_smartsat_root(smartsat_root: Optional[str]) -> Path:
    if smartsat_root is not None:
        return Path(smartsat_root).resolve()
    # problems.py lives at ITCAS/itcas/pipeline/problems.py
    # SmartSat is at <parent-of-ITCAS>/SmartSat/
    _here = Path(__file__).resolve()
    _itcas_root = _here.parent.parent.parent
    return (_itcas_root.parent / "SmartSat").resolve()


class SpacecraftFormationFlyingA1(Problem):
    """Spacecraft formation flying with A1 (LQR) controller.

    D = 16: 10 design dimensions (hyperparameters) + 6 context (initial conditions).

    Design variables (indices 0-9):
        0-5  : Q weights in log10 scale, each in [-15, -12]
        6-8  : R weights in log10 scale, each in [-4, 0]
        9    : control_update_interval [s], in [1, 200]

    Context variables (indices 10-15):
        10-12: initial position in Hill frame [m]
        13-15: initial velocity in Hill frame [m/s]

    Constraints (thresholds all 0; y[i] ≥ 0 means constraint i satisfied):
        0: rmse_threshold_m    - transform(RMSE [m])
        1: fuel_threshold_g    - transform(fuel [g])
        2: peak_thrust_threshold_n - peak_thrust_command [N]

    Difficulty levels (set via thresholds=[rmse_m, fuel_g, thrust_n]):
        Level 1 (strict):  rmse ≤ 50 m,  fuel ≤ 50 g,  thrust ≤ 2 N
        Level 2:           rmse ≤ 100 m, fuel ≤ 100 g, thrust ≤ 5 N
        Level 3:           rmse ≤ 150 m, fuel ≤ 150 g, thrust ≤ 10 N
        Level 4 (lenient): rmse ≤ 200 m, fuel ≤ 200 g, thrust ≤ 10 N

    Evaluation runs Basilisk simulations via parallel Slurm array jobs.

    Initialization: the first call to sample_uniform() draws from a pre-filtered
    pool of A1_hard historical evaluations (RMSE < 500 m, fuel < 300 g, params in
    range) using the provided seed, enabling reproducible warm starts. Subsequent
    calls to sample_uniform() return uniform random samples from the bounds.
    """

    def __init__(
        self,
        smartsat_root: Optional[str] = None,
        poll_interval: float = 30.0,
        job_timeout: float = 1800.0,
        sim_conda_env: Optional[str] = None,
        rmse_threshold_m: Optional[float] = None,
        fuel_threshold_g: Optional[float] = None,
        peak_thrust_threshold_n: Optional[float] = None,
    ):
        super().__init__(
            name="spacecraft_formation_flying_a1",
            bounds=_FF_BOUNDS.clone(),
            thresholds=torch.zeros(3, dtype=torch.double),
            fn=self._slurm_batch_fn,
            noise_std=0.0,
            context_dims=tuple(range(10, 16)),
        )
        self._rmse_threshold_m: float = (
            float(rmse_threshold_m) if rmse_threshold_m is not None else _RMSE_THRESHOLD
        )
        self._fuel_threshold_g: float = (
            float(fuel_threshold_g) if fuel_threshold_g is not None else _FUEL_THRESHOLD_G
        )
        self._peak_thrust_threshold_n: float = (
            float(peak_thrust_threshold_n) if peak_thrust_threshold_n is not None else _PEAK_THRUST_THRESHOLD
        )
        self._smartsat_root = _resolve_smartsat_root(smartsat_root)
        self._poll_interval = float(poll_interval)
        self._job_timeout = float(job_timeout)
        # Conda env the Basilisk simulator runs in. The simulator needs
        # ``control`` + ``matplotlib`` (present in ``scarlet``, NOT in the
        # ``itcas`` BO env). We must pin this explicitly: the parent sweep job
        # exports CONDA_ENV=itcas, and a default ``sbatch --export=ALL`` would
        # leak that into the child FF job, overriding the launcher's
        # ``CONDA_ENV:-scarlet`` default and making every simulation fail its
        # ``import control`` preflight (yielding penalty constraints).
        self._sim_conda_env = (
            sim_conda_env
            or os.environ.get("FF_SIM_CONDA_ENV")
            or "scarlet"
        )

        self._hist_X: Optional[torch.Tensor] = None
        self._hist_Y: Optional[torch.Tensor] = None
        self._hist_loaded: bool = False

        # True until the first call to sample_uniform(); uses historical data
        # for warm-start initialization, then switches to uniform sampling.
        self._use_history_next: bool = True

        # Cache: tuple(x.tolist()) -> y tensor (populated from historical data)
        self._eval_cache: dict[tuple, torch.Tensor] = {}

    # ------------------------------------------------------------------ #
    # Historical data                                                       #
    # ------------------------------------------------------------------ #

    def _load_historical_data(self) -> None:
        """Load filtered A1_hard results into _hist_X and _hist_Y (once)."""
        if self._hist_loaded:
            return
        self._hist_loaded = True

        results_dir = (
            self._smartsat_root / "scarlet-gamma-deakin-dev" / "results" / "A1_hard"
        )
        std_settings_path = (
            self._smartsat_root
            / "scarlet-gamma-deakin-dev"
            / "settings"
            / "settings_A1.json"
        )
        opt_settings_dir = (
            self._smartsat_root
            / "scarlet-gamma-deakin-dev"
            / "settings"
            / "optimized_hyperparameters"
            / "A1"
            / "settings"
        )

        if not results_dir.exists():
            return

        std_sets: dict[str, dict] = {}
        if std_settings_path.exists():
            with open(std_settings_path) as f:
                data = _json.load(f)
            for h in data.get("hyperparameter_sets", []):
                std_sets[str(h["name"])] = h["parameters"]

        opt_sets: dict[str, dict] = {}
        if opt_settings_dir.exists():
            for p in opt_settings_dir.glob("*.json"):
                try:
                    with open(p) as f:
                        opt_sets[p.stem] = _json.load(f)
                except Exception:
                    pass

        Q_LO_ACT = 10.0 ** _Q_LOG_LO
        Q_HI_ACT = 10.0 ** _Q_LOG_HI
        R_LO_ACT = 10.0 ** _R_LOG_LO
        R_HI_ACT = 10.0 ** _R_LOG_HI

        def _in_range(params: dict) -> bool:
            dt = float(params.get("control_update_interval", 0.0))
            if not (_DT_LO <= dt <= _DT_HI):
                return False
            q = params.get("Q_weight")
            if isinstance(q, list):
                if any(not (Q_LO_ACT <= v <= Q_HI_ACT) for v in q):
                    return False
            elif isinstance(q, (int, float)):
                if not (Q_LO_ACT <= q <= Q_HI_ACT):
                    return False
            r = params.get("R_weight")
            if isinstance(r, list):
                if any(not (R_LO_ACT <= v <= R_HI_ACT) for v in r):
                    return False
            elif isinstance(r, (int, float)):
                if not (R_LO_ACT <= r <= R_HI_ACT):
                    return False
            return True

        def _to_x_design(params: dict) -> Optional[list]:
            """Return 10-element design vector [q_log×6, r_log×3, dt]."""
            q = params.get("Q_weight")
            r = params.get("R_weight")
            dt = float(params.get("control_update_interval", 1.0))
            try:
                if isinstance(q, list):
                    if len(q) != 6:
                        return None
                    q_log = [math.log10(v) for v in q]
                elif isinstance(q, (int, float)):
                    q_log = [math.log10(float(q))] * 6
                else:
                    return None
                if isinstance(r, list):
                    if len(r) != 3:
                        return None
                    r_log = [math.log10(v) for v in r]
                elif isinstance(r, (int, float)):
                    r_log = [math.log10(float(r))] * 3
                else:
                    return None
            except (ValueError, ZeroDivisionError):
                return None
            return q_log + r_log + [dt]

        X_rows: list[list] = []
        Y_rows: list[list] = []

        for setting_dir in results_dir.glob("setting_*"):
            if not setting_dir.is_dir():
                continue
            sid = setting_dir.name.replace("setting_", "")

            summary_path = setting_dir / "setting_summary.json"
            if not summary_path.exists():
                continue
            try:
                with open(summary_path) as f:
                    summary = _json.load(f)
            except Exception:
                continue

            rmse_mean = summary.get("rmse_position_mean")
            fuel_mean = summary.get("fuel_consumption_mean")
            if (
                rmse_mean is None
                or fuel_mean is None
                or not math.isfinite(rmse_mean)
                or not math.isfinite(fuel_mean)
                or rmse_mean >= _HIST_RMSE_FILTER_M
                or fuel_mean >= _HIST_FUEL_FILTER_KG
            ):
                continue

            params = std_sets.get(sid) or opt_sets.get(sid)
            if params is None or not _in_range(params):
                continue

            x_design = _to_x_design(params)
            if x_design is None:
                continue

            for run_file in sorted(setting_dir.glob("run_*.json")):
                try:
                    with open(run_file) as f:
                        run = _json.load(f)
                except Exception:
                    continue
                pos = run.get("initial_position")
                vel = run.get("initial_velocity")
                if pos is None or vel is None or len(pos) != 3 or len(vel) != 3:
                    continue
                y = _metrics_to_constraints(
                    run.get("rmse_position"),
                    run.get("fuel_consumption"),
                    run.get("peak_thrust_command"),
                    rmse_threshold=self._rmse_threshold_m,
                    fuel_threshold_g=self._fuel_threshold_g,
                    peak_thrust_threshold=self._peak_thrust_threshold_n,
                )
                X_rows.append(x_design + list(pos) + list(vel))
                Y_rows.append(y)

        if X_rows:
            self._hist_X = torch.tensor(X_rows, dtype=torch.double)
            self._hist_Y = torch.tensor(Y_rows, dtype=torch.double)

    def get_historical_data(
        self,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return (X_hist, Y_hist) tensors of all filtered historical evaluations."""
        self._load_historical_data()
        return self._hist_X, self._hist_Y

    def get_initial_data_for_seed(
        self, n: int, seed: int
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Return (X, Y, row_indices) for n historical rows selected by seed.

        The row_indices allow recovery of the exact evaluations used, enabling
        reproducibility: the same seed always selects the same rows.
        """
        self._load_historical_data()
        if self._hist_X is None:
            raise RuntimeError("No historical data available for initialization.")
        N = self._hist_X.shape[0]
        n_draw = min(n, N)
        g = torch.Generator()
        g.manual_seed(seed)
        idx = torch.randperm(N, generator=g)[:n_draw].tolist()
        return self._hist_X[idx], self._hist_Y[idx], idx

    # ------------------------------------------------------------------ #
    # Problem interface overrides                                           #
    # ------------------------------------------------------------------ #

    def sample_uniform(
        self,
        n: int,
        seed: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Return n points from bounds.

        On the first call (initialization), draws from the pre-filtered
        historical pool using ``seed`` for reproducibility and pre-populates
        the eval cache so that the subsequent evaluate() call returns the
        stored Y values without running new Slurm simulations.

        All subsequent calls return uniformly sampled points from the bounds
        (standard behavior used for candidate pool generation).
        """
        self._load_historical_data()

        if self._use_history_next and self._hist_X is not None:
            N_hist = self._hist_X.shape[0]
            n_draw = min(n, N_hist)

            g = torch.Generator()
            if seed is not None:
                g.manual_seed(seed)
            idx = torch.randperm(N_hist, generator=g)[:n_draw].tolist()

            X_sel = self._hist_X[idx]

            # Pre-populate eval cache: evaluate() will return these Y values
            # directly without submitting Slurm jobs.
            for hist_i in idx:
                key = tuple(self._hist_X[hist_i].tolist())
                self._eval_cache[key] = self._hist_Y[hist_i]

            self._use_history_next = False

            if n_draw < n:
                extra = super().sample_uniform(
                    n - n_draw, seed=seed, device="cpu", dtype=torch.double
                )
                X_sel = torch.cat([X_sel, extra], dim=0)

            if dtype is not None:
                X_sel = X_sel.to(dtype=dtype)
            if device is not None:
                X_sel = X_sel.to(device=device)
            return X_sel

        self._use_history_next = False
        return super().sample_uniform(n, seed=seed, device=device, dtype=dtype)

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        """Return (N, 3) constraint tensor.

        Points found in the eval cache (populated from historical data during
        initialization) are returned instantly. All other points are evaluated
        by submitting Slurm array jobs via _slurm_batch_fn.
        """
        X_cpu = X.detach().to("cpu", dtype=torch.double)
        N = X_cpu.shape[0]

        Y_out = torch.full((N, 3), float("nan"), dtype=torch.double)
        uncached_local: list[int] = []
        uncached_rows: list[torch.Tensor] = []

        for i in range(N):
            key = tuple(X_cpu[i].tolist())
            if key in self._eval_cache:
                Y_out[i] = self._eval_cache[key].to(dtype=torch.double)
            else:
                uncached_local.append(i)
                uncached_rows.append(X_cpu[i])

        if uncached_rows:
            X_new = torch.stack(uncached_rows, dim=0)
            Y_new = self._slurm_batch_fn(X_new)
            for j, i in enumerate(uncached_local):
                Y_out[i] = Y_new[j].to(dtype=torch.double)

        return Y_out.to(device=X.device, dtype=X.dtype)

    def reference_objectives(
        self,
        n: int,
        seed: int,
        thresholds: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Return feasible rows from historical data as reference objectives.

        Avoids launching Slurm simulations just for reference-set construction.
        Returns None if no historical data is available or none is feasible.
        """
        try:
            self._load_historical_data()
        except Exception:
            return None
        if self._hist_Y is None:
            return None
        h = thresholds.to(dtype=torch.double)
        Y = self._hist_Y
        mask = (Y >= h).all(dim=-1)
        return Y[mask] if bool(mask.any()) else None

    # ------------------------------------------------------------------ #
    # Slurm batch evaluator                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ff_row_to_params(row) -> dict:
        """Convert one 16-D design+context row into the simulator's params dict.

        Shared by the direct-sbatch path and the opt-in broker path (see
        ff_sim_broker_client.py) so both build identical simulator inputs.
        """
        q_log = row[0:6].tolist()
        r_log = row[6:9].tolist()
        dt = float(row[9])
        pos = row[10:13].tolist()
        vel = row[13:16].tolist()
        return {
            "Q_weight": [10.0 ** q for q in q_log],
            "R_weight": [10.0 ** r for r in r_log],
            "control_update_interval": dt,
            "initial_position": pos,
            "initial_velocity": vel,
        }

    def _ff_result_dict_to_y(self, res: Optional[dict]) -> list:
        """Convert one parsed result_NNNN.json dict (or None) to a Y row.

        ``None`` (missing/unreadable/failed result) yields the worst-case
        penalty constraints, exactly matching the pre-broker behavior for
        crashed/timed-out simulations. Shared by the direct-sbatch and
        broker dispatch paths so the scoring logic lives in exactly one
        place.
        """
        y = _penalty_constraints(
            rmse_threshold=self._rmse_threshold_m,
            fuel_threshold_g=self._fuel_threshold_g,
            peak_thrust_threshold=self._peak_thrust_threshold_n,
        )
        if res is not None:
            try:
                if res.get("status") == "ok":
                    y = _metrics_to_constraints(
                        res.get("rmse_position"),
                        res.get("fuel_consumption"),
                        res.get("peak_thrust_command"),
                        rmse_threshold=self._rmse_threshold_m,
                        fuel_threshold_g=self._fuel_threshold_g,
                        peak_thrust_threshold=self._peak_thrust_threshold_n,
                    )
            except Exception:
                pass
        return y

    def _slurm_batch_fn(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate N inputs by submitting a Slurm array job.

        Each task runs formation_flying_simulator.py for one row of X.
        Results are collected from output JSON files and returned as (N, 3).
        Failed or timed-out tasks receive worst-case penalty values.

        Dispatch mode: by default (FF_SIM_USE_BROKER unset/false) this
        submits one direct ``sbatch`` job per call, unchanged from before.
        Setting FF_SIM_USE_BROKER=1 in the environment routes the same
        request through the opt-in Slurm-dispatch broker instead (see
        ff_sim_broker_client.py / scripts/ff_sim_broker/broker.py), which
        merges concurrent single-simulation requests from multiple processes
        into fewer, bigger Slurm jobs. Purely a dispatch/resource change --
        scoring is identical either way (_ff_result_dict_to_y is shared).
        """
        if _ff_sim_use_broker():
            return self._slurm_batch_fn_broker(X)

        X_np = X.detach().cpu().double().numpy()
        N = int(X_np.shape[0])

        slurm_script = self._smartsat_root / "ff_sim_batch.sbatch"
        if not slurm_script.exists():
            raise FileNotFoundError(
                f"Slurm script not found: {slurm_script}. "
                "Expected at SmartSat/ff_sim_batch.sbatch"
            )

        # Use a shared-filesystem path so compute nodes can read/write the
        # same directory.  /tmp is node-local on most HPC clusters, so tasks
        # running on compute nodes cannot see files written here on the login
        # node.  SmartSat/ff_sim_work/ is on NFS and is visible cluster-wide.
        work_root = self._smartsat_root / "ff_sim_work"
        work_root.mkdir(parents=True, exist_ok=True)
        job_dir = Path(tempfile.mkdtemp(prefix="ff_slurm_", dir=work_root))
        job_id: str = ""
        n_results: int = 0
        all_ok: bool = False
        try:
            # Write per-task params files
            for i in range(N):
                params = self._ff_row_to_params(X_np[i])
                with open(job_dir / f"params_{i:04d}.json", "w") as f:
                    _json.dump(params, f)

            # Submit Slurm array job.
            #
            # Pin CONDA_ENV to the simulator env via --export. Without this,
            # sbatch defaults to --export=ALL and the child FF job inherits
            # CONDA_ENV=itcas from the parent sweep, so the launcher activates
            # the wrong env (no ``control``) and every task fails its preflight
            # import, producing penalty constraints for all points.
            log_pat = str(job_dir / "slurm_%j.log")
            sbatch_cmd = [
                "sbatch",
                "--parsable",
                f"--cpus-per-task={N}",
                f"--output={log_pat}",
                f"--error={log_pat}",
                f"--export=ALL,CONDA_ENV={self._sim_conda_env}",
                str(slurm_script),
                str(job_dir),
                str(self._smartsat_root),  # argv[2]: avoids $0 path issues when Slurm stages the script
            ]
            try:
                proc = subprocess.run(
                    sbatch_cmd, capture_output=True, text=True, check=True
                )
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or "").strip()
                stdout = (exc.stdout or "").strip()
                details = "\n".join(
                    part for part in [
                        f"stdout: {stdout}" if stdout else "",
                        f"stderr: {stderr}" if stderr else "",
                    ] if part
                )
                raise RuntimeError(
                    "Failed to submit formation-flying Slurm array job via sbatch. "
                    f"Command: {' '.join(sbatch_cmd)}"
                    + (f"\n{details}" if details else "")
                ) from exc
            job_id = proc.stdout.strip().split(";")[0]

            # Poll for result files; also check if the Slurm job has already
            # finished (all tasks completed or failed) so we don't wait the
            # full job_timeout when tasks crash at startup.
            deadline = time.monotonic() + self._job_timeout
            while True:
                n_done = sum(
                    1
                    for i in range(N)
                    if (job_dir / f"result_{i:04d}.json").exists()
                )
                if n_done == N:
                    break
                if time.monotonic() > deadline:
                    subprocess.run(["scancel", job_id], capture_output=True)
                    break
                # Early exit: if the job array is no longer in the queue, all
                # tasks have finished (possibly with errors) — stop waiting.
                try:
                    sq = subprocess.run(
                        ["squeue", "--job", job_id, "--noheader"],
                        capture_output=True, text=True, timeout=10,
                    )
                    if not sq.stdout.strip():
                        break
                except Exception:
                    pass
                time.sleep(self._poll_interval)

            # Read results and convert to constraint values
            n_results = sum(
                1 for i in range(N) if (job_dir / f"result_{i:04d}.json").exists()
            )
            all_ok = n_results == N
            Y_rows: list[list] = []
            for i in range(N):
                result_path = job_dir / f"result_{i:04d}.json"
                res = None
                if result_path.exists():
                    try:
                        with open(result_path) as f:
                            res = _json.load(f)
                    except Exception:
                        res = None
                Y_rows.append(self._ff_result_dict_to_y(res))

        finally:
            if not all_ok and job_id:
                import warnings as _warnings
                _warnings.warn(
                    f"[ff_sim] {N - n_results}/{N} tasks produced no result.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            shutil.rmtree(job_dir, ignore_errors=True)

        return torch.tensor(Y_rows, dtype=X.dtype)

    def _slurm_batch_fn_broker(self, X: torch.Tensor) -> torch.Tensor:
        """Broker-dispatched variant of _slurm_batch_fn (opt-in via
        FF_SIM_USE_BROKER=1). Same inputs/outputs/scoring as the direct
        path -- only how the Slurm job gets submitted differs: this process
        writes its pending simulation requests into a shared queue directory
        instead of calling ``sbatch`` itself, and a separate broker daemon
        (scripts/ff_sim_broker/broker.py) merges requests from potentially
        many concurrent callers into fewer, bigger jobs.

        If no broker daemon appears to be running against the queue
        directory, fails fast with a clear error rather than silently
        blocking every caller for the full job_timeout.
        """
        X_np = X.detach().cpu().double().numpy()
        N = int(X_np.shape[0])

        slurm_script = self._smartsat_root / "ff_sim_batch.sbatch"
        if not slurm_script.exists():
            raise FileNotFoundError(
                f"Slurm script not found: {slurm_script}. "
                "Expected at SmartSat/ff_sim_batch.sbatch"
            )

        queue_dir = Path(
            os.environ.get("FF_SIM_BROKER_QUEUE_DIR")
            or _ff_broker_default_queue_dir(self._smartsat_root)
        )
        if not _ff_broker_heartbeat_ok(queue_dir):
            raise RuntimeError(
                f"FF_SIM_USE_BROKER=1 but no live broker heartbeat found at "
                f"{queue_dir}/broker.heartbeat. Start it with "
                "scripts/ff_sim_broker/start_broker.sh before running with "
                "FF_SIM_USE_BROKER=1 (or unset FF_SIM_USE_BROKER to fall back "
                "to direct per-call sbatch submission)."
            )

        params_list = [self._ff_row_to_params(X_np[i]) for i in range(N)]
        results = _ff_broker_submit(
            params_list,
            queue_dir=queue_dir,
            job_timeout=self._job_timeout,
            poll_interval=self._poll_interval,
        )

        n_missing = sum(1 for r in results if r is None)
        if n_missing:
            import warnings as _warnings
            _warnings.warn(
                f"[ff_sim broker] {n_missing}/{N} tasks produced no result.",
                RuntimeWarning,
                stacklevel=2,
            )

        Y_rows = [self._ff_result_dict_to_y(res) for res in results]
        return torch.tensor(Y_rows, dtype=X.dtype)


def spacecraft_formation_flying_a1(
    smartsat_root: Optional[str] = None,
    poll_interval: float = 30.0,
    job_timeout: float = 1800.0,
    sim_conda_env: Optional[str] = None,
    thresholds: Optional[Sequence[float]] = None,
    **_kwargs,
) -> SpacecraftFormationFlyingA1:
    """Create the spacecraft formation flying A1 problem.

    Registered as ``spacecraft_formation_flying_a1`` in REGISTRY.

    Args:
        smartsat_root: Absolute path to the SmartSat directory. Defaults to
            ``<parent-of-ITCAS>/SmartSat/``.
        poll_interval: Seconds between Slurm result polls (default 30).
        job_timeout: Maximum seconds to wait for all Slurm jobs (default 1800).
        sim_conda_env: Conda env the Basilisk simulator runs in (needs
            ``control`` + ``matplotlib``). Defaults to ``$FF_SIM_CONDA_ENV`` or
            ``scarlet``. Pinned explicitly so the child FF Slurm job does not
            inherit the parent sweep's ``CONDA_ENV=itcas``.
        thresholds: Optional 3-element sequence ``[rmse_m, fuel_g, thrust_n]``
            selecting a named difficulty level. Loaded from
            ``configs/thresholds.json`` via ``--threshold_pct 1|2|3|4``.
            Defaults to the module-level constants when None.
    """
    extra: dict = {}
    if thresholds is not None:
        if len(thresholds) != 3:
            raise ValueError(
                f"spacecraft_formation_flying_a1 expects thresholds=[rmse_m, fuel_g, thrust_n] "
                f"(length 3); got {thresholds}"
            )
        extra["rmse_threshold_m"] = thresholds[0]
        extra["fuel_threshold_g"] = thresholds[1]
        extra["peak_thrust_threshold_n"] = thresholds[2]
    return SpacecraftFormationFlyingA1(
        smartsat_root=smartsat_root,
        poll_interval=poll_interval,
        job_timeout=job_timeout,
        sim_conda_env=sim_conda_env,
        **extra,
    )


REGISTRY["spacecraft_formation_flying_a1"] = spacecraft_formation_flying_a1