# ITCAS

Python implementation of the proposed Contextual MO-CAS algorithm
(pluggable ROI-MI / EFIG / EDIG / NDIG quality + Quality-Diversity DPP batch selection) and
baselines for Constraint Active Search experiments.

## Layout

```
itcas/
  algorithms/      # Continuous C-MO-CAS: RFF Thompson + smooth margin +
                   # pluggable quality (quality.py: ROI-MI / EFIG / EDIG / NDIG)
                   # + QD-DPP submodular batch (continuous.py); discrete pool
                   # kept in roi_mi.py/qd_dpp.py/itcas.py
  baselines/       # Random, ONE-S, EZ, EISR, STRADDLE, CAS-ECI, MOC-CAS hard/soft
                   # (eps-constraint and MOO+Cluster: TODO)
  pipeline/        # Experiment loop, problem registry, casd_client.py (CASD
                   # evaluator-server HTTP client)
  metrics/         # Context fill distance (CFD), feasible context fill distance
                   # (FCFD), FCHV, LogDet, number of positives, AUP
  io/              # JSONL run logger
  reporting/       # Benchmark visualization to PDF
  utils/           # GP construction, seeding
  cli.py           # CLI entry point
  visualize.py     # `python -m itcas.visualize`
configs/           # YAML/JSON experiment configs + calibrated thresholds.json
scripts/           # Slurm submission: install_env, calibrate, submit(.sh),
                   # submit_jobs.sbatch, run_seedset.sh, jobs.json,
                   # casd_server/ (CASD evaluator server, separate env)
tests/             # Pure-Python smoke tests
requirements.txt              # itcas env (BoTorch/GPyTorch) — used by everything above
requirements-casd-server.txt  # SEPARATE env for the CASD evaluator server (see below)
```

## Run

```bash
# Inside an env with: torch, botorch, gpytorch, numpy, scipy (+ optional pyyaml)
python -m itcas.cli --config configs/smoke.yaml
```

### Choosing the ITCAS quality: ROI-MI, EFIG, EDIG, or NDIG

The proposed `itcas` method scores candidates with a pluggable quality term
(`--quality`, or the `quality:` config key). Four variants ship:

- `roi_mi` (**default**) — *Region-of-Interest Mutual Information*. Global
  information gain against a hallucinated continuous reference set built by
  differentiable Thompson sampling + smooth-margin multi-start ascent. Targets
  the deep feasible interior; honours `--gamma` and `--n_ts_samples`.
- `efig` — *Expected Feasible Information Gain*. Weights the continuous info gain
  by the Probability of Feasibility `∏ᵢ Φ(Zᵢ)`. No reference set needed, fast
  and fully local, but can saturate (PoF→1) deep in the interior and can
  vanish in cold-start scenarios.
- `edig` — *Expected Depth Information Gain*. Replaces the PoF multiplier with
  the Standardized Expected Feasible Margin `∏ᵢ [Zᵢ Φ(Zᵢ) + φ(Zᵢ)]`, which
  grows linearly with predicted depth (no saturation) and stays non-zero for
  Z<0 (no cold-start vanishing). Like EFIG, needs no reference set.
- `ndig` — *Normalized Depth Information Gain*. Applies a rational squashing
  function to the EDIG depth term so the QD-DPP L-ensemble stays numerically
  stable and retains context-space diversity.

```bash
python -m itcas.cli --config configs/smoke.yaml --quality roi_mi   # default
python -m itcas.cli --config configs/smoke.yaml --quality efig     # PoF-weighted, cheap
python -m itcas.cli --config configs/smoke.yaml --quality edig     # depth-weighted, robust
python -m itcas.cli --config configs/smoke.yaml --quality ndig     # normalized depth-weighted, diversity-safe
```

Or set it in a config file:

```yaml
method: itcas
quality: efig
```

The active variant is recorded in each iteration's `info.quality` (per-iteration
JSONL) and in the run `summary.json` config. `--quality` only affects `itcas`;
baselines ignore it.

### GPU / device selection

GP fitting and acquisition optimization run on the device selected via
`--device` (or the `device:` config key):

```bash
python -m itcas.cli --config configs/smoke.yaml --device auto    # CUDA if available, else CPU (default)
python -m itcas.cli --config configs/smoke.yaml --device cuda    # force GPU (errors if unavailable)
python -m itcas.cli --config configs/smoke.yaml --device cuda:1  # specific GPU
python -m itcas.cli --config configs/smoke.yaml --device cpu     # force CPU
```

The resolved device is recorded in each run's `summary.json` under `"device"`.
Cheap analytic objective evaluations stay on CPU; the heavy GP work uses the
selected device. RNG is seeded across CPU and CUDA for reproducibility.

Outputs:
- `results/smoke/itcas_smoke.jsonl`     — per-iteration log
- `results/smoke/itcas_smoke.summary.json` — aggregate metrics

### Running a sequence of seeds (resume / fill gaps)

Use `--seeds` to run a whole sequence in one invocation, automatically skipping
seeds whose run already finished (detected via the per-seed
`<run_name>.summary.json` marker). Each seed gets a unique run name: include the
literal `{seed}` in `--run_name` to control placement, otherwise `_seed<n>` is
appended.

```bash
# Already ran seeds 1, 3, 6, 7 earlier; this runs only 2, 4, 5, 8, 9, 10:
python -m itcas.cli --problem sphere2_6d --method itcas --threshold_pct 0.1 \
    --out_dir results/sweep --run_name sphere2_itcas --seeds 1-10

# Preview which seeds are pending without running them:
python -m itcas.cli ... --seeds 1-10 --dry_run

# Re-run every seed regardless of existing outputs:
python -m itcas.cli ... --seeds 1-10 --force
```

Seed specs accept comma-separated singletons and inclusive ranges, e.g.
`1-10`, `1,3,6-8`, `0-2, 5`. Without `--seeds`, the single `--seed` path is
unchanged (run name is used verbatim).

To visualize a benchmark directory that contains one or more runs:

```bash
python -m itcas.visualize --input-dir results/smoke
# or pin to one problem if the directory has more than one
python -m itcas.visualize --input-dir results/sweep --benchmark sphere2_6d
```

This writes **two PDFs per metric per benchmark** comparing every method
(seed-averaged with +/- 1 std band when multiple seeds are present):

- `<problem>_<metric>_vs_evaluations.pdf`
- `<problem>_<metric>_vs_steps.pdf`

with metrics `cumulative_positives` (number of positives),
`context_fill_distance` (CFD), `feasible_context_fill_distance` (FCFD), and
`feasible_convex_hull_volume` (FCHV), and `logdet_diversity` (LogDet). The
context fill-distance metrics need context reference sets; FCHV and LogDet use
problem-specific feasible objective ranges for normalization. Metrics that need
unavailable references are skipped silently, and CFD/FCFD are emitted only for
contextual problems. AUP is a single number (`sum_t P(t)`) recorded in each
run's `summary.json` and is not plotted.

A `--seeds` sweep auto-runs the same visualization at the end of the job and
**writes the comparison PDFs at the comparison root** — the directory above
the per-method `out_dir` (i.e. the difficulty/problem folder under the
cluster layout `<root>/<problem>/<difficulty>/<method>/`). This way the same
set of plots accumulates curves for each new method as its sweep finishes.
Override the location with `--compare_dir <path>`. For flat layouts (all runs
in one folder) the comparison root is just that folder.

## Threshold calibration (difficulty levels)

Feasibility thresholds `tau` for the synthetic benchmarks are calibrated so
that a target percentage of uniformly-sampled points are jointly feasible
(default 10%; smaller = harder). Calibrated values are stored in
`configs/thresholds.json`, keyed by `[problem][percentage]`.

```bash
# Generate thresholds for the four benchmarks at several difficulty levels
python -m itcas.calibrate --percentages 0.20 0.10 0.05 0.01 --n-samples 200000

# Run an experiment using the 10% calibrated thresholds instead of the defaults
python -m itcas.cli --problem sphere2_6d --method itcas --threshold_pct 0.1
```

Each entry records the thresholds, achieved fraction, per-objective maxima/
minima, the solved quantile level, sample count and seed for reproducibility.

## CASD benchmark (LLM decoding-hyperparameter search)

`casd_llm` (registered in `itcas.pipeline.PROBLEM_REGISTRY`) is the
Context-Aware Safe Decoding benchmark described in
[`contexts/llm_application.md`](contexts/llm_application.md): a 5-D contextual
problem (3 decoding hyperparameters `x` + 2 prompt features `c`) whose two
objectives (safety, utility) are scored by actually generating text with an
LLM and judging the output. Unlike every other problem in `PROBLEM_REGISTRY`,
its objective function is **not** evaluated in-process — evaluation is
delegated over HTTP to a separate, persistent evaluator server
(`scripts/casd_server/server.py`) that keeps a vLLM engine and two HF judge
models resident on a GPU. That server runs in its **own** Python environment
(`requirements-casd-server.txt`), never `itcas`'s (`requirements.txt`), because
vLLM's own torch/transformers pinning would conflict with the BoTorch/GPyTorch
stack — see that file's header for the full rationale and install/launch
commands.

The `itcas` process finds the server via the `CASD_SERVER_URL` env var
(default `http://localhost:8008`); see `ContextAwareSafeDecoding` in
`itcas/pipeline/problems.py` for the client-side details (nearest-real-prompt
context snapping onto a sampled real-prompt *neighborhood*, not just the
single snapped prompt; penalty fallback on server errors; etc.).

### Quick local smoke test (no GPU, no model downloads)

The server has a `--mock` mode that skips loading vLLM/the judges/the dataset
entirely and serves cheap synthetic scores instead — enough to exercise the
HTTP protocol and the full BO loop plumbing:

```bash
# In the casd-server env (or anywhere with fastapi+uvicorn installed):
python scripts/casd_server/server.py --mock --port 8008

# In another shell, in the itcas env:
CASD_SERVER_URL=http://localhost:8008 python -m itcas.cli --config configs/casd_llm.yaml
```

### Real runs (GPU required)

Real evaluation needs the server started with real vLLM + judge models
loaded on a GPU node (`CASD_MOCK` unset), which on this project's cluster
means the Slurm-driven launch documented in
[`scripts/README.md`](scripts/README.md#casd-server-h100h200). Once a real
server is running and its address is exported as `CASD_SERVER_URL`, `itcas`
runs against `casd_llm` exactly as in the smoke test above.

**Thresholds are calibrated.** Four difficulty levels
(`configs/thresholds.json["casd_llm"]`, selectable via `--threshold_pct
1|2|3|4`) were hand-picked from a real 1000-sample calibration run under the
current neighborhood-evaluation protocol — see `ContextAwareSafeDecoding`'s
"Difficulty levels" docstring section and
[`contexts/llm_application.md`](contexts/llm_application.md) Part 5 for the
full grid and rationale. `configs/casd_llm.yaml` uses the default, Level 3
("Moderate"). Recalibrate (`itcas/pipeline/calibrate_casd.py`) if the
neighborhood-evaluation protocol, target model, or judge models change.

## Cluster (Slurm) submission

Large-scale GPU sweeps run on Slurm via the operator-owned scripts in
`scripts/`. Slurm is our specific cluster workflow rather than a universal
standard, so those submission/monitoring instructions live separately in
[`scripts/README.md`](scripts/README.md).

## Method (proposed)

See `latex/.../methodology.tex`. The proposed **C-MO-CAS** acquisition is now
fully *continuous* (no discrete candidate grid). Per iteration:

1. **Differentiable Thompson sampling (RFF).** Draw a joint posterior path
   `f^(s) ~ GP(mu_t, Sigma_t)` with BoTorch Matheron pathwise sampling — a
   deterministic, differentiable realization over `X x C`.
2. **Smooth margin.** Feasibility depth is the softmin
   `M^(s)(z) = -gamma log sum_i exp(-(f_i^(s)(z) - tau_i)/gamma)`, which
   lower-bounds `min_i (f_i - tau_i)`, so `M > 0` certifies strict feasibility.
3. **Multi-start `Z_ref` construction.** Gradient ascent of `M^(s)` over
   `X x C` finds local maxima `Z_opt`; the reference set is built without
   relaxation hyperparameters — `{z in Z_opt : M(z) > 0}` when a feasible
   interior exists, else the closest peaks `argmax_z M(z)` (cold start).
4. **Candidate quality (`--quality`, pluggable).** Four registered variants
   feed the QD-DPP; all are differentiable in `z`:
   - `roi_mi` (default) — **Region-of-Interest MI**
     `q(z) = I(y(z); Y(Z_ref) | D_t)` against the hallucinated reference set
     `Z_ref`, via the closed-form rank-1 GP variance update. Steps 1–3 above
     build `Z_ref`; targets the deep feasible interior.
   - `efig` — **Expected Feasible Information Gain** `q(z) = p(z) * I_f(z)`,
     where `p(z)` is the Probability of Feasibility. No reference set or
     Thompson sampling is needed.
   - `edig` — **Expected Depth Information Gain** uses the standardized
     expected feasible margin `∏ᵢ [Zᵢ Φ(Zᵢ) + φ(Zᵢ)]` to avoid EFIG's interior
     saturation and cold-start vanishing.
   - `ndig` — **Normalized Depth Information Gain** is the same depth-based
     score with a rational squashing function, which keeps the DPP kernel more
     numerically stable when depth scores become large.

   New quality measures register via `@register_quality("name")` in
   `algorithms/quality.py` and become available through `--quality name`
   without touching the pipeline.
5. **Continuous QD-DPP batching.** Greedily maximise `F(B) = log det(I + L_B)`
   with `L_ij = q_i [k_obj(mu_i, mu_j) · k_ctx(c_i, c_j)] q_j`; each greedy step
   runs multi-start gradient ascent of the marginal gain over `X x C`. Monotone
   submodularity gives the `(1 - 1/e)` guarantee. The objective-space RBF
   spreads the batch across the feasible performance manifold; the context-space
   RBF space-fills the environment space.

Key `itcas` knobs (CLI/config): `--quality` (`roi_mi` | `efig` | `edig` |
`ndig`), `--gamma` (margin smoothing), `--n_restarts`, `--n_opt_steps`,
`--opt_lr` (continuous optimizer), `--n_ts_samples` (RFF draws),
`--dpp_lambda`, `--dpp_lambda_ctx` (kernel length-scales; median heuristic if
unset). `--gamma` / `--n_ts_samples` only affect `roi_mi`. `--n_candidates`
now only affects the *baselines*.

## Open items / spec ambiguities

- `H(Y(Z_ref))` is approximated by sum-of-binary-entropies (independence
  across `Z_ref` points). Tightening this with a proper joint Bernoulli
  copula is left for follow-up.
- The continuous optimizer uses projected-gradient Adam multi-start; the spec
  mentions L-BFGS-B. Adam was chosen for robustness with the differentiable
  RFF/ROI-MI objectives; swapping in `gen_candidates_scipy` (L-BFGS-B) is a
  drop-in change in `algorithms/continuous.multistart_ascent`.
- The cold-start `Z_ref = argmax_z M(z)` is implemented as the top-`k`
  closest-to-feasible peaks (the spec text says "closest peaks", plural).
- Soft probit gate / unit-mass Gaussian kernel acquisition from
  `moccas.md` remains a separate baseline, not part of this proposed method.
- `eps_constraint` and `moo_cluster` baselines are placeholders.
