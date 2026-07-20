
---

# Benchmark Application: Context-Aware Safe Decoding (CASD) Optimization

### Introduction

Modern deployment of Large Language Models (LLMs) requires a delicate balance between helpfulness and harmlessness. Standard practice typically relies on static, inference-time hyperparameters (e.g., a fixed temperature or top-p value) applied uniformly across all user queries. However, this "one-size-fits-all" approach fails in real-world scenarios. A benign coding question benefits from higher temperature to explore creative solutions, whereas a politically sensitive or borderline-toxic prompt requires strict, low-temperature, highly penalized decoding to prevent the model from generating harmful content or falling into degenerate loops.

The **Context-Aware Safe Decoding (CASD)** benchmark models this exact tension. It challenges algorithms operating in the CMCAS setting to dynamically map out the safe operating boundaries of an LLM. The goal for any evaluated algorithm is to sample-efficiently discover a *diverse set* of feasible decoding hyperparameters that satisfy both safety and utility thresholds, adapting dynamically to the incoming prompt's inherent risk level. This provides a fast, highly practical, and low-dimensional testbed for evaluating active search algorithms on modern generative AI pipelines.

---

### Part 1: Problem Formulation

The CASD benchmark is formulated with a continuous, low-dimensional input space, a continuous context space representing the user's prompt, and two competing black-box objective functions evaluated via secondary AI judges.

* **Input Space ($x \in \mathbb{R}^3$):** The continuous decoding hyperparameters passed to the LLM generation API.
1. **Temperature** (Bounds: $[0.1, 2.0]$): Controls the randomness of the token distribution.
2. **Top-p / Nucleus Sampling** (Bounds: $[0.1, 1.0]$): Truncates the long tail of the probability distribution.
3. **Repetition Penalty** (Bounds: $[1.0, 2.0]$): Penalizes degenerate or repetitive generation loops, a common failure mode during safety breakdowns.


* **Context Space ($c \in \mathbb{R}^2$):** The semantic characteristics of the incoming user prompt.
1. **Prompt Toxicity** (Bounds: $[0.0, 1.0]$): The inherent risk of the prompt (e.g., asking for a recipe vs. probing for dangerous chemical synthesis).
2. **Prompt Length** (Bounds: $[0.0, 1.0]$ normalized): Shorter prompts often require the model to extrapolate more, carrying different hallucination and safety risks than long, highly detailed prompts.


* **Multi-Objective Values ($f(x, c)$):**
1. $f_1$: **Safety Score (Maximize):** The inverse toxicity of the *generated* text. Must strictly exceed a safety threshold $\tau_1$.
2. $f_2$: **Utility / Helpfulness (Maximize):** A proxy score for how well the generated text addresses the prompt, ensuring the model doesn't safely but uselessly output "I cannot answer that" every time. Must exceed a utility threshold $\tau_2$.

  Calibrated $(\tau_1, \tau_2)$ pairs are given in Part 5 below — see that section for why they had to be derived differently than a straightforward Monte-Carlo quantile fit.



---

### Part 2: Real-World Practicality and Semantic Routing

This benchmark perfectly mirrors the rapidly adopting architecture of **Semantic Routers and Input Guardrails** in enterprise LLM deployments.

In production environments, it is standard practice to measure a prompt's context *before* invoking the massive, expensive target LLM. Frameworks like NVIDIA NeMo Guardrails or vLLM Semantic Routers intercept HTTP requests in the proxy data plane, evaluating signals like prompt toxicity and complexity in milliseconds using tiny embedding models. This saves computing costs and allows for dynamic parameter routing.

Algorithms solving the CMCAS problem map directly to the offline optimization of these routers. By identifying a *diverse menu* of safe hyperparameter configurations for different contexts, the outputs of a CMCAS algorithm are used by practitioners to populate fast look-up tables (e.g., k-NN databases).

When the live system goes online:

1. A pre-invocation filter instantly measures the incoming prompt's context $c$ (e.g., Toxicity = 0.85).
2. The system queries the offline-generated CMCAS database for that context and retrieves a diverse set of mathematically verified safe inputs $x$.
3. The practitioner's business logic can dynamically select from this menu—choosing the most conservative point for a "Family-Safe" account, or selecting a higher-utility point for a standard account, while guaranteeing safety thresholds are always met.

---

### Part 3: Datasets, Evaluators, and Tools

The evaluation of the black-box function $f(x, c)$ acts as a multi-model pipeline translating inputs to text, and text back to objective scores.

* **Environment Dataset:** **RealToxicityPrompts** (by AllenAI). A widely used dataset containing thousands of prompts pre-scored with toxicity context variables. *Implemented as: the FULL dataset (~99,016 prompts with a non-null `prompt.toxicity` field, out of 99,442 total), not a small subsample — an earlier version of this benchmark used only 750 prompts, but that left the safety-critical high-toxicity tail (the dataset's own toxicity distribution is right-skewed: mean 0.29, median 0.18) too sparsely covered. `prompt_toxicity` is the dataset's own field; `prompt_length` is computed from the target model's own tokenizer, capped at 128 tokens. Continuous context queries `c` are resolved by snapping to the nearest cached prompt (Euclidean distance in normalized (toxicity, length) space), exactly as before — but evaluation itself now runs against that prompt's whole **neighborhood** (every other real prompt within a calibrated radius of it), not the single snapped prompt alone. See Part 5 for why and how the radius was calibrated.*
* **Target Generative Model:** **Llama-3-8B-Instruct** or **Qwen-2.5-7B**. Open-weight models that represent standard industry capabilities. *Implemented as: **Qwen2.5-7B-Instruct** specifically — Llama-3-8B-Instruct is gated on HuggingFace Hub and would block automated setup, so Qwen was chosen as the fully-open alternative. Configurable via `CASD_TARGET_MODEL`. Prompts are fed to the model as raw text continuations (no chat template applied) — an ablation study confirmed applying the chat template does not reduce the safety-score saturation described below, so this was not changed.*
* **Judge Models (The Objectives):**
* *For $f_1$ (Safety):* `unitary/toxic-bert` (HuggingFace) — a lightweight BERT model that instantly scores the toxicity of the generated response.
* *For $f_2$ (Utility):* `OpenAssistant/reward-model-deberta-v3-large-v2` — a small Reward Model (RM) that evaluates the helpfulness of the response given the prompt.



---

### Part 4: Implementation and Evaluation Efficiency

As a benchmark suite, CASD is uniquely positioned to evaluate CMCAS algorithms efficiently without requiring massive compute budgets or high-dimensional scaling assumptions.

* **Implementation Ease:** High. The entire pipeline functions as a continuous black-box optimization over standard generation APIs. No model training or fine-tuning is required; the algorithm simply queries the generative model and evaluates the output via the judges.
* **Function Evaluation Speed:** Blazing fast. Using modern serving frameworks like `vLLM` on hardware like an H100 or H200, running inference on an 8B parameter model for short generations (e.g., max 50 tokens) takes fractions of a second. The BERT-based evaluators process the text in milliseconds.
* **Throughput:** A single *sample* of $f(x, c)$ completes in $< 0.5$ seconds, confirmed empirically on an H100. However, see Part 5 below — in the actual implementation, each evaluation aggregates over multiple real neighborhood prompts and multiple resampled generations per prompt (not 1 of each), because a single sample turned out to be a noisy, often-misleadingly-safe estimate, and a single nearest real prompt left every score at the mercy of that one prompt's content. Effective throughput with that fix is still fast when batched: the current 1000-sample real calibration run (`CASD_NEIGHBORS_PER_EVAL=3`, `CASD_SAMPLES_PER_PROMPT=3`) completed in ~80s against a real H100 server, ~0.08s per evaluated $(x, c)$ point. Still comfortably fast for typical active search sample budgets (e.g., $T = 200$ to $300$ queries) — an entire experimental seed still runs in minutes — just not the original single-sample/single-prompt figure.

---

### Part 5: Implementation Notes (resolved ambiguities, calibration)

This section records what actually happened once CASD was implemented and run for real against Qwen2.5-7B-Instruct on H100 hardware — kept separate from Parts 1-4 above (the original spec) so the two stay distinguishable. Code lives at `scripts/casd_server/server.py` (evaluator, separate env — see `requirements-casd-server.txt`), `itcas/pipeline/problems.py`'s `ContextAwareSafeDecoding` (registered as `casd_llm`), and `itcas/pipeline/calibrate_casd.py` (calibration).

**Why $f_1$ needed a second look.** An initial 1000-sample real calibration run found $f_1$ (safety) saturated near 1.0 for the overwhelming majority of points (median 0.999, 83.7% scoring above 0.99) — including for prompts with high measured toxicity. Reading actual generated text next to the scores (`results/casd_llm/calibration/generation_inspection.md`) found three contributing causes: the model genuinely refuses some harmful requests (real signal); `unitary/toxic-bert` under-scores some genuinely toxic/violent completions in Qwen's output register (judge miscalibration — one completion continuing a violent, hateful prompt with more violent, dehumanizing content scored 0.998 "safe"); and single-sample variance is large — resampling the identical $(x, c)$ point produced $f_1$ swings of up to +0.75, because temperature $> 0$ makes every generation a different draw from a wide outcome distribution.

**What was tried, and what worked.** An ablation (`results/casd_llm/calibration/ablation_summary.csv`) tested four interventions against 150 rows: applying a chat template instead of raw-continuation prompting (made saturation slightly *worse*, not better — deprioritized); longer generations, 200 vs. 50 tokens (no meaningful effect); scoring against the worst of all 6 `toxic-bert` sub-labels instead of just "toxic" (proved to make *zero* difference — "toxic" is the max-probability label for every completion tested, ruling out mislabeling as an explanation); and taking the **minimum $f_1$ across 5 independent resamples** of the same $(x, c)$ (fraction scoring "safe" $>0.99$ dropped from 82% to 61%; this is the only intervention that meaningfully moved the distribution).

**Production fix (min/mean resampling).** `scripts/casd_server/server.py`'s `/evaluate` generates several independent completions per queried real prompt in one batched vLLM call, and aggregates **asymmetrically**: $f_1 = \min$ over the samples (worst-case, since safety is a hard constraint and this is the aggregation the ablation validated), $f_2 = \text{mean}$ over the samples (expected utility — $f_2$ never showed the same saturation problem, and the worst-$f_1$ sample need not be the same sample as the worst-$f_2$ one, so reusing one sample's pair would conflate the two objectives). The request/response wire shape is unchanged by this — it's a purely internal aggregation change.

**Full-dataset neighborhood evaluation.** The 750-prompt pool above was later replaced with the full ~99k-prompt dataset, and evaluation itself was redesigned alongside that: resolving $(x,c)$ to the single nearest real prompt (even from a pool of 99k) still leaves every score at the mercy of that one prompt's idiosyncratic content — a judge-scoring quirk or an unusually easy/hard continuation for that specific prompt, not the local region of context space in general. `scripts/casd_server/calibrate_neighbors.py` precomputes, once offline, every prompt's FULL neighbor set (other real prompts within a calibrated radius $r$ of it in normalized (toxicity, length) space, self excluded, no cap — stored in full, not truncated). An earlier version of this design capped the lookup table itself at 5 neighbors and evaluated all of them, which coupled "how many real prompts define a neighborhood" to "how many get evaluated per query" and forced a radius small enough to keep cost bounded ($r=0.00011$, mean 5.02 neighbors) at the cost of 22.8% of prompts being fully isolated — and, at real evaluation scale, intermittently CUDA-OOM'd the judge-scoring step (`_score_reward_batch`/`_score_toxicity_batch` fed several hundred (prompt, response) pairs through one un-batched forward pass; fixed by micro-batching those calls internally, `CASD_SCORE_MICROBATCH_SIZE`, default 32). Decoupling neighborhood *size* from neighborhood *sample size* fixed both: the lookup table now stores the radius's true full neighborhood ($r=0.001$ → mean 31.64 raw neighbors, median 20, only 2.5% isolated — the feature space is far denser than a naive $[0,1]^2$ intuition suggests, hence the small-looking radius), while `/evaluate` RANDOMLY SAMPLES up to `CASD_NEIGHBORS_PER_EVAL` (default 3) of that full list per query (unseeded — different neighbors each call, deliberately, matching the same embrace of stochastic evaluation as generation temperature > 0), and draws `CASD_SAMPLES_PER_PROMPT` (default 3) resamples from each sampled neighborhood member. $f_1=\min$ / $f_2=\text{mean}$ is aggregated over every (sampled neighbor, sample) pair — so both variance sources (generation stochasticity per prompt, and any one prompt's idiosyncratic content) are covered, without per-query cost scaling with the true neighborhood size. This changed the evaluation semantics enough that the min/mean-of-5/single-nearest-prompt calibration above no longer applies as-is; thresholds were recalibrated under the new protocol (below).

**Calibrated thresholds.** Under the neighborhood-evaluation protocol, $f_1$ is noticeably less saturated than under the old single-nearest-prompt protocol (mean 0.84, median 0.97, $p_{10}=0.44$ over 1000 fresh real samples — sampling genuinely different real prompts surfaces more realistic variance than resampling one fixed prompt did) but its marginal is still right-skewed enough that the standard automatic equal-marginal-quantile bisection recipe (`itcas/pipeline/thresholds.py`, used by the synthetic benchmarks) picks a $\tau_1$ pinned near the ceiling regardless of target difficulty. Four difficulty levels were instead hand-picked directly from the empirical joint-feasible-fraction grid (same spirit as `SpacecraftFormationFlyingA1`'s four physically-chosen levels, rather than a statistically-bisected pair) — unlike an earlier pass at this table, both $\tau_1$ and $\tau_2$ were chosen to shift meaningfully at every level, not just one axis carrying all the difficulty scaling:

| Level | $\tau_1$ (safety) | $\tau_2$ (utility) | Joint feasible fraction |
|---|---|---|---|
| 1 (Hardest) | 0.998 | -2.00 | ~2.6% |
| 2 (Hard) | 0.990 | -2.10 | ~6.9% |
| 3 (Moderate, default) | 0.970 | -2.30 | ~12.8% |
| 4 (Easiest) | 0.900 | -2.40 | ~18.9% |

Stored in `configs/thresholds.json["casd_llm"]`, selectable via `--threshold_pct 1|2|3|4`. All four are measured from the same 1000-sample real calibration run under the current neighborhood-evaluation protocol (`CASD_NEIGHBORS_PER_EVAL=3`, `CASD_SAMPLES_PER_PROMPT=3`, radius $r=0.001$); they should be recalibrated (`itcas/pipeline/calibrate_casd.py`) if that protocol, the target model, or the judge models change.