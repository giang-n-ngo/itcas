# Implementation Plan: C-MO-CAS & Baselines Benchmark Matrix

## 1. Experimental Matrix Overview
The framework will evaluate 24 algorithmic configurations. We group them into three conceptual families:

### Family A: Direct Active Search ("The Miners")
1. `Random` (Seq) / 2. `Random + DPP` (Batch)
3. `ECI` (Seq) / 4. `ECI + DPP` (Batch)
5. `MOC-CAS` (Seq) / 6. `MOC-CAS + DPP` (Batch)
7. `NDIG` [Ours] (Seq) / 8. `NDIG + DPP` [Ours] (Batch)

### Family B: Pure Level Set Estimation ("Naive Cartographers")
9. `Straddle` (Seq) / 10. `Straddle + DPP` (Batch)
11. `C2LSE` (Seq) / 12. `C2LSE + DPP` (Batch)
13. `RMILE` (Seq) / 14. `RMILE + DPP` (Batch)
15. `BES` (Seq) / 16. `BES + DPP` (Batch)

### Family C: Two-Stage LSE-then-Sample ("Hybrid Cartographers")
17. `Straddle-then-Sample` (Seq) / 18. `Straddle-then-Sample + DPP` (Batch)
19. `C2LSE-then-Sample` (Seq) / 20. `C2LSE-then-Sample + DPP` (Batch)
21. `RMILE-then-Sample` (Seq) / 22. `RMILE-then-Sample + DPP` (Batch)
23. `BES-then-Sample` (Seq) / 24. `BES-then-Sample + DPP` (Batch)


## 2. Mathematical Specifications of Novel Acquisitions
All functions below must be implemented as differentiable PyTorch modules (e.g., subclassing `botorch.acquisition.AcquisitionFunction`) to allow exact continuous optimization via L-BFGS-B or Adam. 

### 2.1 C2LSE (Confidence-based Continuous LSE)
*Source: Ngo et al. (ACML 2023)*
Queries the point where the GP is least confident about its classification. 
**Equation:**
$$a(x) = \frac{\sigma(x)}{\max(\epsilon, |\mu(x) - \tau|)}$$
*Implementation Note:* $\epsilon$ is a small positive hyperparameter (e.g., $10^{-2}$) to prevent division by zero and ensure the algorithm doesn't get permanently stuck exactly on the threshold once variance is depleted.

### 2.2 BES (Binary Entropy Search)
*Source: Nguyen et al. (AAAI 2021)*
Measures the information gain on the superlevel-set classification label. 
**Equation:**
$$\alpha_{BES}(x) = \mathbb{E}_{p(y_x|y_D)} \left[ \sum_{\gamma \in \{-1, 1\}} \Phi(\gamma g_x(y_x, \tau)) \log \frac{\Phi(\gamma g_x(y_x, \tau))}{\Phi(\gamma h_x(\tau))} \right]$$
**Where:**
* $p(y_x | y_D) = \mathcal{N}(\mu(x), \sigma^2(x) + \sigma_n^2)$
* $\sigma_+ = \sqrt{\sigma^2(x) + \sigma_n^2}$
* $h_x(\tau) = \frac{\tau - \mu(x)}{\sigma(x)}$
* $g_x(y_x, \tau) = \frac{\sigma_+^2 \tau - \sigma_n^2 \mu(x) - \sigma^2(x) y_x}{\sigma(x) \sigma_n \sigma_+}$
* $\Phi$ is the standard normal CDF.

*Implementation Note:* Because the expectation is over a 1D Gaussian $p(y_x|y_D)$, use **Gauss-Hermite Quadrature** (e.g., `torch.special.hermite_polynomial` or standard BoTorch quadrature utilities) to evaluate the integral differentiably and efficiently.

### 2.3 RMILE (Robust Maximum Improvement for Level-Set Estimation)
*Source: Zanette et al. (NeurIPS 2019)*
Maximizes the expected improvement in the volume of the superlevel set, robustified by a variance term.
**Equation:**
$$E_{GP}(x^+) = \max \left( \mathbb{E}_{y^+}[|I_{GP^+}|] - |I_{GP}^\epsilon|, \gamma \sigma(x^+) \right)$$
**Where the expected volume is estimated over a continuous domain via a reference grid/LHS $\mathcal{X}_{ref}$:**
$$\mathbb{E}_{y^+}[|I_{GP^+}|] \approx \sum_{x' \in \mathcal{X}_{ref}} \Phi \left( \frac{\sqrt{\sigma^2(x^+) + \sigma_n^2}}{|Cov(x', x^+)|} \times (\mu(x') - \beta \sigma_{GP^+}(x') - \tau) \right)$$
* $\sigma_{GP^+}^2(x') = \sigma^2(x') - \frac{Cov^2(x', x^+)}{\sigma^2(x^+) + \sigma_n^2}$
* $\beta \approx 1.96$

*Implementation Note:* Generate a static LHS of $\sim 1000$ points for $\mathcal{X}_{ref}$ at the start of each acquisition step. Use the BoTorch GP model to compute the exact predictive covariance $Cov(x', x^+)$ between the candidate $x^+$ and all points in $\mathcal{X}_{ref}$. 


## 3. Continuous Optimization Strategy

Because we are dealing with continuous input spaces, we cannot just evaluate the acquisition function on a static LHS grid. We must utilize multi-start continuous optimization (L-BFGS-B/Adam).

### Mechanism 1: Sequential Continuous
Standard active learning approach. Use the acquisition optimizer by BoTorch to find the single best candidate $x^*$ for the next query. This is the default behavior of BoTorch's `optimize_acqf` function (remember the FF problem has mixed input types).

### Mechanism 2: Batch DPP
This has already been implemented for ITCAS variants. Reuse the same code (modified if necessary).

### Mechanism 3: Stage 2 - Surrogate Interior Sampling (Continuous)
For Family C baselines when $t \ge T_{split}$ (e.g., $t \ge 0.5 T$):
* **Goal:** Sample the interior of the LSE surrogate.
* **Continuous Constraint:** We require the joint Probability of Feasibility to be $> 0.95$. 
* **Sequential Variant:** Optimize $a(x) = \sigma(x)$ (pure exploration/variance) subject to the non-linear constraint $PoF(x) \ge 0.95$. (BoTorch `optimize_acqf` supports `nonlinear_inequality_constraints`).
* **Batch Variant:** Similar to Mechanism 2, but with the additional constraint that $PoF(x) \ge 0.95$ during the L-BFGS-B optimization of the top $N$ candidates. We have to extend the initial pool to 50,000 candidates to ensure enough feasible candidates for the DPP selection step. The quality-diversity ensemble only uses the objective and context kernels because there are no acquisition values to consider in Stage 2 for these baselines.


## 4. Software Architecture Overview

Implement using a modular Strategy Pattern in Python/BoTorch:

```python
class BudgetController:
    # Determines mode: 'Search' (Stage 1) or 'Interior' (Stage 2)
    def get_mode(self, t, T, is_two_stage):
        if not is_two_stage: return 'Search'
        return 'Search' if t < (0.5 * T) else 'Interior'

class AcquisitionFactory:
    # Returns initialized BoTorch AcqF object
    def get_acqf(self, name, gp_model, thresholds):
        # e.g., if name == 'C2LSE': return C2LSE(gp_model, thresholds)
        pass

class BatchStrategy:
    # Handles continuous optimization & DPP
    def select_candidates(self, acqf, gp_model, batch_size, use_dpp, bounds, constraints=None):
        if not use_dpp:
            # Multi-start L-BFGS-B, return top 1
            return botorch.optim.optimize_acqf(...) 
        else:
            # Multi-start L-BFGS-B to get N optimized peaks
            # Run submodular maximization on the peaks
            # Return B points
            return submodular_maximization_select(...)