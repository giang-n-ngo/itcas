### Multi-Objective Coverage via Constraint Active Search (MOC-CAS) Algorithm Specification

#### 1. Problem Definition & Goal

* Unlike standard CAS which measures coverage in the parameter space, MOC-CAS aims to identify a small, diverse set of representative samples whose predicted outcomes broadly cover a feasible region in the **objective space**.

#### 2. Core Mathematical Modeling

* **Objective Functions:** Similar to CAS, MOC-CAS models the $m$ objectives using independent Gaussian processes (GPs).
* **Feasibility Estimation:** It utilizes an optimistic Upper Confidence Bound (UCB) prediction for feasibility estimations:

$$U^{(i)}_{t-1}(x) = \mu^{(i)}_{t-1}(x) + \sqrt{\beta_t}\sigma^{(i)}_{t-1}(x)$$



#### 3. Hyperparameters

* **Thresholds ($\tau$):** Defined as $\tau = [\tau_1, ..., \tau_m]$, representing the per-objective minimum thresholds that dictate the feasible region within the objective space.
* **Resolution Radius ($r$):** The coverage resolution radius, evaluated strictly in the objective space.
* **Confidence Schedule ($\beta_t$):** The multiplier controlling the UCB optimism level.
* **Softness Parameter ($\lambda > 0$):** A parameter used specifically to control the smoothness in the soft probit gate variant of the acquisition function.

#### 4. Acquisition Functions (Exact & Smooth Variants)

MOC-CAS evaluates candidates based on the optimistic estimate of the *new* feasible volume covered in the objective space. There are two distinct formulations for implementation:

* **Hard Geometric Acquisition (Exact):**
* Relies on exact set differences and hard feasibility indicators.
* It calculates the volume newly covered by the $r$-ball around the optimistic prediction $U_{t-1}(x)$, explicitly discarding regions already covered by previous outcomes $y_s$.
* **Formulation:**

$$\alpha_{\text{moc-cas}_{t-1}}(x) := Z_{t-1}(x) \cdot \text{Vol}((B_r(U_{t-1}(x)) \cap S) \setminus \cup_{s=1}^{t-1} B_r(y_s))$$




* **Soft Surrogate Acquisition (Smooth):**
* Designed for efficient, gradient-based inner maximization by relaxing the hard indicators into differentiable functions.
* The ball indicator is replaced by a unit-mass Gaussian kernel, the orthant indicator by a smooth probit gate $p_{\text{sat}}$, and the set union by a bounded soft-OR sum $n(U_{t-1}(x))$.
* **Formulation:**

$$\alpha_{\text{moc-cas}_{t-1}}(x) := V_m(r) p_{\text{sat}}(U_{t-1}(x)) n(U_{t-1}(x))$$





#### 5. Algorithmic Loop

1. **Initialize:** Initialize the dataset $\mathcal{D}_0$ with initial observations.
2. **Iterate for $t$ steps:**
* **Update Posteriors:** Update the $m$ independent GP posteriors using the current dataset $\mathcal{D}_{t-1}$.
* **Compute UCB:** Compute the optimistic UCB predictions $U_{t-1}(x)$ for all objectives.
* **Optimize Acquisition:** Maximize the chosen acquisition function (typically the soft/smooth surrogate to allow analytical gradients) to find the next optimal candidate $x_t$.
* **Apply Tie-Breaking Mechanism:** If multiple candidates yield similar acquisition scores, promote diversity by selecting the candidate with the greatest objective-space distance from prior feasible outcomes.
* **Evaluate Function:** Query the true objective functions to obtain the potentially noisy outcome $y_t = f(x_t) + \varepsilon_t$.
* **Augment Dataset:** Update the dataset $\mathcal{D}_t = \mathcal{D}_{t-1} \cup \{(x_t, y_t)\}$.



#### 6. Experimental Baselines

To benchmark the MOC-CAS algorithm, the following four baseline methods were employed in the experiments:

* **Random Search (Random):** Utilizes a uniform random selection method from the remaining candidate pool at each round.
* **One-Step Active Search (One-Step):** A myopic active search method that strictly selects the sample maximizing the predicted feasibility probability $p(Z=1 \mid \mathcal{D}_{t-1})$, typically calculated as the product of per-objective Gaussians.
* **STRADDLE:** A level-set estimation algorithm adapted to target and search the constraint boundaries of the satisfactory region.
* **MOO+Cluster:** A two-stage pipeline baseline that first performs standard Multi-Objective Optimization (MOO) to identify promising candidates, followed by a clustering algorithm to extract a diverse, representative set from the identified Pareto/feasible front.