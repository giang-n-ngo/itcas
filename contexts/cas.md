### Constraint Active Search (CAS) Algorithm Specification

#### 1. Problem Definition & Goal

* Constraint Active Search (CAS) reformulates multiobjective design as an active search problem, treating objectives as constraints governed by known threshold values.


* The primary goal is to efficiently discover the region of satisfaction and simultaneously sample diverse acceptable configurations in the parameter space, rather than merely identifying a Pareto-efficient frontier.


* Formally, the method sequentially selects design configurations $x$ from the satisfactory set $S = \{x \mid f(x) \succeq \tau\}$.



#### 2. Core Mathematical Modeling

* **Objective Functions:** The algorithm employs $m$ independent Gaussian process (GP) models to capture prior beliefs about $m$ expensive, black-box objective functions.


* **Observations:** The data are modeled with additive Gaussian noise as $y_i = f_i(x_i) + \varepsilon_i$ for $i = 1, 2, \dots, m$.


* **Feasibility Indicator:** An indicator variable $Z(x)$ determines if a given point $x$ satisfies all thresholds, defined as $Z(x) = 1 [y(x) \succeq \tau]$. The model computes the probability that a point belongs to the satisfactory region given the dataset $\mathcal{D}_t$, denoted as $p(Z(x) = 1 \mid \mathcal{D}_t)$.


#### 3. Hyperparameters

* **Thresholds ($\tau$):** Defined as $\tau = [\tau_1, \tau_2, \dots, \tau_m]^\top$, representing the desired minimum performance thresholds for all objectives.


* **Resolution Radius ($r$):** The radius defining the boundary of a coverage ball $\mathcal{N}_r(x)$ around an observation in the parameter space, which controls design precision and distinctness.


* **GP Parameters:** Hyperparameters for the chosen covariance kernel (e.g., $C_4$ Matérn length scales) and a zero mean function.



#### 4. Main Algorithm: Expected Coverage Improvement (ECI)

At iteration $t$, the policy selects the next location $x^*$ that maximizes the acquisition function $\alpha(x \mid \mathcal{D}_t)$. The main acquisition function for CAS is **Expected Coverage Improvement (ECI)**.

* ECI maximizes the expected volume of the $r$-ball around a candidate point that intersects the satisfactory region $S_Z$, specifically focusing on the volume not already covered by existing observations $X_t$.


* **Formulation:**


$$\alpha(x \mid \mathcal{D}_t) = \mathbb{E}_Z \left[ \text{Vol} \left( \{ \mathcal{N}_r(x) \cap S_Z \} \setminus \mathcal{N}_r(X_t) \right) \right]$$






#### 5. Algorithmic Loop

1. **Initialize:** Begin with an initial dataset $\mathcal{D}_t = (X_t, Y_t)$ consisting of evaluated design configurations and their respective, potentially noisy outputs.


2. **Iterate for $t$ steps:**
* **Update Posteriors:** Update the posteriors of the $m$ independent GP models $p(y \mid \mathcal{D}_t)$ using the current dataset.


* **Optimize Acquisition:** Maximize the acquisition function across the compact search space $\mathcal{X}$ to select the next candidate $x^* = \text{arg} \max_{x \in \mathcal{X}} \alpha(x \mid \mathcal{D}_t)$.


* **Evaluate Function:** Query the expensive black-box objective functions at the chosen configuration $x^*$ to obtain the new observation $y^*$.


* **Augment Dataset:** Update the dataset by taking the pairwise union of the prior data and the new observation $\mathcal{D}_t \cup (x^*, y^*)$.

#### 6. Experimental Baselines

To rigorously evaluate the ECI policy, the following baselines were utilized in the experiments:

* **Random Search (RND):** A standard random uniform sampling approach.


* **$\varepsilon$-constraint BO:** A Bayesian Optimization baseline extended to handle threshold constraints.


* **STRADDLE:** A level set estimation heuristic, adapted to the multiobjective setting by alternating the objective functions at each iteration.


* **One-step Active Search (ONE-S):** A myopic algorithm that greedily maximizes the candidate point with the highest probability of constraint satisfaction $p(Z(x) = 1 \mid \mathcal{D}_t)$ at each iteration.


* **Mutual Information (EZ):** An information-theoretic baseline that evaluates the mutual information between $y$ and $Z$, simplifying to the entropy of $Z$, denoted as $H(Z)$.


* **Expected Information Safe Region (EISR):** Another information-theoretic baseline that rewards sampling configurations inside the satisfactory region $S$ that possess high entropy, formulated as $\alpha_{\text{EISR}} = p(Z(x) = 1) H(y)$.