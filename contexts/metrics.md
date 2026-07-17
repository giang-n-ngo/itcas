### **Contextual Multi-Objective Constraint Active Search (C-MO-CAS) - Evaluation Metrics Specification**

**General Notation:**

* $\mathcal{X}$: Design space.
* $\mathcal{C}$: Context (environment) space.
* $T$: Total evaluation budget (number of queries).
* $f_i(x, c)$: The $i$-th objective function evaluated at design $x$ and context $c$.
* $\tau_i$: The strict feasibility threshold for the $i$-th objective.
* **Feasible Sample**: A sample $(x, c)$ is deemed "positive" or "feasible" if $f_i(x, c) \geq \tau_i$ for all objectives $i \in \{1, \dots, m\}$.

---

#### **1. Context Fill Distance (Contextual Diversity)**

**What it is:** Measures how evenly the actively queried contexts are distributed across the entire environment space, ensuring the algorithm doesn't just exploit "easy" initial states.
**Mathematical Definition:** The radius of the largest empty sphere within the context space $\mathcal{C}$.


$$\text{CFD} = \max_{c \in \mathcal{C}} \min_{c_t \in \mathcal{C}_{evaluated}} ||c - c_t||_2$$


**Implementation Logic:**

1. Generate a massive, dense grid or Latin Hypercube Sample (LHS) of reference points $C_{ref}$ covering the bounds of the context space $\mathcal{C}$.
2. Extract the list of all contexts evaluated by the algorithm, $C_{evaluated}$.
3. For each reference point $c \in C_{ref}$, calculate the Euclidean distance to its *nearest neighbor* in $C_{evaluated}$.
4. Find the maximum of these minimum distances.
*Note: A lower CFD indicates better, more uniform exploration of the context space.*

#### **2. Feasible Context Fill Distance**

**What it is:** Measures how evenly the *successfully solved* environments are distributed across the entire context space. Unlike standard Context Fill Distance (which measures where the algorithm simply *looked*), this metric strictly evaluates where the algorithm actually *succeeded* in finding feasible designs.
**Mathematical Definition:** The radius of the largest empty sphere within the context space $\mathcal{C}$ relative to the set of contexts where feasible designs were discovered.


$$\text{FCFD} = \max_{c \in \mathcal{C}} \min_{c_k \in \mathcal{C}_{feasible}} ||c - c_k||_2$$


**Implementation Logic:**

1. Generate a massive, dense grid or Latin Hypercube Sample (LHS) of reference points $C_{ref}$ covering the bounds of the context space $\mathcal{C}$.
2. Filter the algorithm's evaluated history to identify only the "positive" queries where the pair $(x, c)$ strictly satisfied all multi-objective thresholds ($f_i(x, c) \geq \tau_i \ \forall i$).
3. Extract the unique contexts from these successful evaluations to form the set $C_{feasible}$.
4. *Edge case handling:* If $C_{feasible}$ is empty (no positives found yet), return a maximum penalty distance (e.g., the diagonal distance of the context space bounds).
5. For each reference point $c \in C_{ref}$, calculate the Euclidean distance to its *nearest neighbor* in the successful set $C_{feasible}$.
6. Return the maximum of these minimum distances.
*Note: A lower value indicates that the algorithm has successfully mapped out valid solutions across a highly diverse and evenly spread set of operating environments.

#### **3. Feasible Convex Hull Volume (Macro-Spread)**

**What it is:** Instead of measuring empty space against an unknown boundary, this measures the total geometric volume encapsulated by your discovered feasible objective vectors. A larger volume indicates that the algorithm has pushed further outward in multiple conflicting directions, discovering a wider variety of valid performance trade-offs.
**Mathematical Definition:** The $m$-dimensional Lebesgue measure (volume) of the convex hull formed by the set of feasible objective vectors $Y_{feasible}$.


$$\text{FCHV} = \text{Volume}(\text{Conv}(Y_{feasible}))$$


**Implementation Logic:**

1. Extract the set of feasible objective vectors successfully discovered by the algorithm, $Y_{feasible}$.
2. Normalize the vectors.
3. *Edge Case Handling:* If the number of discovered points is less than $m + 1$ (where $m$ is the number of objectives), the volume is 0.
4. Use a computational geometry library (e.g., `scipy.spatial.ConvexHull` in Python) on $Y_{feasible}$.
5. Return the `.volume` property of the resulting hull.
*Note: A higher Convex Hull Volume indicates a broader overall discovery of the feasible performance space.*

#### **4. Scale-Invariant $\epsilon$-Archive Size (Gridless Micro-Diversity)**

**What it is:** Evaluates how many distinct, non-redundant feasible solutions the algorithm has discovered. To handle unknown synthetic scales and prevent outlier-induced collapse, the distinguishability threshold ($\epsilon$) is determined dynamically from the empirical distribution of the pooled data using percentile anchoring.

**Implementation Logic:**

Data Pooling (Offline): After all algorithms have completed their runs for a specific problem, collect the set of all strictly feasible objective vectors discovered by all methods across all seeds into a single global set, $Y_{global\_feasible}$.

Log-Transform: Transform the global set based on the distance from the strict threshold $\tau_i$:


$$y'_{i} = \log(1 + (y_i - \tau_i))$$

Systematic $\epsilon$ Selection:

Compute the pairwise Euclidean distance matrix for all points in the transformed global set $Y'_{global\_feasible}$ (e.g., using scipy.spatial.distance.pdist).

Filter out all zero distances (distances between a point and itself, or exact duplicate points).

Set $\epsilon$ to the 5th percentile of these strictly positive pairwise distances (e.g., np.percentile(distances, 5)).

Archive Construction (Per Algorithm Trial): * Initialize an empty list: archive = []

Initialize a history array: archive_size = zeros(T)

For each iteration $t=1 \dots T$ in the specific algorithm's history:

Get the transformed feasible point(s) $y'$ discovered at $t$.

Calculate the Euclidean distance from $y'$ to all points currently in the archive.

If archive is empty OR $\min(\text{distances}) \ge \epsilon$:

Append $y'$ to archive.

archive_size[t] = len(archive)

Return: The strictly monotonic archive_size curve over time, and its final scalar value at $T$.

Note on Percentile Tuning: The 5th percentile represents a strict requirement that a new point must be further away than the closest 5% of all points ever discovered to be considered "novel". If the metric still looks too similar to the 'Number of Positives' curve, increase this to the 10th or 15th percentile to demand wider spacing.

#### **5. Number of Positives (Sample Count)**

**What it is:** A strict, cumulative count of how many evaluated $(x, c)$ pairs successfully satisfied all multi-objective constraints.
**Implementation Logic:**

1. Initialize a counter `positives = 0`.
2. For each query $(x_t, c_t)$ at iteration $t$:
* Check if $f_1(x_t, c_t) \geq \tau_1$ AND $f_2(x_t, c_t) \geq \tau_2 \dots$ AND $f_m(x_t, c_t) \geq \tau_m$.
* If True, `positives += 1`.


3. Track this cumulative sum over the timeline of iterations $t = 1 \dots T$.

---

### System and Data Overview

The goal is to statistically compare the performance of different methods given a specific combination of problem and difficulty. The evaluation framework is defined by the following characteristics:

* **Metric Summarization:** Performance metrics are represented by the areas under or above their respective curves.


* **Hypervolume Calculation:** These areas are multiplied together to form a single hypervolume value.


* **Data Structure:** Each hypervolume value corresponds to a specific (problem, difficulty, seed, method) combination.


* **Paired Data:** Because the exact same 20 seeds are shared across the problems, a (problem, difficulty, method) combination is represented by a paired array of 20 numbers. The performance scores are dependent.


* **Distribution:** Hypervolume values are strictly positive, highly skewed, and not normally distributed.

---

## Statistical Testing Pipeline

Because the data is paired and non-normally distributed , traditional parametric tests are inappropriate without data transformations. The following non-parametric pipeline should be implemented:

### Step 1: The Omnibus Test (Friedman Test)

Before conducting pairwise comparisons, test whether there is any statistically significant difference among all methods for a specific (problem, difficulty) pair.

* **Test:** Friedman Test (the non-parametric equivalent of a repeated-measures ANOVA).


* **Input:** A $20 \times K$ matrix, where 20 is the number of seeds and $K$ is the number of methods.


* **Null Hypothesis:** $H_0$ assumes that all methods perform equally well.


* **Condition:** If the $p$-value is less than the chosen significance level (e.g., $\alpha=0.05$), reject $H_0$ and proceed to Step 2. Otherwise, conclude there is no statistical difference.

### Step 2: Post-Hoc Pairwise Testing (Wilcoxon Signed-Rank Test)

If the omnibus test indicates a significant difference, evaluate whether the proposed method statistically dominates the baselines.

* **Test:** Wilcoxon Signed-Rank Test (the non-parametric equivalent of a paired t-test).


* **Directionality:** Use a one-sided (greater) alternative hypothesis to prove strictly greater hypervolume dominance.


* **Input:** Two paired arrays of 20 numbers (e.g., Proposed Method vs. Baseline A).



### Step 3: Multiple Testing Correction (Holm-Bonferroni)

Because multiple pairwise comparisons will inflate the chance of a Type I error (false positive), the $p$-values must be adjusted.

* **Method:** Apply the Holm-Bonferroni correction (or Benjamini-Hochberg FDR if scaling to many methods). This is strictly more powerful than a standard Bonferroni correction.
