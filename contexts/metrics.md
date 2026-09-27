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

Data Pooling (Offline): After all algorithms have completed their runs for a specific problem, replay each individual trial's (one method + one seed) own strictly feasible objective vectors, in chronological discovery order.

Log-Transform: Transform each trial's own feasible points based on the distance from the strict threshold $\tau_i$:


$$y'_{i} = \log(1 + (y_i - \tau_i))$$

Systematic $\epsilon$ Selection (within-trial-gap anchoring — see "Revision note" below for why this replaced the original cross-trial-pool anchoring):

For each individual trial, walk its own chronological feasible sequence $Y'_{trial}$ and, for every point after the first, record its Euclidean distance to the single nearest point *already discovered by that same trial* (i.e. every point is unconditionally treated as "discovered" — no $\epsilon$ threshold applied yet, since $\epsilon$ is exactly what is being calibrated).

Pool these per-trial "nearest-already-discovered" gap values across every trial (all methods, all seeds) for the (problem, difficulty) group into a single population $G_{within\_trial}$.

Filter out all zero gaps (duplicate evaluations within a trial).

Set $\epsilon$ to the 75th percentile of $G_{within\_trial}$ (e.g., np.percentile(gaps, 75)) — a much higher percentile than the original cross-trial-pool anchoring used, because the population being percentiled is different; see "Revision note" and "Note on Percentile Tuning" below.

Archive Construction (Per Algorithm Trial): * Initialize an empty list: archive = []

Initialize a history array: archive_size = zeros(T)

For each iteration $t=1 \dots T$ in the specific algorithm's history:

Get the transformed feasible point(s) $y'$ discovered at $t$.

Calculate the Euclidean distance from $y'$ to all points currently in the archive.

If archive is empty OR $\min(\text{distances}) \ge \epsilon$:

Append $y'$ to archive.

archive_size[t] = len(archive)

Return: The strictly monotonic archive_size curve over time, and its final scalar value at $T$.

**Revision note (global-pool vs. per-trial scale mismatch):** The original version of this section pooled *all* trials' feasible objective vectors into one combined set $Y_{global\_feasible}$ and set $\epsilon$ to a low percentile (5th, per the original "Note on Percentile Tuning" below) of `scipy.spatial.distance.pdist` over that combined set. On a real dataset (`spacecraft_formation_flying_a1`, difficulty `p2`, ~560 trials across 6+ methods) this produced an $\epsilon$-Archive-Size curve that was still visually indistinguishable from the Number-of-Positives curve, even after following this section's own troubleshooting note and raising the percentile from 5th to 10th.

Diagnosis against the real dataset (not just theorized) found the root cause: the Archive Construction step above compares each new point only against *that same trial's own* archive-so-far, but the old calibration anchored $\epsilon$ against a *cross-trial* population — pairwise distances between points discovered by possibly-different methods/seeds. With ~560 independent trials covering the same feasible region (and many trials' initial designs drawn from the same historical pool), that cross-trial cloud is systematically denser than any single trial's own sequential discoveries, so a low percentile of it produced an $\epsilon$ far smaller than the typical spacing between a single trial's own real, distinct discoveries. Measured directly against real trial reconstructions: at the old calibration (5th percentile of the cross-trial pool, $\epsilon=0.396$) the median trial's final archive size equaled its final positive count exactly (ratio 1.0 — every feasible point admitted); at 10th percentile ($\epsilon=0.534$) the median ratio only improved to 0.83.

The fix: anchor $\epsilon$ against the population the archive-construction step actually compares against — each trial's own "nearest-already-discovered" gap, pooled across trials (not "any two points from possibly-different trials") — and use a much higher percentile of *that* population (see "Note on Percentile Tuning" below for why). Empirically, on the same real dataset, the 75th percentile of the within-trial-gap population produced a median archive/positives ratio of ~0.56 (mean ~0.62) with degenerate collapse (archive stuck at $\le 1$ point despite $>3$ positives) in only 0.4% of trials.

Note on Percentile Tuning: with the corrected within-trial-gap population, a *low* percentile (5th/10th/25th) reproduces the same degenerate ratio-1.0 behavior the original 5th-percentile cross-trial anchoring had — most within-trial consecutive gaps are small by construction (expected local exploration steps), so a low percentile of this population is dominated by small local steps rather than genuine trial-to-trial distinguishability. A percentile in the 70th-75th range was found empirically to be the sweet spot: high enough to separate the archive-size curve from the raw positives curve, while rarely collapsing the archive to a single degenerate point. If a specific problem's $\epsilon$-Archive-Size curve still looks too similar to the Number-of-Positives curve after using the default (75th percentile of the within-trial-gap population), try increasing further (80th-90th); if the archive instead collapses to $\le 1$ point too often, decrease toward 50th-65th.

#### **5. Number of Positives (Sample Count)**

**What it is:** A strict, cumulative count of how many evaluated $(x, c)$ pairs successfully satisfied all multi-objective constraints.

**Implementation Logic:**

1. Initialize a counter `positives = 0`.
2. For each query $(x_t, c_t)$ at iteration $t$:
* Check if $f_1(x_t, c_t) \geq \tau_1$ AND $f_2(x_t, c_t) \geq \tau_2 \dots$ AND $f_m(x_t, c_t) \geq \tau_m$.
* If True, `positives += 1`.


3. Track this cumulative sum over the timeline of iterations $t = 1 \dots T$.

#### **6. Localized Context-Conditioned Feasible Convex Hull Volume**

**What it is:** Localised Context-Conditioned Feasible Convex Hull Volume (FCHV) evaluates the objective-space diversity of an algorithm across distinct environmental regions. 
It builds upon the global FCHV, which calculates the $m$-dimensional Lebesgue measure of the convex hull of $Y_{feasible}$. 
By calculating volume within partitioned local context bins rather than pooling all discoveries into a single global set, this metric prevents an algorithm from artificially inflating its score by generating massive objective diversity in a single trivial context while leaving the broader environment space empty. 

**Implementation Logic**

Step 1: Isolate Feasible Points. 
Filter all evaluated design-context-objective tuples, $D_T = \{(x_t, c_t, y_t)\}_{t=1}^T$, to retain only strictly feasible points where $f_i(x_t, c_t) \ge \tau_i$ for all objectives $i \in \{1, \dots, m\}$.   

Step 2: Context Space Partitioning. 
Define $K$ distinct local regions within the context space $C$. 
This can be achieved dynamically using a clustering algorithm (e.g., K-means fit on the evaluated context vectors $c_t$) or statically using a predefined spatial grid.
Prefer a grid for simplicity.

Step 3: Local Assignment. 
Distribute the feasible objective vectors into local sets, $Y_{feasible, k}$, by mapping each corresponding context vector $c_t$ to its assigned bin $k \in \{1, \dots, K\}$.

Step 4: Local Volume Calculation. For each context bin $k$, compute the localized convex hull volume $V_k = Vol(Conv(Y_{feasible, k}))$. 
If a bin contains insufficient points to form a valid $m$-dimensional hull (typically requiring at least $m+1$ non-coplanar points), set $V_k = 0$.

Step 5: Aggregation. 
Compute the final scalar metric by taking the mean of the localized volumes: $FCHV_{cond} = \frac{1}{K} \sum_{k=1}^K V_k$.

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
