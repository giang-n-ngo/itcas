## **Implementation Specification: C-MO-CAS Synthetic Benchmark Suite**

**Note to Implementer:**
Please implement the following four synthetic benchmarks in Python (e.g., using `numpy`, `torch`, or `botorch` conventions depending on the testbed). These benchmarks evaluate a Contextual Multi-Objective Constraint Active Search (C-MO-CAS) algorithm.

**Core Problem Formulation:**

* The algorithm attempts to find a context-dependent feasible region $S(c)$ where multiple unknown functions strictly satisfy a lower threshold: $f_i(\mathbf{x}, \mathbf{w}) \ge \tau_i$.
* Because standard optimization functions are typically minimized, **all functions defined below have been mathematically negated** to fit this superlevel set (maximization) constraint.
* Let $\mathbf{x} \in \mathbb{R}^{d_x}$ be the design variables and $\mathbf{w} \in \mathbb{R}^{d_w}$ be the context variables. Let $\mathbf{z} = [\mathbf{x}, \mathbf{w}] \in \mathbb{R}^D$ be the concatenated joint input space, where $D = d_x + d_w$.

### **Benchmark 1: 2-Objective Shifted Sphere (Low-D)**

* **Dimensionality:** $D = 6$ ($d_x = 3$, $d_w = 3$).
* **Recommended Domain bounds:** $[-5.0, 5.0]^6$
* **Description:** A baseline convex landscape to verify basic convergence and hypervolume expansion. The first objective centers at the origin, and the second objective is shifted uniformly by $c=2.0$.
* **Equations:**

$$f_1(\mathbf{z}) = - \sum_{i=1}^{6} z_i^2$$


$$f_2(\mathbf{z}) = - \sum_{i=1}^{6} (z_i - 2.0)^2$$



### **Benchmark 2: 2-Objective Contextual Rosenbrock vs. Shifted Sphere (Low-D)**

* **Dimensionality:** $D = 6$ ($d_x = 4$, $d_w = 2$).
* **Recommended Domain bounds:** $[-2.0, 2.0]^6$
* **Description:** Tests non-linear correlations between the context and design variables. The context variable $w_1$ explicitly bends the optimal manifold of the design variable $x_1$ in the Rosenbrock valley. This is paired against a shifted sphere to force a multi-objective trade-off.
* **Equations:**

$$f_1(\mathbf{x}, \mathbf{w}) = - \left[ 100(x_1 - w_1^2)^2 + (1-w_1)^2 + \sum_{i=1}^{3} \left( 100(x_{i+1} - x_i^2)^2 + (1-x_i)^2 \right) + w_2^2 \right]$$


$$f_2(\mathbf{x}, \mathbf{w}) = - \sum_{i=1}^{4} (x_i + 2.0)^2 - \sum_{j=1}^{2} (w_j + 2.0)^2$$



### **Benchmark 3: 3-Objective Multi-Modal Trap (High-D)**

* **Dimensionality:** $D = 20$ ($d_x = 10$, $d_w = 10$).
* **Recommended Domain bounds:** $[-5.0, 5.0]^{20}$
* **Description:** A rugged, high-dimensional stress test combining Ackley, a shifted Rastrigin, and a shifted Sphere function. Evaluates the acquisition function's robustness against local optima and high-dimensional volume scaling.
* **Equations:**

$$f_1(\mathbf{z}) = - \left[ -20 \exp\left(-0.2 \sqrt{\frac{1}{20} \sum_{i=1}^{20} z_i^2}\right) - \exp\left(\frac{1}{20} \sum_{i=1}^{20} \cos(2\pi z_i)\right) + 20 + \exp(1) \right]$$


$$f_2(\mathbf{z}) = - \left[ 200 + \sum_{i=1}^{20} \left( (z_i - 2.0)^2 - 10\cos(2\pi (z_i - 2.0)) \right) \right]$$


$$f_3(\mathbf{z}) = - \sum_{i=1}^{20} (z_i + 2.0)^2$$



### **Benchmark 4: 4-Objective Contextual DTLZ2 (Low-D)**

* **Dimensionality:** $D = 6$ ($d_x = 3$, $d_w = 3$).
* **Recommended Domain bounds:** $[0.0, 1.0]^6$
* **Description:** An adaptation of the standard scalable DTLZ2 benchmark. The design variables $\mathbf{x}$ act as the position parameters mapping the Pareto front, while the context variables $\mathbf{w}$ control the distance function $g(\mathbf{w})$, directly dictating the difficulty of achieving feasibility.
* **Equations:**

$$g(\mathbf{w}) = \sum_{j=1}^{3} (w_j - 0.5)^2$$


$$f_1(\mathbf{x}, \mathbf{w}) = -(1 + g(\mathbf{w})) \cos\left(x_1 \frac{\pi}{2}\right) \cos\left(x_2 \frac{\pi}{2}\right) \cos\left(x_3 \frac{\pi}{2}\right)$$


$$f_2(\mathbf{x}, \mathbf{w}) = -(1 + g(\mathbf{w})) \cos\left(x_1 \frac{\pi}{2}\right) \cos\left(x_2 \frac{\pi}{2}\right) \sin\left(x_3 \frac{\pi}{2}\right)$$


$$f_3(\mathbf{x}, \mathbf{w}) = -(1 + g(\mathbf{w})) \cos\left(x_1 \frac{\pi}{2}\right) \sin\left(x_2 \frac{\pi}{2}\right)$$


$$f_4(\mathbf{x}, \mathbf{w}) = -(1 + g(\mathbf{w})) \sin\left(x_1 \frac{\pi}{2}\right)$$

### **Benchmark 5: 2-Objective Shifted Styblinski-Tang (Low-D)**

* **Dimensionality:** $D = 6$ ($d_x = 3$, $d_w = 3$).
* **Recommended Domain bounds:** $[-5.0, 5.0]^6$
* **Description:** A highly non-convex, independent multi-modal trap. Unlike Ackley, which has one deep global funnel, the Styblinski-Tang function has multiple equally compelling local "pockets." Shifting this function guarantees that moving toward one objective's global optimum drags the algorithm through deep, deceptive local traps of the opposing objective.
* **Equations:**

$$f_1(\mathbf{z}) = - \frac{1}{2} \sum_{i=1}^{6} \left( z_i^4 - 16z_i^2 + 5z_i \right)$$


$$f_2(\mathbf{z}) = - \frac{1}{2} \sum_{i=1}^{6} \left( (z_i - 2.0)^4 - 16(z_i - 2.0)^2 + 5(z_i - 2.0) \right)$$



### **Benchmark 6: 3-Objective Contextual DTLZ7 (Low-D)**

* **Dimensionality:** $D = 6$ ($d_x = 2$, $d_w = 4$).
* **Recommended Domain bounds:** $[0.0, 1.0]^6$
* **Description:** DTLZ7 is famous for having a **disconnected and discontinuous** Pareto front. This is a brutal test for C-MO-CAS's diversity mechanism. Instead of finding one continuous feasible manifold, the algorithm must simultaneously discover and maintain coverage over multiple isolated "islands" of feasibility across the context space.
* **Equations:**

$$g(\mathbf{w}) = 1 + \frac{9}{4} \sum_{j=1}^{4} w_j$$


$$f_1(\mathbf{x}, \mathbf{w}) = - x_1$$


$$f_2(\mathbf{x}, \mathbf{w}) = - x_2$$


$$f_3(\mathbf{x}, \mathbf{w}) = - (1 + g(\mathbf{w})) \left[ 3 - \sum_{i=1}^{2} \left( \frac{x_i}{1 + g(\mathbf{w})} \left(1 + \sin(3\pi x_i)\right) \right) \right]$$



### **Benchmark 7: 2-Objective Ill-Conditioned Ellipsoid (High-D)**

* **Dimensionality:** $D = 20$ ($d_x = 10$, $d_w = 10$).
* **Recommended Domain bounds:** $[-5.12, 5.12]^{20}$
* **Description:** A critical diagnostic test for the Gaussian Process ARD (Automatic Relevance Determination) kernel in high dimensions. The Ellipsoid function applies an exponentially increasing weight to each dimension. This tests whether the active search acquisition function can ignore "flat" environmental dimensions and correctly allocate sample budgets to the highly sensitive dimensions.
* **Equations:**

$$f_1(\mathbf{z}) = - \sum_{i=1}^{20} 1000^{\frac{i-1}{19}} z_i^2$$


$$f_2(\mathbf{z}) = - \sum_{i=1}^{20} 1000^{\frac{i-1}{19}} (z_i - 2.0)^2$$



### **Benchmark 8: 4-Objective Contextual DTLZ1 (High-D)**

* **Dimensionality:** $D = 12$ ($d_x = 3$, $d_w = 9$).
* **Recommended Domain bounds:** $[0.0, 1.0]^{12}$
* **Description:** DTLZ1 uses a highly multi-modal context/distance function $g(\mathbf{w})$ based on the Rastrigin topology. It creates literally thousands of "local" Pareto fronts. It evaluates if the C-MO-CAS acquisition function gets trapped mapping out a sub-optimal feasible region, or if it can push through the environmental noise to find the true global feasible boundary in a heavily scaled 4-objective space.
* **Equations:**

$$g(\mathbf{w}) = 100 \left[ 9 + \sum_{j=1}^{9} \left( (w_j - 0.5)^2 - \cos(20\pi(w_j - 0.5)) \right) \right]$$


$$f_1(\mathbf{x}, \mathbf{w}) = - \frac{1}{2} x_1 x_2 x_3 (1 + g(\mathbf{w}))$$


$$f_2(\mathbf{x}, \mathbf{w}) = - \frac{1}{2} x_1 x_2 (1 - x_3) (1 + g(\mathbf{w}))$$


$$f_3(\mathbf{x}, \mathbf{w}) = - \frac{1}{2} x_1 (1 - x_2) (1 + g(\mathbf{w}))$$


$$f_4(\mathbf{x}, \mathbf{w}) = - \frac{1}{2} (1 - x_1) (1 + g(\mathbf{w}))$$

### **Benchmark 9: 2-Objective Contextual ZDT3 (Low-D)**

* **Dimensionality:** $D = 6$ ($d_x = 1$, $d_w = 5$).
* **Recommended Domain bounds:** $[0.0, 1.0]^6$
* **Description:** ZDT3 is the gold standard for testing an algorithm's ability to handle **disconnected Pareto fronts**. The feasible region is not a single continuous manifold but rather multiple disjoint islands in the objective space.
* **Equations:**

$$g(\mathbf{w}) = 1 + \frac{9}{5} \sum_{j=1}^{5} w_j$$


$$f_1(x_1, \mathbf{w}) = - x_1$$


$$f_2(x_1, \mathbf{w}) = - g(\mathbf{w}) \left[ 1 - \sqrt{\frac{x_1}{g(\mathbf{w})}} - \left(\frac{x_1}{g(\mathbf{w})}\right) \sin(10\pi x_1) \right]$$



### **Benchmark 10: 3-Objective Shifted Levy (High-D)**

* **Dimensionality:** $D = 16$ ($d_x = 8$, $d_w = 8$).
* **Recommended Domain bounds:** $[-10.0, 10.0]^{16}$
* **Description:** The Levy function is highly rugged with massive local optimum traps. By shifting three variants against each other in 16 dimensions, this tests the algorithm's ability to resolve competing, highly complex multimodal landscapes simultaneously.
* **Helper Variable:** $v_i(\mathbf{z}) = 1 + \frac{z_i - 1}{4}$
* **Base Levy Function:** $L(\mathbf{v}) = \sin^2(\pi v_1) + \sum_{i=1}^{15} (v_i - 1)^2 [1 + 10 \sin^2(\pi v_i + 1)] + (v_{16} - 1)^2 [1 + \sin^2(2\pi v_{16})]$
* **Equations:**

$$f_1(\mathbf{z}) = - L(\mathbf{v}(\mathbf{z}))$$


$$f_2(\mathbf{z}) = - L(\mathbf{v}(\mathbf{z} - 2.0))$$


$$f_3(\mathbf{z}) = - L(\mathbf{v}(\mathbf{z} + 2.0))$$



### **Benchmark 11: 4-Objective Contextual DTLZ3 (Low-D)**

* **Dimensionality:** $D = 8$ ($d_x = 3$, $d_w = 5$).
* **Recommended Domain bounds:** $[0.0, 1.0]^8$
* **Description:** DTLZ3 uses a massive Rastrigin-based distance function $g(\mathbf{w})$. This creates extreme "environmental noise"—thousands of local, sub-optimal Pareto fronts that act as traps before the algorithm can reach the global feasible boundary.
* **Equations:**

$$g(\mathbf{w}) = 100 \left[ 5 + \sum_{j=1}^{5} \left( (w_j - 0.5)^2 - \cos(20\pi(w_j - 0.5)) \right) \right]$$


$$f_1(\mathbf{x}, \mathbf{w}) = - (1 + g(\mathbf{w})) \cos\left(x_1 \frac{\pi}{2}\right) \cos\left(x_2 \frac{\pi}{2}\right) \cos\left(x_3 \frac{\pi}{2}\right)$$


$$f_2(\mathbf{x}, \mathbf{w}) = - (1 + g(\mathbf{w})) \cos\left(x_1 \frac{\pi}{2}\right) \cos\left(x_2 \frac{\pi}{2}\right) \sin\left(x_3 \frac{\pi}{2}\right)$$


$$f_3(\mathbf{x}, \mathbf{w}) = - (1 + g(\mathbf{w})) \cos\left(x_1 \frac{\pi}{2}\right) \sin\left(x_2 \frac{\pi}{2}\right)$$


$$f_4(\mathbf{x}, \mathbf{w}) = - (1 + g(\mathbf{w})) \sin\left(x_1 \frac{\pi}{2}\right)$$



### **Benchmark 12: 2-Objective Exponential VLMOP2 (Low-D)**

* **Dimensionality:** $D = 6$ ($d_x = 3$, $d_w = 3$).
* **Recommended Domain bounds:** $[-2.0, 2.0]^6$
* **Description:** Tests the Gaussian Process's ability to handle **vanishing gradients**. The VLMOP2 topology is completely flat almost everywhere except in the immediate vicinity of the optima, starving the acquisition function of directional information during exploration.
* **Equations:**

$$f_1(\mathbf{z}) = - \left[ 1 - \exp\left(-\sum_{i=1}^{6} \left(z_i - \frac{1}{\sqrt{6}}\right)^2\right) \right]$$


$$f_2(\mathbf{z}) = - \left[ 1 - \exp\left(-\sum_{i=1}^{6} \left(z_i + \frac{1}{\sqrt{6}}\right)^2\right) \right]$$



### **Benchmark 13: 3-Objective Contextual DTLZ4 (High-D)**

* **Dimensionality:** $D = 12$ ($d_x = 2$, $d_w = 10$).
* **Recommended Domain bounds:** $[0.0, 1.0]^{12}$
* **Description:** DTLZ4 modifies the design mapping with a high power $\alpha = 100$. This creates a highly **biased density**, clustering the true Pareto front mapping into a tiny, skewed geometric corner of the design space. It tests if the active search operates purely on geometry or true uncertainty.
* **Equations:**

$$g(\mathbf{w}) = \sum_{j=1}^{10} (w_j - 0.5)^2$$


$$f_1(\mathbf{x}, \mathbf{w}) = - (1 + g(\mathbf{w})) \cos\left(x_1^{100} \frac{\pi}{2}\right) \cos\left(x_2^{100} \frac{\pi}{2}\right)$$


$$f_2(\mathbf{x}, \mathbf{w}) = - (1 + g(\mathbf{w})) \cos\left(x_1^{100} \frac{\pi}{2}\right) \sin\left(x_2^{100} \frac{\pi}{2}\right)$$


$$f_3(\mathbf{x}, \mathbf{w}) = - (1 + g(\mathbf{w})) \sin\left(x_1^{100} \frac{\pi}{2}\right)$$



### **Benchmark 14: 2-Objective Contextual Dixon-Price vs. Sphere (Medium-D)**

* **Dimensionality:** $D = 10$ ($d_x = 5$, $d_w = 5$).
* **Recommended Domain bounds:** $[-10.0, 10.0]^{10}$
* **Description:** The Dixon-Price function is notoriously ill-conditioned, forming a very deep, narrow, sweeping valley. Satisfying this benchmark requires tracking a narrow constraint ridge against a conflicting smooth sphere.
* **Equations:**

$$f_1(\mathbf{z}) = - \left[ (z_1 - 1)^2 + \sum_{i=2}^{10} i (2 z_i^2 - z_{i-1})^2 \right]$$


$$f_2(\mathbf{z}) = - \sum_{i=1}^{10} (z_i - 2.0)^2$$



### **Benchmark 15: 4-Objective Shifted Griewank (High-D)**

* **Dimensionality:** $D = 16$ ($d_x = 8$, $d_w = 8$).
* **Recommended Domain bounds:** $[-10.0, 10.0]^{16}$
* **Description:** Griewank has a macro-convex structure but is micro-rugged everywhere due to product-cosine interference. This 4-objective setup shifts the variables into four conflicting directional quadrants.
* **Base Function:** $G(\mathbf{v}) = 1 + \sum_{i=1}^{16} \frac{v_i^2}{4000} - \prod_{i=1}^{16} \cos\left(\frac{v_i}{\sqrt{i}}\right)$
* **Equations:** *(Note: Let $\mathbf{z}_{[:8]}$ be the first 8 dims, and $\mathbf{z}_{[8:]}$ be the last 8)*

$$f_1(\mathbf{z}) = - G(\mathbf{z} - 2.0)$$


$$f_2(\mathbf{z}) = - G(\mathbf{z} + 2.0)$$


$$f_3(\mathbf{z}) = - G([\mathbf{z}_{[:8]} - 2.0, \; \mathbf{z}_{[8:]} + 2.0])$$


$$f_4(\mathbf{z}) = - G([\mathbf{z}_{[:8]} + 2.0, \; \mathbf{z}_{[8:]} - 2.0])$$



### **Benchmark 16: 3-Objective Shifted Alpine N.1 (Medium-D)**

* **Dimensionality:** $D = 12$ ($d_x = 6$, $d_w = 6$).
* **Recommended Domain bounds:** $[-10.0, 10.0]^{12}$
* **Description:** The Alpine function introduces absolute value operations, making the derivatives at the local minima **non-differentiable**. This tests the robustness of the acquisition function optimizer (e.g., L-BFGS-B or CMA-ES) when handling sharp, non-smooth landscape kinks.
* **Base Function:** $A(\mathbf{v}) = \sum_{i=1}^{12} \left| v_i \sin(v_i) + 0.1 v_i \right|$
* **Equations:**

$$f_1(\mathbf{z}) = - A(\mathbf{z})$$


$$f_2(\mathbf{z}) = - A(\mathbf{z} - 2.0)$$


$$f_3(\mathbf{z}) = - A(\mathbf{z} + 2.0)$$


### **Benchmark 17: 2-Objective Ackley vs. Rosenbrock (Low-D)**

* **Dimensionality:** $D = 6$ ($d_x = 3$, $d_w = 3$).
* **Recommended Domain bounds:** $[-2.0, 2.0]^6$
* **Description:** Pits a central, symmetric, multi-modal funnel (Ackley) directly against an asymmetric, flat, banana-shaped continuous valley (Rosenbrock).
* **Equations:**

$$f_1(\mathbf{z}) = - \left[ -20 \exp\left(-0.2 \sqrt{\frac{1}{6} \sum_{i=1}^{6} z_i^2}\right) - \exp\left(\frac{1}{6} \sum_{i=1}^{6} \cos(2\pi z_i)\right) + 20 + \exp(1) \right]$$


$$f_2(\mathbf{z}) = - \sum_{i=1}^{5} \left[ 100(z_{i+1} - z_i^2)^2 + (1 - z_i)^2 \right]$$



### **Benchmark 18: 3-Objective Rastrigin vs. Griewank vs. Sphere (High-D)**

* **Dimensionality:** $D = 20$ ($d_x = 10$, $d_w = 10$).
* **Recommended Domain bounds:** $[-5.0, 5.0]^{20}$
* **Description:** A brutal high-dimensional test combining three different macroscopic structures. Rastrigin introduces massive independent local optima, Griewank introduces micro-rugged cosine interference over a convex macro-structure, and Sphere acts as a smooth, conflicting anchor.
* **Equations:**

$$f_1(\mathbf{z}) = - \left[ 200 + \sum_{i=1}^{20} \left( z_i^2 - 10\cos(2\pi z_i) \right) \right]$$


$$f_2(\mathbf{z}) = - \left[ 1 + \sum_{i=1}^{20} \frac{(z_i - 2.0)^2}{4000} - \prod_{i=1}^{20} \cos\left(\frac{z_i - 2.0}{\sqrt{i}}\right) \right]$$


$$f_3(\mathbf{z}) = - \sum_{i=1}^{20} (z_i + 2.0)^2$$



### **Benchmark 19: 2-Objective Styblinski-Tang vs. Levy (Medium-D)**

* **Dimensionality:** $D = 10$ ($d_x = 5$, $d_w = 5$).
* **Recommended Domain bounds:** $[-5.0, 5.0]^{10}$
* **Description:** Tests the algorithm's capability to escape two completely different styles of deceptive local minimums. Levy uses a nested polynomial-trigonometric trap, while Styblinski-Tang features massive, disjoint polynomial basins.
* **Equations:** Let $v_i(\mathbf{z}) = 1 + \frac{z_i - 1}{4}$

$$f_1(\mathbf{z}) = - \frac{1}{2} \sum_{i=1}^{10} \left( z_i^4 - 16z_i^2 + 5z_i \right)$$


$$f_2(\mathbf{z}) = - \left[ \sin^2(\pi v_1) + \sum_{i=1}^{9} (v_i - 1)^2 [1 + 10 \sin^2(\pi v_i + 1)] + (v_{10} - 1)^2 [1 + \sin^2(2\pi v_{10})] \right]$$



### **Benchmark 20: 4-Objective Heterogeneous Quadrants (Low-D)**

* **Dimensionality:** $D = 6$ ($d_x = 3$, $d_w = 3$).
* **Recommended Domain bounds:** $[-5.0, 5.0]^6$
* **Description:** Shifts four entirely different landscape topologies into four conflicting directional quadrants. The feasible region forms at the highly asymmetric intersection of a Sphere, an Ackley funnel, a Rastrigin grid, and a Griewank bowl.
* **Helper Variables:** Let $\mathbf{z}_{A} = \mathbf{z} - 2.0$
Let $\mathbf{z}_{B} = \mathbf{z} + 2.0$
Let $\mathbf{z}_{C} = [z_1-2.0, z_2-2.0, z_3-2.0, z_4+2.0, z_5+2.0, z_6+2.0]$
Let $\mathbf{z}_{D} = [z_1+2.0, z_2+2.0, z_3+2.0, z_4-2.0, z_5-2.0, z_6-2.0]$
* **Equations:** 
$$f_1(\mathbf{z}) = - \sum_{i=1}^{6} (\mathbf{z}_{A})_i^2$$


$$f_2(\mathbf{z}) = - \left[ -20 \exp\left(-0.2 \sqrt{\frac{1}{6} \sum_{i=1}^{6} (\mathbf{z}_{B})_i^2}\right) - \exp\left(\frac{1}{6} \sum_{i=1}^{6} \cos(2\pi (\mathbf{z}_{B})_i)\right) + 20 + \exp(1) \right]$$


$$f_3(\mathbf{z}) = - \left[ 60 + \sum_{i=1}^{6} \left( (\mathbf{z}_{C})_i^2 - 10\cos(2\pi (\mathbf{z}_{C})_i) \right) \right]$$


$$f_4(\mathbf{z}) = - \left[ 1 + \sum_{i=1}^{6} \frac{(\mathbf{z}_{D})_i^2}{4000} - \prod_{i=1}^{6} \cos\left(\frac{(\mathbf{z}_{D})_i}{\sqrt{i}}\right) \right]$$



### **Benchmark 21: 2-Objective Ellipsoid vs. Rastrigin (High-D)**

* **Dimensionality:** $D = 20$ ($d_x = 10$, $d_w = 10$).
* **Recommended Domain bounds:** $[-5.12, 5.12]^{20}$
* **Description:** A critical stress test for the Automatic Relevance Determination (ARD) kernel. Objective 1 (Ellipsoid) has heavily ill-conditioned scaling where some dimensions barely matter. Objective 2 (Rastrigin) applies equal, heavy multi-modal variance to all dimensions.
* **Equations:**

$$f_1(\mathbf{z}) = - \sum_{i=1}^{20} 1000^{\frac{i-1}{19}} z_i^2$$


$$f_2(\mathbf{z}) = - \left[ 200 + \sum_{i=1}^{20} \left( (z_i - 2.0)^2 - 10\cos(2\pi (z_i - 2.0)) \right) \right]$$



### **Benchmark 22: 3-Objective Dixon-Price vs. Rosenbrock vs. Sphere (Medium-D)**

* **Dimensionality:** $D = 12$ ($d_x = 6$, $d_w = 6$).
* **Recommended Domain bounds:** $[-5.0, 5.0]^{12}$
* **Description:** Combines two notoriously narrow, curving valleys (Dixon-Price and Rosenbrock) with different polynomial scalings, pitted against a smooth Sphere. Finding the feasible intersection requires hyper-precise gradient tracking.
* **Equations:**

$$f_1(\mathbf{z}) = - \left[ (z_1 - 1)^2 + \sum_{i=2}^{12} i (2 z_i^2 - z_{i-1})^2 \right]$$


$$f_2(\mathbf{z}) = - \sum_{i=1}^{11} \left[ 100((z_{i+1} + 2.0) - (z_i + 2.0)^2)^2 + (1 - (z_i + 2.0))^2 \right]$$


$$f_3(\mathbf{z}) = - \sum_{i=1}^{12} (z_i - 2.0)^2$$



### **Benchmark 23: 2-Objective Zakharov vs. Ackley (Medium-D)**

* **Dimensionality:** $D = 10$ ($d_x = 5$, $d_w = 5$).
* **Recommended Domain bounds:** $[-5.0, 5.0]^{10}$
* **Description:** Zakharov operates as a massive, steep, asymmetrical plate with highly correlated dimension gradients. This opposes Ackley's flat outer region and deep center, testing the acquisition function's capacity to balance extreme gradient magnitudes.
* **Equations:**

$$f_1(\mathbf{z}) = - \left[ \sum_{i=1}^{10} z_i^2 + \left(\sum_{i=1}^{10} 0.5 i z_i\right)^2 + \left(\sum_{i=1}^{10} 0.5 i z_i\right)^4 \right]$$


$$f_2(\mathbf{z}) = - \left[ -20 \exp\left(-0.2 \sqrt{\frac{1}{10} \sum_{i=1}^{10} (z_i - 2.0)^2}\right) - \exp\left(\frac{1}{10} \sum_{i=1}^{10} \cos(2\pi (z_i - 2.0))\right) + 20 + \exp(1) \right]$$



### **Benchmark 24: 4-Objective "All-Valley" Trade-off (Medium-D)**

* **Dimensionality:** $D = 8$ ($d_x = 4$, $d_w = 4$).
* **Recommended Domain bounds:** $[-2.0, 2.0]^8$
* **Description:** A severe 4-way trade-off composed entirely of shifted, narrow valleys. Standard algorithms often collapse onto a single valley floor; this forces the algorithm to suspend itself in the high-dimensional space between four competing ridges.
* **Equations:**

$$f_1(\mathbf{z}) = - \sum_{i=1}^{7} \left[ 100(z_{i+1} - z_i^2)^2 + (1 - z_i)^2 \right]$$


$$f_2(\mathbf{z}) = - \left[ (z_1 - 1)^2 + \sum_{i=2}^{8} i (2 z_i^2 - z_{i-1})^2 \right]$$


$$f_3(\mathbf{z}) = - \sum_{i=1}^{7} \left[ 100((z_{i+1}-1.0) - (z_i-1.0)^2)^2 + (1 - (z_i-1.0))^2 \right]$$


$$f_4(\mathbf{z}) = - \left[ ((z_1+1.0) - 1)^2 + \sum_{i=2}^{8} i (2 (z_i+1.0)^2 - (z_{i-1}+1.0))^2 \right]$$



### **Benchmark 25: 3-Objective Schwefel vs. Styblinski-Tang vs. Sphere (High-D)**

* **Dimensionality:** $D = 20$ ($d_x = 10$, $d_w = 10$).
* **Recommended Domain bounds:** $[-5.0, 5.0]^{20}$
* **Description:** The Schwefel function's global optimum is notoriously hidden at the very edge of the search space, explicitly testing if the acquisition function explores boundaries. This is pitted against Styblinski-Tang's massive internal basins. *(Note: Schwefel inputs are scaled by 100 to map its canonical $[-500, 500]$ topology into the standard $[-5, 5]$ domain).*
* **Equations:**

$$f_1(\mathbf{z}) = - \left[ 8379.658 - \sum_{i=1}^{20} (100 z_i) \sin\left(\sqrt{|100 z_i|}\right) \right]$$


$$f_2(\mathbf{z}) = - \frac{1}{2} \sum_{i=1}^{20} \left( (z_i - 2.0)^4 - 16(z_i - 2.0)^2 + 5(z_i - 2.0) \right)$$


$$f_3(\mathbf{z}) = - \sum_{i=1}^{20} (z_i + 2.0)^2$$



### **Benchmark 26: 3-Objective "Micro-Rugged vs. Macro-Flat" (High-D)**

* **Dimensionality:** $D = 16$ ($d_x = 8$, $d_w = 8$).
* **Recommended Domain bounds:** $[-5.0, 5.0]^{16}$
* **Description:** Griewank is jagged everywhere, providing constant (but misleading) local gradient information. Objectives 2 and 3 utilize the VLMOP2 topology, which provides zero gradient information (exponentially flat) until the algorithm is already directly on top of the feasible region.
* **Equations:**

$$f_1(\mathbf{z}) = - \left[ 1 + \sum_{i=1}^{16} \frac{z_i^2}{4000} - \prod_{i=1}^{16} \cos\left(\frac{z_i}{\sqrt{i}}\right) \right]$$


$$f_2(\mathbf{z}) = - \left[ 1 - \exp\left(-\sum_{i=1}^{16} \left(z_i - \frac{1}{\sqrt{16}}\right)^2\right) \right]$$


$$f_3(\mathbf{z}) = - \left[ 1 - \exp\left(-\sum_{i=1}^{16} \left(z_i + \frac{1}{\sqrt{16}}\right)^2\right) \right]$$
