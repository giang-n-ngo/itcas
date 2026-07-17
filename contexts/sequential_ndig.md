# Implementation Specification: Context-Repulsive NDIG (CR-NDIG)

This document specifies the implementation of **CR-NDIG** (Context-Repulsive Normalized Depth Information Gain). This is a purely sequential acquisition function designed for environments where batch sampling (with DPP) is unavailable.

It hardwires context-space exploration directly into the acquisition gradient by applying a mathematical repulsion penalty based on previously evaluated contexts.

**The Equation:**
$$q_{CR-NDIG}(x, c) = q_{NDIG}(x, c) \times \prod_{i=1}^{t-1} \big( 1 - k_{ctx}(c, c_i) \big)$$

Where:
* $q_{NDIG}(x, c)$ is the standard joint NDIG acquisition score.
* $c$ is the context vector currently being evaluated by the optimizer.
* $c_i$ are all previously evaluated context vectors from iterations $1 \dots t-1$.
* $k_{ctx}(c, c_i)$ is an RBF (Gaussian) kernel operating strictly on the context subspace.

