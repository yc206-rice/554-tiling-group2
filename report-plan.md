# Report Plan: Performance Optimization of Compute-Intensive Kernels on Apple M1

## Overview
This document outlines the structure and content for the conference paper based on the experimental results in this repository. The goal is to produce a 5+ page report following IEEE two-column format.

## 1. Abstract
*   **Summary**: briefly state the goal (optimize MM, Conv, LU), the target hardware (Apple M1 Pro/Max), the techniques used (Tiling, Register Blocking, OpenMP), and the key result (e.g., "Achieved ~134 GFLOP/s in GEMM, a 10x speedup over naive...").

## 2. Introduction
*   **Motivation**: Why are these kernels important? (Deep Learning, Scientific Computing).
*   **Problem**: Memory wall bottleneck. CPU speed > Memory speed.
*   **Solution**: Software-hardware co-design. Optimizing memory access patterns to match the cache hierarchy.
*   **Contributions**:
    *   Evaluation of Loop Tiling on M1.
    *   Implementation of Register Blocking (Micro-kernels).
    *   Analysis of OpenMP scalability.

## 3. Background & Experimental Setup
*   **Hardware Architecture**:
    *   **CPU**: Apple M1 (ARMv8-A based).
    *   **Cache Hierarchy**:
        *   L1 Data: 128KB (P-core) / 64KB (E-core).
        *   L2 Unified: 12MB (P-core) / 4MB (E-core).
    *   **Memory**: High-bandwidth unified memory.
*   **Software Environment**:
    *   Compiler: Clang (Apple LLVM).
    *   Flags: `-O3 -march=native -fopenmp`.
*   **Metrics**:
    *   GFLOP/s (Billions of Floating Point Operations per Second).
    *   Execution Time (seconds).

## 4. Implementation Details
This section details the three optimization levels applied to each kernel.

### 4.1. Matrix Multiplication (GEMM)
*   **Naive**: Triple nested loop. Explain poor spatial/temporal locality.
*   **Tiled**: Blocked loops. Explain how $T$ is chosen to fit $3 \times T^2$ (or similar) into L2 cache.
*   **Advanced (Register Blocked)**:
    *   Explain the $4 \times 4$ micro-kernel.
    *   Show code snippet of the unrolled inner loop.
    *   Explain how this minimizes `load`/`store` instructions and maximizes arithmetic intensity.

### 4.2. 2D Convolution
*   **Algorithm**: Sliding window approach.
*   **Optimizations**:
    *   Similar tiling strategy to GEMM.
    *   Impact of Kernel Size ($K$) on arithmetic intensity.

### 4.3. LU Decomposition
*   **Algorithm**: Gaussian elimination.
*   **Challenges**: Data dependencies limit parallelism compared to GEMM.
*   **Optimizations**: Tiling the update step.

## 5. Results & Analysis
Each subsection should include the plots generated in the repo.

### 5.1. Matrix Multiplication
*   **N-Scaling (Problem Size)**:
    *   *Graph*: `n_gflops.png` (OMP & No-OMP).
    *   *Analysis*: Why does Naive drop? Why does Advanced scale best? Compare peak GFLOPs.
*   **T-Scaling (Tile Size)**:
    *   *Graph*: `t_gflops.png`.
    *   *Analysis*: The "Goldilocks" zone. Small $T$ = overhead; Large $T$ = cache misses. Identify the peak (likely around $T=32$ to $128$).

### 5.2. 2D Convolution
*   **K-Scaling (Kernel Size)**:
    *   *Graph*: `k_gflops.png`.
    *   *Analysis*: Larger $K$ = more reuse = higher GFLOPs.
*   **Performance**: Compare Naive vs. Advanced.

### 5.3. LU Decomposition
*   **Anomaly Discussion**: Why is `Tiled` faster than `Advanced` here? (Compiler vectorization vs. Manual overhead).
*   **Scalability**: How well does it parallelize with OpenMP?

## 6. Conclusion
*   **Summary**: Recap the speedups (e.g., "Advanced GEMM is X times faster than Naive").
*   **Key Takeaway**: Tiling is essential, but Register Blocking provides the final boost for regular patterns (GEMM). However, for complex dependencies (LU), compiler auto-vectorization of simple tiled loops can beat complex manual micro-kernels.

## 7. Future Work
*   **SIMD Intrinsics**: Explicit use of NEON instructions (`<arm_neon.h>`) instead of relying on compiler auto-vectorization.
*   **Assembly Tuning**: Hand-written assembly for the micro-kernel.
*   **GPU Offloading**: Using Metal Performance Shaders (MPS) on M1.

## 8. References
1.  Golub, G. H., & Van Loan, C. F. *Matrix Computations*.
2.  Hennessy, J. L., & Patterson, D. A. *Computer Architecture: A Quantitative Approach*.
3.  Lam, M. S., et al. "The cache performance and optimizations of blocked algorithms."
4.  Apple Developer Documentation (M1 Architecture).
