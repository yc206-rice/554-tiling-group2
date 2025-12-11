#include "lu.h"
#include "matmul.h" // for utils

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float* allocate_matrix(size_t bytes) {
    void* ptr = NULL;
    int err = posix_memalign(&ptr, 64, bytes);
    if (err != 0) return NULL;
    return (float*)ptr;
}

// FLOPs for LU is approx 2/3 * N^3
static double lu_gflops(double sec, int N, int reps) {
    double ops = (2.0 / 3.0) * (double)N * (double)N * (double)N;
    return (ops * reps) / sec / 1e9;
}

static void run_kernel(float* A, int N, int T, int reps, int impl) {
    // Note: LU modifies A in-place. We need to reset A for each rep.
    // However, the interface usually assumes one call. 
    // For benchmarking correct timing of REPS, we would typically need to copy A each time.
    // To simplify, we will time just ONE rep inside the measurement loop in the caller,
    // or we assume the caller handles the reset.
    // Given existing structure, let's adapt:
    
    switch (impl) {
        case LU_IMPL_NAIVE:
            lu_naive(A, N);
            break;
        case LU_IMPL_TILED:
            lu_tiled(A, N, T);
            break;
        case LU_IMPL_ADVANCED:
            lu_advanced(A, N, T);
            break;
        default:
            break;
    }
}

int run_lu_benchmark(const lu_config* cfg) {
    if (!cfg) return 1;

    int N = cfg->N > 0 ? cfg->N : 1024;
    int T = cfg->T > 0 ? cfg->T : 64;
    int reps = cfg->reps > 0 ? cfg->reps : 1;
    int impl = cfg->impl;

    printf("LU: N=%d, T=%d, reps=%d, impl=%d\n", N, T, reps, impl);

    size_t elements = (size_t)N * N;
    size_t bytes = elements * sizeof(float);

    float* A_orig = allocate_matrix(bytes);
    float* A_work = allocate_matrix(bytes);
    float* A_ref = allocate_matrix(bytes);

    if (!A_orig || !A_work || !A_ref) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    srand(0);
    // Initialize with a diagonally dominant matrix to ensure numerical stability without pivoting
    fill_random(A_orig, elements);
    for (int i = 0; i < N; ++i) {
        A_orig[i * N + i] += (float)N; 
    }

    // Warmup / Reference
    memcpy(A_ref, A_orig, bytes);
    lu_naive(A_ref, N);

    if (impl < LU_IMPL_NAIVE || impl > LU_IMPL_ALL) {
        impl = LU_IMPL_ALL;
    }

    // Since LU is in-place, we must copy A_orig to A_work before EACH run.
    // We will measure total time for 'reps' executions including copy overhead?
    // Usually benchmarks exclude copy time.
    // Let's measure accumulation of kernel times.

    // Naive
    if (impl == LU_IMPL_NAIVE || impl == LU_IMPL_ALL) {
        double total_time = 0.0;
        for(int r=0; r<reps; ++r) {
            memcpy(A_work, A_orig, bytes);
            double t0 = now_sec();
            lu_naive(A_work, N);
            double t1 = now_sec();
            total_time += (t1 - t0);
        }
        
        // Verify last run
        int bad = compare(A_work, A_ref, N);
        printf("[naive]   time=%.3f s  GFLOP/s=%.2f  diff=%d\n", 
               total_time, lu_gflops(total_time, N, reps), bad);
    }

    // Tiled
    if (impl == LU_IMPL_TILED || impl == LU_IMPL_ALL) {
        double total_time = 0.0;
        for(int r=0; r<reps; ++r) {
            memcpy(A_work, A_orig, bytes);
            double t0 = now_sec();
            lu_tiled(A_work, N, T);
            double t1 = now_sec();
            total_time += (t1 - t0);
        }
        
        int bad = compare(A_work, A_ref, N);
        printf("[tiled]   time=%.3f s  GFLOP/s=%.2f  diff=%d\n", 
               total_time, lu_gflops(total_time, N, reps), bad);
    }

    // Advanced
    if (impl == LU_IMPL_ADVANCED || impl == LU_IMPL_ALL) {
        double total_time = 0.0;
        for(int r=0; r<reps; ++r) {
            memcpy(A_work, A_orig, bytes);
            double t0 = now_sec();
            lu_advanced(A_work, N, T);
            double t1 = now_sec();
            total_time += (t1 - t0);
        }
        
        int bad = compare(A_work, A_ref, N);
        printf("[advance] time=%.3f s  GFLOP/s=%.2f  diff=%d\n", 
               total_time, lu_gflops(total_time, N, reps), bad);
    }

    free(A_orig);
    free(A_work);
    free(A_ref);

    return 0;
}
