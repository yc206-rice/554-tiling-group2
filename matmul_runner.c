#include "matmul.h"

#include <stdio.h>
#include <stdlib.h>

static float* allocate_matrix(size_t bytes) {
    void* ptr = NULL;
    int err = posix_memalign(&ptr, 64, bytes);
    if (err != 0) {
        return NULL;
    }
    return (float*)ptr;
}

static void warmup_reference(float* scratch, const float* A, const float* B, int N, int impl) {
    if (impl == MATMUL_IMPL_NAIVE || impl == MATMUL_IMPL_ALL) {
        gemm_naive(A, B, scratch, N);
    } else {
        gemm_tiled(A, B, scratch, N, 32);
    }
}

static void run_kernel(const float* A, const float* B, float* C, int N, int T, int reps, int impl) {
    for (int r = 0; r < reps; ++r) {
        switch (impl) {
            case MATMUL_IMPL_NAIVE:
                gemm_naive(A, B, C, N);
                break;
            case MATMUL_IMPL_TILED:
                gemm_tiled(A, B, C, N, T);
                break;
            case MATMUL_IMPL_ADVANCED:
                gemm_advanced(A, B, C, N, T);
                break;
            default:
                break;
        }
    }
}

int run_benchmark(const matmul_config* cfg) {
    if (!cfg) {
        return 1;
    }

    int N = cfg->N > 0 ? cfg->N : 1024;
    int T = cfg->T > 0 ? cfg->T : 64;
    int reps = cfg->reps > 0 ? cfg->reps : 1;
    int impl = cfg->impl;

    printf("N=%d, T=%d, reps=%d, impl=%d\n", N, T, reps, impl);

    size_t elements = (size_t)N * (size_t)N;
    size_t bytes = elements * sizeof(float);

    float* A = allocate_matrix(bytes);
    float* B = allocate_matrix(bytes);
    float* C = allocate_matrix(bytes);
    float* Cref = allocate_matrix(bytes);

    if (!A || !B || !C || !Cref) {
        fprintf(stderr, "alloc failed\n");
        free(A);
        free(B);
        free(C);
        free(Cref);
        return 1;
    }

    srand(0);
    fill_random(A, elements);
    fill_random(B, elements);
    zero_mat(C, elements);
    zero_mat(Cref, elements);

    if (impl < MATMUL_IMPL_NAIVE || impl > MATMUL_IMPL_ALL) {
        impl = MATMUL_IMPL_ALL;
    }

    warmup_reference(Cref, A, B, N, impl);
    zero_mat(Cref, elements);

    int do_check = (N <= 256);
    if (do_check) {
        run_kernel(A, B, Cref, N, T, reps, MATMUL_IMPL_NAIVE);
    }

    if (impl == MATMUL_IMPL_NAIVE || impl == MATMUL_IMPL_ALL) {
        zero_mat(C, elements);
        double t0 = now_sec();
        run_kernel(A, B, C, N, T, reps, MATMUL_IMPL_NAIVE);
        double t1 = now_sec();
        printf("[naive]   time=%.3f s  GFLOP/s=%.2f\n", t1 - t0, gflops(t1 - t0, N, reps));
        if (do_check) {
            int bad = compare(C, Cref, N);
            printf("[naive]   check mismatches=%d\n", bad);
        }
    }

    if (impl == MATMUL_IMPL_TILED || impl == MATMUL_IMPL_ALL) {
        zero_mat(C, elements);
        double t0 = now_sec();
        run_kernel(A, B, C, N, T, reps, MATMUL_IMPL_TILED);
        double t1 = now_sec();
        printf("[tiled]   time=%.3f s  GFLOP/s=%.2f\n", t1 - t0, gflops(t1 - t0, N, reps));
        if (do_check) {
            int bad = compare(C, Cref, N);
            printf("[tiled]   check mismatches=%d\n", bad);
        }
    }

    if (impl == MATMUL_IMPL_ADVANCED || impl == MATMUL_IMPL_ALL) {
        zero_mat(C, elements);
        double t0 = now_sec();
        run_kernel(A, B, C, N, T, reps, MATMUL_IMPL_ADVANCED);
        double t1 = now_sec();
        printf("[advanced] time=%.3f s  GFLOP/s=%.2f\n", t1 - t0, gflops(t1 - t0, N, reps));
        if (do_check) {
            int bad = compare(C, Cref, N);
            printf("[advanced] check mismatches=%d\n", bad);
        }
    }

    free(A);
    free(B);
    free(C);
    free(Cref);
    return 0;
}
