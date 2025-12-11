#include "convolution.h"
#include "matmul.h" // For utils

#include <stdio.h>
#include <stdlib.h>

static float* allocate_matrix(size_t bytes) {
    void* ptr = NULL;
    int err = posix_memalign(&ptr, 64, bytes);
    if (err != 0) return NULL;
    return (float*)ptr;
}

static double conv_gflops(double sec, int N, int K, int reps) {
    // 2 * N * N * K * K ops per rep
    double ops = 2.0 * (double)N * (double)N * (double)K * (double)K;
    return (ops * reps) / sec / 1e9;
}

static void run_kernel(const float* input, const float* kernel, float* output, int N, int K, int T, int reps, int impl) {
    for (int r = 0; r < reps; ++r) {
        switch (impl) {
            case CONV_IMPL_NAIVE:
                conv_naive(input, kernel, output, N, K);
                break;
            case CONV_IMPL_TILED:
                conv_tiled(input, kernel, output, N, K, T);
                break;
            case CONV_IMPL_ADVANCED:
                conv_advanced(input, kernel, output, N, K, T);
                break;
            default:
                break;
        }
    }
}

int run_conv_benchmark(const conv_config* cfg) {
    if (!cfg) return 1;

    int N = cfg->N > 0 ? cfg->N : 1024;
    int K = cfg->K > 0 ? cfg->K : 3;
    int T = cfg->T > 0 ? cfg->T : 64;
    int reps = cfg->reps > 0 ? cfg->reps : 1;
    int impl = cfg->impl;

    printf("CONV: N=%d, K=%d, T=%d, reps=%d, impl=%d\n", N, K, T, reps, impl);

    size_t input_elements = (size_t)N * N;
    size_t kernel_elements = (size_t)K * K;
    
    float* input = allocate_matrix(input_elements * sizeof(float));
    float* kernel = allocate_matrix(kernel_elements * sizeof(float));
    float* output = allocate_matrix(input_elements * sizeof(float));
    float* ref = allocate_matrix(input_elements * sizeof(float));

    if (!input || !kernel || !output || !ref) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    srand(0);
    fill_random(input, input_elements);
    fill_random(kernel, kernel_elements);
    zero_mat(output, input_elements);
    zero_mat(ref, input_elements);

    if (impl < CONV_IMPL_NAIVE || impl > CONV_IMPL_ALL) {
        impl = CONV_IMPL_ALL;
    }

    // Warmup / Reference
    conv_naive(input, kernel, ref, N, K);

    // Naive Benchmark
    if (impl == CONV_IMPL_NAIVE || impl == CONV_IMPL_ALL) {
        zero_mat(output, input_elements);
        double t0 = now_sec();
        run_kernel(input, kernel, output, N, K, T, reps, CONV_IMPL_NAIVE);
        double t1 = now_sec();
        
        int bad = compare(output, ref, N); // reusing compare from matmul, treats as NxN
        printf("[naive]   time=%.3f s  GFLOP/s=%.2f  diff=%d\n", 
               t1 - t0, conv_gflops(t1 - t0, N, K, reps), bad);
    }

    // Tiled Benchmark
    if (impl == CONV_IMPL_TILED || impl == CONV_IMPL_ALL) {
        zero_mat(output, input_elements);
        double t0 = now_sec();
        run_kernel(input, kernel, output, N, K, T, reps, CONV_IMPL_TILED);
        double t1 = now_sec();
        
        int bad = compare(output, ref, N);
        printf("[tiled]   time=%.3f s  GFLOP/s=%.2f  diff=%d\n", 
               t1 - t0, conv_gflops(t1 - t0, N, K, reps), bad);
    }

    // Advanced Benchmark
    if (impl == CONV_IMPL_ADVANCED || impl == CONV_IMPL_ALL) {
        zero_mat(output, input_elements);
        double t0 = now_sec();
        run_kernel(input, kernel, output, N, K, T, reps, CONV_IMPL_ADVANCED);
        double t1 = now_sec();
        
        int bad = compare(output, ref, N);
        printf("[advance] time=%.3f s  GFLOP/s=%.2f  diff=%d\n", 
               t1 - t0, conv_gflops(t1 - t0, N, K, reps), bad);
    }

    free(input);
    free(kernel);
    free(output);
    free(ref);

    return 0;
}
