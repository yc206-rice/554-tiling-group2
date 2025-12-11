#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CONV_IMPL_NAIVE = 0,
    CONV_IMPL_TILED = 1,
    CONV_IMPL_ADVANCED = 2,
    CONV_IMPL_ALL = 3
} conv_impl;

typedef struct {
    int N;      // Image size (NxN)
    int K;      // Kernel size (KxK)
    int T;      // Tile size
    int reps;   // Number of repetitions
    int impl;   // Implementation choice
} conv_config;

// Benchmark functions
void conv_naive(const float* input, const float* kernel, float* output, int N, int K);
void conv_tiled(const float* input, const float* kernel, float* output, int N, int K, int T);
void conv_advanced(const float* input, const float* kernel, float* output, int N, int K, int T);

int run_conv_benchmark(const conv_config* cfg);

#ifdef __cplusplus
}
#endif

#endif // CONVOLUTION_H
