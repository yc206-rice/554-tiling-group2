#ifndef MATMUL_H
#define MATMUL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MATMUL_IMPL_NAIVE = 0,
    MATMUL_IMPL_TILED = 1,
    MATMUL_IMPL_ADVANCED = 2,
    MATMUL_IMPL_ALL = 3
} matmul_impl;

typedef struct {
    int N;
    int T;
    int reps;
    int impl;
} matmul_config;

double now_sec(void);
double gflops(double sec, int N, int reps);

void fill_random(float* data, size_t count);
void zero_mat(float* data, size_t count);

void gemm_naive(const float* A, const float* B, float* C, int N);
void gemm_tiled(const float* A, const float* B, float* C, int N, int T);
void gemm_advanced(const float* A, const float* B, float* C, int N, int T);

int compare(const float* X, const float* Y, int N);

int run_benchmark(const matmul_config* cfg);

#ifdef __cplusplus
}
#endif

#endif // MATMUL_H
