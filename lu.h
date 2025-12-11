#ifndef LU_H
#define LU_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    LU_IMPL_NAIVE = 0,
    LU_IMPL_TILED = 1,
    LU_IMPL_ADVANCED = 2,
    LU_IMPL_ALL = 3
} lu_impl;

typedef struct {
    int N;      // Matrix size (NxN)
    int T;      // Block/Tile size (TxT)
    int reps;   // Number of repetitions
    int impl;   // Implementation choice
} lu_config;

// Benchmark functions
void lu_naive(float* A, int N);
void lu_tiled(float* A, int N, int T);
void lu_advanced(float* A, int N, int T);

int run_lu_benchmark(const lu_config* cfg);

#ifdef __cplusplus
}
#endif

#endif // LU_H
