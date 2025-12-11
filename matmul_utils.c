#include "matmul.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

double gflops(double sec, int N, int reps) {
    double flops = 2.0 * (double)N * (double)N * (double)N * (double)reps;
    return flops / sec / 1e9;
}

void fill_random(float* data, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        data[i] = (float)(rand() / (double)RAND_MAX) - 0.5f;
    }
}

void zero_mat(float* data, size_t count) {
    memset(data, 0, count * sizeof(float));
}

static int almost_equal(float a, float b) {
    float diff = fabsf(a - b);
    float tol = 1e-2f;
    float max_val = fmaxf(1.0f, fmaxf(fabsf(a), fabsf(b)));
    return diff <= tol * max_val;
}

int compare(const float* X, const float* Y, int N) {
    int bad = 0;
    int total = N * N;
    for (int i = 0; i < total; ++i) {
        if (!almost_equal(X[i], Y[i])) {
            if (++bad <= 5) {
                fprintf(stderr, "mismatch at %d: %g vs %g\n", i, X[i], Y[i]);
            }
        }
    }
    return bad;
}
