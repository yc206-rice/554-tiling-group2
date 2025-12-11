#include "matmul.h"

#include <stddef.h>

static inline void micro_kernel_4x4(
    const float* A, const float* B, float* C,
    int lda, int ldb, int ldc, int Kvalid)
{
    float c00 = 0, c01 = 0, c02 = 0, c03 = 0;
    float c10 = 0, c11 = 0, c12 = 0, c13 = 0;
    float c20 = 0, c21 = 0, c22 = 0, c23 = 0;
    float c30 = 0, c31 = 0, c32 = 0, c33 = 0;

    for (int k = 0; k < Kvalid; ++k) {
        float a0 = A[0 * lda + k];
        float a1 = A[1 * lda + k];
        float a2 = A[2 * lda + k];
        float a3 = A[3 * lda + k];

        const float* Bk = &B[(size_t)k * ldb];
        float b0 = Bk[0];
        float b1 = Bk[1];
        float b2 = Bk[2];
        float b3 = Bk[3];

        c00 += a0 * b0;
        c01 += a0 * b1;
        c02 += a0 * b2;
        c03 += a0 * b3;
        c10 += a1 * b0;
        c11 += a1 * b1;
        c12 += a1 * b2;
        c13 += a1 * b3;
        c20 += a2 * b0;
        c21 += a2 * b1;
        c22 += a2 * b2;
        c23 += a2 * b3;
        c30 += a3 * b0;
        c31 += a3 * b1;
        c32 += a3 * b2;
        c33 += a3 * b3;
    }

    C[0 * ldc + 0] += c00;
    C[0 * ldc + 1] += c01;
    C[0 * ldc + 2] += c02;
    C[0 * ldc + 3] += c03;
    C[1 * ldc + 0] += c10;
    C[1 * ldc + 1] += c11;
    C[1 * ldc + 2] += c12;
    C[1 * ldc + 3] += c13;
    C[2 * ldc + 0] += c20;
    C[2 * ldc + 1] += c21;
    C[2 * ldc + 2] += c22;
    C[2 * ldc + 3] += c23;
    C[3 * ldc + 0] += c30;
    C[3 * ldc + 1] += c31;
    C[3 * ldc + 2] += c32;
    C[3 * ldc + 3] += c33;
}

void gemm_naive(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float cij = C[(size_t)i * N + j];
            for (int k = 0; k < N; ++k) {
                cij += A[(size_t)i * N + k] * B[(size_t)k * N + j];
            }
            C[(size_t)i * N + j] = cij;
        }
    }
}

void gemm_tiled(const float* A, const float* B, float* C, int N, int T) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < N; ii += T) {
        for (int jj = 0; jj < N; jj += T) {
            int iimax = ii + T;
            if (iimax > N) iimax = N;
            int jjmax = jj + T;
            if (jjmax > N) jjmax = N;
            for (int kk = 0; kk < N; kk += T) {
                int kkmax = kk + T;
                if (kkmax > N) kkmax = N;
                for (int i = ii; i < iimax; ++i) {
                    for (int j = jj; j < jjmax; ++j) {
                        float cij = C[(size_t)i * N + j];
                        for (int k = kk; k < kkmax; ++k) {
                            cij += A[(size_t)i * N + k] * B[(size_t)k * N + j];
                        }
                        C[(size_t)i * N + j] = cij;
                    }
                }
            }
        }
    }
}

void gemm_advanced(const float* A, const float* B, float* C, int N, int T) {
    const int MR = 4, NR = 4;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < N; ii += T) {
        for (int jj = 0; jj < N; jj += T) {
            int iimax = ii + T;
            if (iimax > N) iimax = N;
            int jjmax = jj + T;
            if (jjmax > N) jjmax = N;
            for (int kk = 0; kk < N; kk += T) {
                int kkmax = kk + T;
                if (kkmax > N) kkmax = N;
                for (int i = ii; i < iimax; i += MR) {
                    int imax = i + MR;
                    if (imax > iimax) imax = iimax;
                    for (int j = jj; j < jjmax; j += NR) {
                        int jmax = j + NR;
                        if (jmax > jjmax) jmax = jjmax;

                        int m = imax - i;
                        int n = jmax - j;

                        if (m == MR && n == NR) {
                            micro_kernel_4x4(&A[(size_t)i * N + kk],
                                             &B[(size_t)kk * N + j],
                                             &C[(size_t)i * N + j],
                                             N, N, N, kkmax - kk);
                        } else {
                            for (int ii2 = 0; ii2 < m; ++ii2) {
                                for (int jj2 = 0; jj2 < n; ++jj2) {
                                    float sum = 0.f;
                                    for (int k = kk; k < kkmax; ++k) {
                                        sum += A[(size_t)(i + ii2) * N + k] *
                                               B[(size_t)k * N + (j + jj2)];
                                    }
                                    C[(size_t)(i + ii2) * N + (j + jj2)] += sum;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
