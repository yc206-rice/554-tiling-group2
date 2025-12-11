#include "lu.h"
#include <math.h>
#include <stddef.h>

// Minimal implementation of naive scalar LU factorization (no pivoting for simplicity in this benchmark)
// A is modified in-place to store L (below diagonal) and U (diagonal and above).
void lu_naive(float* A, int N) {
    for (int k = 0; k < N; ++k) {
        // For each row i > k
        for (int i = k + 1; i < N; ++i) {
            // L[i, k] = A[i, k] / U[k, k]
            // Note: A[k, k] is U[k, k]
            float factor = A[i * N + k] / A[k * N + k];
            A[i * N + k] = factor; // Store L part

            // Update the rest of the row: A[i, j] = A[i, j] - L[i, k] * U[k, j]
            for (int j = k + 1; j < N; ++j) {
                A[i * N + j] -= factor * A[k * N + j];
            }
        }
    }
}

// Helper for the tiled version: process a small diagonal block
static void lu_block_naive(float* A, int N, int row_start, int col_start, int bsize) {
    for (int k = 0; k < bsize; ++k) {
        int gk = row_start + k; // global k index
        for (int i = k + 1; i < bsize; ++i) {
            int gi = row_start + i;
            float factor = A[gi * N + gk] / A[gk * N + gk];
            A[gi * N + gk] = factor;
            for (int j = k + 1; j < bsize; ++j) {
                int gj = col_start + j;
                A[gi * N + gj] -= factor * A[gk * N + gj];
            }
        }
    }
}

// Tiled (Blocked) LU Factorization
// This is a Right-Looking Blocked LU algorithm.
void lu_tiled(float* A, int N, int T) {
    for (int k = 0; k < N; k += T) {
        int bsize = (k + T > N) ? (N - k) : T;

        // 1. Factorize the diagonal block A[k:k+bsize, k:k+bsize]
        lu_block_naive(A, N, k, k, bsize);

        // 2. Update Panel U (Top block to the right): A[k:k+bsize, j:j+T]
        // This is essentially a triangular solve (TRSM).
        // For U part: L11 * U12 = A12 -> U12 = L11^-1 * A12.
        // We use a Right-Looking update to ensure stride-1 access.
        #pragma omp parallel for schedule(static)
        for (int j = k + T; j < N; j += T) {
            int jsize = (j + T > N) ? (N - j) : T;
            
            // Loop order: kk (pivot), i (row), jj (col).
            // For each pivot kk, we update all rows i > kk.
            for (int kk = 0; kk < bsize; ++kk) {
                for (int i = kk + 1; i < bsize; ++i) {
                    float factor = A[(k + i) * N + (k + kk)];
                    for (int jj = 0; jj < jsize; ++jj) {
                         A[(k + i) * N + (j + jj)] -= factor * A[(k + kk) * N + (j + jj)];
                    }
                }
            }
        }

        // 3. Update Panel L (Left block below): A[i:i+T, k:k+bsize]
        // L21 * U11 = A21 -> L21 = A21 * U11^-1
        // Since U11 is upper triangular, we solve by back substitution or simply:
        // A[i, k] = A[i, k] / U[k, k]
        #pragma omp parallel for schedule(static)
        for (int i = k + T; i < N; i += T) {
            int isize = (i + T > N) ? (N - i) : T;
            for (int ii = 0; ii < isize; ++ii) {
                for (int kk = 0; kk < bsize; ++kk) {
                    // Division by pivot (diagonal of U)
                    float u_diag = A[(k + kk) * N + (k + kk)];
                    A[(i + ii) * N + (k + kk)] /= u_diag;
                    
                    // Update remaining: L21 row - (scaled L21 element) * (U11 row)
                    for (int jj = kk + 1; jj < bsize; ++jj) {
                        A[(i + ii) * N + (k + jj)] -= A[(i + ii) * N + (k + kk)] * A[(k + kk) * N + (k + jj)];
                    }
                }
            }
        }

        // 4. Trailing Submatrix Update (GEMM): A[i, j] -= L[i, k] * U[k, j]
        // This is the most compute-intensive part: C = C - A * B
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = k + T; i < N; i += T) {
            for (int j = k + T; j < N; j += T) {
                int isize = (i + T > N) ? (N - i) : T;
                int jsize = (j + T > N) ? (N - j) : T;
                
                // Optimized triple loop matmul (ikj order for stride-1 access)
                for (int ii = 0; ii < isize; ++ii) {
                    for (int kk = 0; kk < bsize; ++kk) {
                        float factor = A[(i + ii) * N + (k + kk)];
                        for (int jj = 0; jj < jsize; ++jj) {
                            A[(i + ii) * N + (j + jj)] -= factor * A[(k + kk) * N + (j + jj)];
                        }
                    }
                }
            }
        }
    }
}

// Advanced Micro-Kernel (4x4 register blocking) for the trailing update
// This mimics gemm_advanced but handles the subtraction A -= L*U
static inline void lu_gemm_micro_kernel_4x4(
    const float* L_ptr, const float* U_ptr, float* A_ptr,
    int ldl, int ldu, int lda, int Kvalid)
{
    float c00 = 0, c01 = 0, c02 = 0, c03 = 0;
    float c10 = 0, c11 = 0, c12 = 0, c13 = 0;
    float c20 = 0, c21 = 0, c22 = 0, c23 = 0;
    float c30 = 0, c31 = 0, c32 = 0, c33 = 0;

    // Accumulate L*U product in registers
    for (int k = 0; k < Kvalid; ++k) {
        float l0 = L_ptr[0 * ldl + k];
        float l1 = L_ptr[1 * ldl + k];
        float l2 = L_ptr[2 * ldl + k];
        float l3 = L_ptr[3 * ldl + k];

        const float* Uk = &U_ptr[(size_t)k * ldu];
        float u0 = Uk[0];
        float u1 = Uk[1];
        float u2 = Uk[2];
        float u3 = Uk[3];

        c00 += l0 * u0; c01 += l0 * u1; c02 += l0 * u2; c03 += l0 * u3;
        c10 += l1 * u0; c11 += l1 * u1; c12 += l1 * u2; c13 += l1 * u3;
        c20 += l2 * u0; c21 += l2 * u1; c22 += l2 * u2; c23 += l2 * u3;
        c30 += l3 * u0; c31 += l3 * u1; c32 += l3 * u2; c33 += l3 * u3;
    }

    // Subtract result from A
    A_ptr[0 * lda + 0] -= c00; A_ptr[0 * lda + 1] -= c01; A_ptr[0 * lda + 2] -= c02; A_ptr[0 * lda + 3] -= c03;
    A_ptr[1 * lda + 0] -= c10; A_ptr[1 * lda + 1] -= c11; A_ptr[1 * lda + 2] -= c12; A_ptr[1 * lda + 3] -= c13;
    A_ptr[2 * lda + 0] -= c20; A_ptr[2 * lda + 1] -= c21; A_ptr[2 * lda + 2] -= c22; A_ptr[2 * lda + 3] -= c23;
    A_ptr[3 * lda + 0] -= c30; A_ptr[3 * lda + 1] -= c31; A_ptr[3 * lda + 2] -= c32; A_ptr[3 * lda + 3] -= c33;
}

// Advanced LU: Uses register-blocked GEMM for the trailing submatrix update
void lu_advanced(float* A, int N, int T) {
    const int MR = 4, NR = 4;

    for (int k = 0; k < N; k += T) {
        int bsize = (k + T > N) ? (N - k) : T;

        // 1. Factorize diagonal block (Same as tiled)
        lu_block_naive(A, N, k, k, bsize);

        // 2. Update Panel U (Same as tiled)
        #pragma omp parallel for schedule(static)
        for (int j = k + T; j < N; j += T) {
            int jsize = (j + T > N) ? (N - j) : T;
            // Optimize this loop: This is TRSM (U12 = L11^-1 * A12)
            // L11 is unit lower triangular.
            for (int jj = 0; jj < jsize; ++jj) {
                for (int kk = 0; kk < bsize; ++kk) {
                    float val = A[(k + kk) * N + (j + jj)];
                    // No division needed as L diagonal is 1
                    for (int i = kk + 1; i < bsize; ++i) {
                         A[(k + i) * N + (j + jj)] -= A[(k + i) * N + (k + kk)] * val;
                    }
                }
            }
        }

        // 3. Update Panel L (Same as tiled)
        #pragma omp parallel for schedule(static)
        for (int i = k + T; i < N; i += T) {
            int isize = (i + T > N) ? (N - i) : T;
            // Optimize this loop: TRSM (L21 = A21 * U11^-1)
            // U11 is upper triangular.
            for (int ii = 0; ii < isize; ++ii) {
                for (int kk = 0; kk < bsize; ++kk) {
                    float u_diag = A[(k + kk) * N + (k + kk)];
                    float val = A[(i + ii) * N + (k + kk)] / u_diag;
                    A[(i + ii) * N + (k + kk)] = val; // Store scaled L value
                    
                    for (int jj = kk + 1; jj < bsize; ++jj) {
                        A[(i + ii) * N + (k + jj)] -= val * A[(k + kk) * N + (k + jj)];
                    }
                }
            }
        }

        // 4. Trailing Submatrix Update (Optimized GEMM)
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = k + T; i < N; i += T) {
            for (int j = k + T; j < N; j += T) {
                int isize = (i + T > N) ? (N - i) : T;
                int jsize = (j + T > N) ? (N - j) : T;
                
                // Blocked loop over the sub-blocks
                for (int ii = 0; ii < isize; ii += MR) {
                    int imax = ii + MR;
                    if (imax > isize) imax = isize;
                    
                    for (int jj = 0; jj < jsize; jj += NR) {
                        int jmax = jj + NR;
                        if (jmax > jsize) jmax = jsize;

                        int m = imax - ii;
                        int n = jmax - jj;

                        if (m == MR && n == NR) {
                            lu_gemm_micro_kernel_4x4(
                                &A[(i + ii) * N + k],  // L part (row i+ii, col k)
                                &A[k * N + (j + jj)],  // U part (row k, col j+jj)
                                &A[(i + ii) * N + (j + jj)], // Target A
                                N, N, N, bsize);
                        } else {
                            // Fallback
                            for (int ii2 = 0; ii2 < m; ++ii2) {
                                for (int jj2 = 0; jj2 < n; ++jj2) {
                                    float sum = 0.0f;
                                    for (int kk = 0; kk < bsize; ++kk) {
                                        sum += A[(i + ii + ii2) * N + (k + kk)] * A[(k + kk) * N + (j + jj + jj2)];
                                    }
                                    A[(i + ii + ii2) * N + (j + jj + jj2)] -= sum;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
