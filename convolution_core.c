#include "convolution.h"
#include <stddef.h>

void conv_naive(const float* input, const float* kernel, float* output, int N, int K) {
    int pad = K / 2;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int ki = 0; ki < K; ++ki) {
                for (int kj = 0; kj < K; ++kj) {
                    int ii = i + ki - pad;
                    int jj = j + kj - pad;
                    if (ii >= 0 && ii < N && jj >= 0 && jj < N) {
                        sum += input[ii * N + jj] * kernel[ki * K + kj];
                    }
                }
            }
            output[i * N + j] = sum;
        }
    }
}

static inline void conv_micro_kernel_4x4(
    const float* input, const float* kernel, float* output,
    int N, int K, int row_start, int col_start)
{
    // Accumulators for a 4x4 block of output
    float c00 = 0, c01 = 0, c02 = 0, c03 = 0;
    float c10 = 0, c11 = 0, c12 = 0, c13 = 0;
    float c20 = 0, c21 = 0, c22 = 0, c23 = 0;
    float c30 = 0, c31 = 0, c32 = 0, c33 = 0;

    int pad = K / 2;
    int r0 = row_start - pad;
    int c0 = col_start - pad;

    // Iterate over kernel
    for (int ki = 0; ki < K; ++ki) {
        for (int kj = 0; kj < K; ++kj) {
            float w = kernel[ki * K + kj];
            
            // For each output pixel (i, j) in the 4x4 block,
            // the corresponding input pixel is at (r0 + i + ki, c0 + j + kj)
            
            // Row 0 of block
            int ri = r0 + ki;
            if (ri >= 0 && ri < N) {
                // Pre-calculate base index for row ri
                int row_offset = ri * N;
                
                // Column 0
                int ci = c0 + kj;
                if (ci >= 0 && ci < N) c00 += input[row_offset + ci] * w;
                
                // Column 1
                ci = c0 + 1 + kj;
                if (ci >= 0 && ci < N) c01 += input[row_offset + ci] * w;

                // Column 2
                ci = c0 + 2 + kj;
                if (ci >= 0 && ci < N) c02 += input[row_offset + ci] * w;

                // Column 3
                ci = c0 + 3 + kj;
                if (ci >= 0 && ci < N) c03 += input[row_offset + ci] * w;
            }

            // Row 1 of block
            ri = r0 + 1 + ki;
            if (ri >= 0 && ri < N) {
                int row_offset = ri * N;
                
                int ci = c0 + kj;
                if (ci >= 0 && ci < N) c10 += input[row_offset + ci] * w;
                
                ci = c0 + 1 + kj;
                if (ci >= 0 && ci < N) c11 += input[row_offset + ci] * w;
                
                ci = c0 + 2 + kj;
                if (ci >= 0 && ci < N) c12 += input[row_offset + ci] * w;
                
                ci = c0 + 3 + kj;
                if (ci >= 0 && ci < N) c13 += input[row_offset + ci] * w;
            }

            // Row 2 of block
            ri = r0 + 2 + ki;
            if (ri >= 0 && ri < N) {
                int row_offset = ri * N;
                
                int ci = c0 + kj;
                if (ci >= 0 && ci < N) c20 += input[row_offset + ci] * w;
                
                ci = c0 + 1 + kj;
                if (ci >= 0 && ci < N) c21 += input[row_offset + ci] * w;
                
                ci = c0 + 2 + kj;
                if (ci >= 0 && ci < N) c22 += input[row_offset + ci] * w;
                
                ci = c0 + 3 + kj;
                if (ci >= 0 && ci < N) c23 += input[row_offset + ci] * w;
            }

            // Row 3 of block
            ri = r0 + 3 + ki;
            if (ri >= 0 && ri < N) {
                int row_offset = ri * N;
                
                int ci = c0 + kj;
                if (ci >= 0 && ci < N) c30 += input[row_offset + ci] * w;
                
                ci = c0 + 1 + kj;
                if (ci >= 0 && ci < N) c31 += input[row_offset + ci] * w;
                
                ci = c0 + 2 + kj;
                if (ci >= 0 && ci < N) c32 += input[row_offset + ci] * w;
                
                ci = c0 + 3 + kj;
                if (ci >= 0 && ci < N) c33 += input[row_offset + ci] * w;
            }
        }
    }

    // Write back
    output[(row_start + 0) * N + (col_start + 0)] = c00;
    output[(row_start + 0) * N + (col_start + 1)] = c01;
    output[(row_start + 0) * N + (col_start + 2)] = c02;
    output[(row_start + 0) * N + (col_start + 3)] = c03;

    output[(row_start + 1) * N + (col_start + 0)] = c10;
    output[(row_start + 1) * N + (col_start + 1)] = c11;
    output[(row_start + 1) * N + (col_start + 2)] = c12;
    output[(row_start + 1) * N + (col_start + 3)] = c13;

    output[(row_start + 2) * N + (col_start + 0)] = c20;
    output[(row_start + 2) * N + (col_start + 1)] = c21;
    output[(row_start + 2) * N + (col_start + 2)] = c22;
    output[(row_start + 2) * N + (col_start + 3)] = c23;

    output[(row_start + 3) * N + (col_start + 0)] = c30;
    output[(row_start + 3) * N + (col_start + 1)] = c31;
    output[(row_start + 3) * N + (col_start + 2)] = c32;
    output[(row_start + 3) * N + (col_start + 3)] = c33;
}

void conv_advanced(const float* input, const float* kernel, float* output, int N, int K, int T) {
    int pad = K / 2;
    const int MR = 4;
    const int NR = 4;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < N; ii += T) {
        for (int jj = 0; jj < N; jj += T) {
            int iimax = ii + T;
            if (iimax > N) iimax = N;
            int jjmax = jj + T;
            if (jjmax > N) jjmax = N;
            
            for (int i = ii; i < iimax; i += MR) {
                int imax = i + MR;
                if (imax > iimax) imax = iimax;
                
                for (int j = jj; j < jjmax; j += NR) {
                    int jmax = j + NR;
                    if (jmax > jjmax) jmax = jjmax;

                    int m = imax - i;
                    int n = jmax - j;

                    // If we have a full 4x4 block, use the micro-kernel
                    if (m == MR && n == NR) {
                        conv_micro_kernel_4x4(input, kernel, output, N, K, i, j);
                    } else {
                        // Fallback for edge cases
                        for (int ii2 = 0; ii2 < m; ++ii2) {
                            for (int jj2 = 0; jj2 < n; ++jj2) {
                                float sum = 0.0f;
                                int r_out = i + ii2;
                                int c_out = j + jj2;
                                
                                for (int ki = 0; ki < K; ++ki) {
                                    for (int kj = 0; kj < K; ++kj) {
                                        int r = r_out + ki - pad;
                                        int c = c_out + kj - pad;
                                        if (r >= 0 && r < N && c >= 0 && c < N) {
                                            sum += input[r * N + c] * kernel[ki * K + kj];
                                        }
                                    }
                                }
                                output[r_out * N + c_out] = sum;
                            }
                        }
                    }
                }
            }
        }
    }
}

void conv_tiled(const float* input, const float* kernel, float* output, int N, int K, int T) {
    int pad = K / 2;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < N; ii += T) {
        for (int jj = 0; jj < N; jj += T) {
            int iimax = ii + T;
            if (iimax > N) iimax = N;
            int jjmax = jj + T;
            if (jjmax > N) jjmax = N;
            
            for (int i = ii; i < iimax; ++i) {
                for (int j = jj; j < jjmax; ++j) {
                    float sum = 0.0f;
                    for (int ki = 0; ki < K; ++ki) {
                        for (int kj = 0; kj < K; ++kj) {
                            int r = i + ki - pad;
                            int c = j + kj - pad;
                            if (r >= 0 && r < N && c >= 0 && c < N) {
                                sum += input[r * N + c] * kernel[ki * K + kj];
                            }
                        }
                    }
                    output[i * N + j] = sum;
                }
            }
        }
    }
}
