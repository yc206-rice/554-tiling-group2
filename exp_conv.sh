#!/bin/bash
set -euo pipefail

# Compilation
clang -O3 -march=native -std=c11 -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp \
    convolution_tiling.c \
    convolution_runner.c \
    convolution_core.c \
    matmul_utils.c \
    -o convolution

# Run
./convolution 1024 64 3 3
