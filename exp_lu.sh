#!/bin/bash
set -euo pipefail

# Compilation
clang -O3 -march=native -std=c11 -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp \
    lu_tiling.c \
    lu_runner.c \
    lu_core.c \
    matmul_utils.c \
    -o lu

# Run
./lu 1024 64 3 3
