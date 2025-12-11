#include "matmul.h"

#include <stdlib.h>

static int clamp_impl(int impl) {
    if (impl < MATMUL_IMPL_NAIVE || impl > MATMUL_IMPL_ALL) {
        return MATMUL_IMPL_ALL;
    }
    return impl;
}

int main(int argc, char** argv) {
    matmul_config cfg = {
        .N = 1024,
        .T = 0,
        .reps = 3,
        .impl = MATMUL_IMPL_ALL
    };

    if (argc > 1) cfg.N = atoi(argv[1]);
    if (argc > 2) cfg.T = atoi(argv[2]);
    if (argc > 3) cfg.reps = atoi(argv[3]);
    if (argc > 4) cfg.impl = atoi(argv[4]);

    if (cfg.T <= 0) cfg.T = 64;
    if (cfg.reps < 1) cfg.reps = 1;
    cfg.impl = clamp_impl(cfg.impl);

    return run_benchmark(&cfg);
}
