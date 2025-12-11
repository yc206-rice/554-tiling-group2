#include "lu.h"
#include <stdlib.h>
#include <stdio.h>

static int clamp_impl(int impl) {
    if (impl < LU_IMPL_NAIVE || impl > LU_IMPL_ALL) {
        return LU_IMPL_ALL;
    }
    return impl;
}

int main(int argc, char** argv) {
    lu_config cfg = {
        .N = 1024,
        .T = 64,
        .reps = 3,
        .impl = LU_IMPL_ALL
    };

    if (argc > 1) cfg.N = atoi(argv[1]);
    if (argc > 2) cfg.T = atoi(argv[2]);
    if (argc > 3) cfg.reps = atoi(argv[3]);
    if (argc > 4) cfg.impl = atoi(argv[4]);

    cfg.impl = clamp_impl(cfg.impl);

    return run_lu_benchmark(&cfg);
}
