#include "convolution.h"
#include <stdlib.h>
#include <stdio.h>

static int clamp_impl(int impl) {
    if (impl < CONV_IMPL_NAIVE || impl > CONV_IMPL_ALL) {
        return CONV_IMPL_ALL;
    }
    return impl;
}

int main(int argc, char** argv) {
    conv_config cfg = {
        .N = 1024,
        .K = 3,
        .T = 64,
        .reps = 3,
        .impl = CONV_IMPL_ALL
    };

    if (argc > 1) cfg.N = atoi(argv[1]);
    if (argc > 2) cfg.T = atoi(argv[2]);
    if (argc > 3) cfg.reps = atoi(argv[3]);
    if (argc > 4) cfg.impl = atoi(argv[4]);
    if (argc > 5) cfg.K = atoi(argv[5]);

    cfg.impl = clamp_impl(cfg.impl);

    return run_conv_benchmark(&cfg);
}
