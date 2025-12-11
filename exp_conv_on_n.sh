#!/bin/bash
set -euo pipefail

t=${1:-64}
reps=${2:-3}
impl=${3:-3}
k=${4:-3}
resfile=${5:-results_conv_n.csv}
n_values=${N_VALUES:-"64 128 256 512 1024 2048"}

# Compile
clang -O3 -march=native -std=c11 -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp convolution_tiling.c convolution_runner.c convolution_core.c matmul_utils.c -o convolution

echo "N,K,T,reps,impl,time_s,gflop_per_s" > "${resfile}"

for n in ${n_values}; do
    logfile=$(mktemp)
    ./convolution "${n}" "${t}" "${reps}" "${impl}" "${k}" > "${logfile}"

    header=$(head -n1 "${logfile}")
    # Header format: CONV: N=1024, K=3, T=64, reps=3, impl=2
    header_clean=${header#CONV: }
    IFS=',' read -r n_field k_field t_field reps_field impl_field <<< "${header_clean}"
    
    n_val=${n_field#*=}
    k_val=${k_field#*=}
    t_val=${t_field#*=}
    reps_val=${reps_field#*=}
    
    while IFS= read -r line; do
        if [[ ${line} =~ \[([a-z]+)\][[:space:]]+time=([0-9.]+)[[:space:]]*s[[:space:]]+GFLOP/s=([0-9.]+) ]]; then
            kernel=${BASH_REMATCH[1]}
            time_s=${BASH_REMATCH[2]}
            gflops=${BASH_REMATCH[3]}
            printf "%s,%s,%s,%s,%s,%s,%s\n" "${n_val}" "${k_val}" "${t_val}" "${reps_val}" "${kernel}" "${time_s}" "${gflops}" >> "${resfile}"
        fi
    done < "${logfile}"
    rm -f "${logfile}"
done
