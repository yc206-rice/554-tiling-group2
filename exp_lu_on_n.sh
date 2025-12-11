#!/bin/bash
set -euo pipefail

t=${1:-64}
reps=${2:-3}
impl=${3:-3}
resfile=${4:-results_lu_n.csv}
n_values=${N_VALUES:-"64 128 256 512 1024 2048"}

# Compile
clang -O3 -march=native -std=c11 -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp lu_tiling.c lu_runner.c lu_core.c matmul_utils.c -o lu

echo "N,T,reps,impl,time_s,gflop_per_s" > "${resfile}"

for n in ${n_values}; do
    logfile=$(mktemp)
    ./lu "${n}" "${t}" "${reps}" "${impl}" > "${logfile}"

    header=$(head -n1 "${logfile}")
    # Header format: LU: N=1024, T=64, reps=3, impl=3
    header_clean=${header#LU: }
    IFS=',' read -r n_field t_field reps_field impl_field <<< "${header_clean}"
    
    n_val=${n_field#*=}
    t_val=${t_field#*=}
    reps_val=${reps_field#*=}
    
    while IFS= read -r line; do
        if [[ ${line} =~ \[([a-z]+)\][[:space:]]+time=([0-9.]+)[[:space:]]*s[[:space:]]+GFLOP/s=([0-9.]+) ]]; then
            kernel=${BASH_REMATCH[1]}
            time_s=${BASH_REMATCH[2]}
            gflops=${BASH_REMATCH[3]}
            printf "%s,%s,%s,%s,%s,%s\n" "${n_val}" "${t_val}" "${reps_val}" "${kernel}" "${time_s}" "${gflops}" >> "${resfile}"
        fi
    done < "${logfile}"
    rm -f "${logfile}"
done
