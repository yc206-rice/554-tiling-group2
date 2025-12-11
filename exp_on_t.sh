#!/bin/bash
set -euo pipefail

n=${1:-1024}
reps=${2:-3}
impl=${3:-3}
resfile=${4:-results_t.csv}
t_values=${T_VALUES:-"16 24 32 48 64 80 96 112 128 160 192 224 256"}

clang -O3 -march=native -std=c11 -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp matmul_tiling.c matmul_runner.c matmul_core.c matmul_utils.c -o matmul

echo "N,T,reps,impl,time_s,gflop_per_s" > "${resfile}"

for t in ${t_values}; do
    logfile=$(mktemp)
    ./matmul "${n}" "${t}" "${reps}" "${impl}" > "${logfile}"

    header=$(head -n1 "${logfile}")
    IFS=',' read -r n_field t_field reps_field impl_field <<< "${header}"
    n_val=${n_field#*=}
    t_val=${t_field#*=}
    reps_val=${reps_field#*=}
    impl_val=${impl_field#*=}

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
