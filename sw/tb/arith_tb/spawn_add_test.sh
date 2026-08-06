#!/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <num_processes> <binary>"
    echo "Example:"
    echo "  $0 16 rtl/build/unit_test/bf16_adder/obj_dir/Vparameterized_adder"
    exit 1
fi

NUM_PROCS="$1"
BIN="$2"

if [[ ! -x "$BIN" ]]; then
    echo "Error: '$BIN' does not exist or is not executable."
    exit 1
fi

mkdir -p logs

echo "Launching $NUM_PROCS processes..."

declare -a pids=()

for ((i=0; i<NUM_PROCS; i++)); do
    echo "  Worker $i"

    "$BIN" \
        --a-offset "$i" \
        --a-stride "$NUM_PROCS" \
        > "logs/worker_${i}.log" 2>&1 &
    pids[i]=$!
done

failed=()
for ((i=0; i<NUM_PROCS; i++)); do
    if ! wait "${pids[i]}"; then
        failed+=("$i")
    fi
done

if [[ ${#failed[@]} -eq 0 ]]; then
    echo "All workers completed. PASS"
    exit 0
else
    echo "FAIL: ${#failed[@]}/$NUM_PROCS worker(s) failed: ${failed[*]}"
    for i in "${failed[@]}"; do
        echo "  Worker $i log: logs/worker_${i}.log"
    done
    exit 1
fi
