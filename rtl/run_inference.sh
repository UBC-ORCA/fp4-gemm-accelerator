#!/usr/bin/env bash
###############################################################################
# run_inference.sh - launch a CVE2 FP4 MNIST inference build under Verilator
#
#   ./run_inference.sh <version> [dataset] [options]
#
#   version : baseline | novec | hardware   (required)
#   dataset : 8 | 80 | 400 | 1k | 2k | 10k               (default: 80)
#
# Run ./run_inference.sh --help for the full option list.
###############################################################################
set -euo pipefail

# always run from this script's dir (rtl/) so the relative sim/hex/data paths
# resolve and uart_out.txt lands here, matching the manual launch command
cd "$(dirname "$(readlink -f "$0")")"

SIM=./build/openhwgroup_cve2_cve2_top_0.1/lint-verilator/Vcve2_top

# version -> software directory
declare -A VDIR=(
  [baseline]=inference_baseline
  [novec]=inference_novec
  [hardware]=inference_hardware
)

usage() {
  cat <<'EOF'
run_inference.sh - launch a CVE2 FP4 MNIST inference build under Verilator

  ./run_inference.sh <version> [dataset] [size] [options]

  version : baseline | novec | hardware    (required)
  dataset : mnist | fashion                (default: mnist)
  size    : 8 | 80 | 400 | 1k | 2k | 10k   (default: 80 | UPDATE IN C)

Run ./run_inference.sh --help for the full option list.

Versions:
  baseline   FP4 read as signed int4, no accel (inaccurate, speed reference)
  novec      scalar + MAC array, vector length 1 (one word per vle32)
  hardware   scalar + vector + MAC array, full length vector loads

Options:
  --traces          enable instruction and data traces
  --trace-if        instruction-fetch trace only
  --trace-d         data trace only
  --print-every N   TB status interval in cycles   (default: 5000000)
  --quiet           minimal TB output (huge print-every, traces off)
  --max-cycles N    cycle cap                       (default: 5e14)
  --no-uart         do not write uart_out.txt (still prints to stdout)
  --save [FILE]     tee stdout to FILE (default: <version>_<dataset>.log)
  -y, --yes         skip the dataset / N_SAMPLES confirmation prompt
  -h, --help        show this help
EOF
}

# defaults
VERSION=""
DATASET=mnist
SIZE=80
TRACE_IF=0
TRACE_D=0
PRINT_EVERY=5000000
MAX_CYCLES=500000000000000
NO_UART=0
SAVE=0
LOGFILE=""
YES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    baseline|novec|hardware) VERSION="$1" ;;
    mnist|fashion)     DATASET="$1" ;;
    8|test_8|test8)          SIZE=8 ;;
    80|test_80|test80)          SIZE=80 ;;
    400|test_400|test400)          SIZE=400 ;;
    1k|test_1k|test1k)          SIZE=1k ;;
    2k|test_2k|test2k)          SIZE=2k ;;
    10k|test_10k|test10k)       SIZE=10k ;;
    --traces)      TRACE_IF=1; TRACE_D=1 ;;
    --trace-if)    TRACE_IF=1 ;;
    --trace-d)     TRACE_D=1 ;;
    --print-every) shift; PRINT_EVERY="${1:?--print-every needs a value}" ;;
    --quiet)       PRINT_EVERY=1000000000000; TRACE_IF=0; TRACE_D=0 ;;
    --no-uart)     NO_UART=1 ;;
    --max-cycles)  shift; MAX_CYCLES="${1:?--max-cycles needs a value}" ;;
    -y|--yes)      YES=1 ;;
    --save)        SAVE=1
                   # optional filename may follow (not another flag / positional)
                   if [[ ${2:-} && ${2:0:1} != "-" ]]; then LOGFILE="$2"; shift; fi ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "error: unknown argument '$1'" >&2; usage; exit 1 ;;
  esac
  shift
done

if [[ -z "$VERSION" ]]; then
  echo "error: version required (baseline|novec|hardware)" >&2
  usage; exit 1
fi

DIR="${VDIR[$VERSION]}"
HEX="../sw/$DIR/inference.hex"
DATA="../sw/headers/${DATASET}/test_${SIZE}.bin"

if [[ $YES -eq 0 && -t 0 ]]; then
  case "$SIZE" in 8) N=8 ;; 80) N=80 ;; 400) N=400 ;; 1k) N=1000 ;; 2k) N=2000 ;; 10k) N=10000 ;; *) N="$SIZE" ;; esac
  echo "version: $VERSION"
  echo "dataset: $DATASET"
  echo "dataset size (check c code): $N"
  read -rp "confirm [y/n] " ans
  [[ "$ans" == [yY] || "$ans" == [yY][eE][sS] ]] || { echo "aborted"; exit 1; }
fi

# preflight checks with actionable messages
[[ -x "$SIM"  ]] || { echo "error: sim binary not found: $SIM" >&2
                      echo "       build the Verilator model first (make -f sim.mk build-sim)" >&2; exit 1; }
[[ -f "$HEX"  ]] || { echo "error: hex not found: $HEX  (build it: make -C ../sw/$DIR -f inference.mk)" >&2; exit 1; }
[[ -f "$DATA" ]] || { echo "error: dataset not found: $DATA" >&2; exit 1; }

# assemble the launch command
ARGS=("$HEX" --data "$DATA" --max-cycles "$MAX_CYCLES" --print-every "$PRINT_EVERY")
[[ $TRACE_IF -eq 1 ]] && ARGS+=(--trace-if)
[[ $TRACE_D  -eq 1 ]] && ARGS+=(--trace-d)
[[ $NO_UART  -eq 1 ]] && ARGS+=(--no-uart)

echo "[run] version=$VERSION  dataset=$DATASET size=test_${SIZE}.bin  traces=if:$TRACE_IF,d:$TRACE_D  print-every=$PRINT_EVERY"
echo "[run] $SIM ${ARGS[*]}"

if [[ $SAVE -eq 1 ]]; then
  [[ -n "$LOGFILE" ]] || LOGFILE="${VERSION}_${DATASET}.log"
  echo "[run] tee -> $LOGFILE   (uart_out.txt also written by the TB)"
  "$SIM" "${ARGS[@]}" 2>&1 | tee "$LOGFILE"
else
  "$SIM" "${ARGS[@]}"
fi
