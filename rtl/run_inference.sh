#!/usr/bin/env bash
###############################################################################
# run_inference.sh - launch a CVE2 INT4 MNIST inference build under Verilator
#
#   ./run_inference.sh [dataset] [size] [options]
#
#   dataset : mnist | fashion                (default: mnist)
#   size    : 8 | 80 | 400 | 1k | 2k | 10k   (default: 80)
#
# Run ./run_inference.sh --help for the full option list.
###############################################################################
set -euo pipefail

# always run from this script's dir (rtl/) so the relative sim/hex/data paths
# resolve and uart_out.txt lands here, matching the manual launch command
cd "$(dirname "$(readlink -f "$0")")"

SIM=./build/openhwgroup_cve2_cve2_top_0.1/lint-verilator/Vcve2_top
DIR=inference_int4

usage() {
  cat <<'EOF'
run_inference.sh - launch a CVE2 INT4 MNIST inference build under Verilator

  ./run_inference.sh [dataset] [size] [options]

  dataset : mnist | fashion                (default: mnist)
  size    : 8 | 80 | 400 | 1k | 2k | 10k   (default: 80 | UPDATE IN C)

Options:
  --traces          enable instruction and data traces
  --trace-if        instruction-fetch trace only
  --trace-d         data trace only
  --print-every N   TB status interval in cycles   (default: 5000000)
  --quiet           minimal TB output (huge print-every, traces off)
  --stall-after N   abort + dump state after N cycles with no UART output
  --hier-depth N    wave dump hierarchy depth        (default: 2)
  --wave-file FILE  waveform output path (FST)   (default: cve2_top.fst)
  --no-wave         disable waveform dumping        (on by default)
  --uart-file FILE  UART tee filename           (default: derived from dataset dir)

  --max-cycles N    cycle cap                       (default: 5e14)
  --no-uart         do not write uart_out.txt (still prints to stdout)
  --save [FILE]     tee stdout to FILE (default: int4_<dataset>.log)
  -y, --yes         skip the dataset / N_SAMPLES confirmation prompt
  -h, --help        show this help
EOF
}

# defaults
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
STALL_AFTER=0
HIER_DEPTH=2
TRACE_WAVE=1
WAVE_FILE="cve2_top.fst"
UART_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    --stall-after) shift; STALL_AFTER="${1:?--stall-after needs a value}" ;;
    --hier-depth)  shift; HIER_DEPTH="${1:?--hier-depth needs a value}" ;;
    --wave-file)   shift; WAVE_FILE="${1:?--wave-file needs a value}" ;;
    --no-wave)     TRACE_WAVE=0 ;;
    --uart-file)   shift; UART_FILE="${1:?--uart-file needs a value}" ;;
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

HEX="../sw/$DIR/inference.hex"
DATA="../sw/headers/${DATASET}/test_${SIZE}_int4.bin"

if [[ $YES -eq 0 && -t 0 ]]; then
  case "$SIZE" in 8) N=8 ;; 80) N=80 ;; 400) N=400 ;; 1k) N=1000 ;; 2k) N=2000 ;; 10k) N=10000 ;; *) N="$SIZE" ;; esac
  echo "dataset: $DATASET"
  echo "dataset size: $N"
  read -rp "confirm [y/n] " ans
  [[ "$ans" == [yY] || "$ans" == [yY][eE][sS] ]] || { echo "aborted"; exit 1; }
fi

# preflight checks with actionable messages
[[ -x "$SIM"  ]] || { echo "error: sim binary not found: $SIM" >&2
                      echo "       build the Verilator model first (make -f sim.mk build-sim)" >&2; exit 1; }
[[ -f "$DATA" ]] || { echo "error: dataset not found: $DATA" >&2; exit 1; }

# rebuild the firmware so N_SAMPLES always matches the dataset being run
echo "[build] $DIR  dataset=$DATASET size=test_${SIZE}_int4.bin"
make -C "../sw/$DIR" -f inference.mk DATASET="$DATASET" SIZE="$SIZE" -B \
  || { echo "error: firmware build failed" >&2; exit 1; }

[[ -f "$HEX"  ]] || { echo "error: hex not found after build: $HEX" >&2; exit 1; }

# assemble the launch command
ARGS=("$HEX" --data "$DATA" --max-cycles "$MAX_CYCLES" --print-every "$PRINT_EVERY")
[[ $TRACE_IF -eq 1 ]] && ARGS+=(--trace-if)
[[ $TRACE_D  -eq 1 ]] && ARGS+=(--trace-d)
[[ $STALL_AFTER -ne 0 ]] && ARGS+=(--stall-after "$STALL_AFTER")
[[ -n "$HIER_DEPTH" ]] && ARGS+=(--hier_depth "$HIER_DEPTH")
if [[ $TRACE_WAVE -eq 1 ]]; then
  ARGS+=(--trace-wave --wave-file "$WAVE_FILE")
fi
[[ -n "$UART_FILE" ]] && ARGS+=(--uart-file "$UART_FILE")
[[ $NO_UART  -eq 1 ]] && ARGS+=(--no-uart)

WAVE_MSG="off"
[[ $TRACE_WAVE -eq 1 ]] && WAVE_MSG="$WAVE_FILE"
echo "[run] dataset=$DATASET size=test_${SIZE}_int4.bin  traces=if:$TRACE_IF,d:$TRACE_D  print-every=$PRINT_EVERY  wave=$WAVE_MSG"
echo "[run] $SIM ${ARGS[*]}"

if [[ $SAVE -eq 1 ]]; then
  [[ -n "$LOGFILE" ]] || LOGFILE="int4_${DATASET}.log"
  echo "[run] tee -> $LOGFILE   (uart_out.txt also written by the TB)"
  "$SIM" "${ARGS[@]}" 2>&1 | tee "$LOGFILE"
else
  "$SIM" "${ARGS[@]}"
fi
