#!/bin/bash
# Dedicated short runs for the per-component residual figure.
#
# The grid arms record NIS only, which is a scalar and loses the sign and the per-axis
# structure a residual plot needs. The recorder now also stores the signed innovation and
# the standard deviation the filter assigns to each of its components, so these runs exist
# purely to produce that. They are separate from the 22-arm grid and do not touch it.
#
# Two fixed along-track magnitudes, chosen from the measured dose response: one well above
# the detection floor and one below it. Everything else follows the primary plan.
#
# Usage:  bash real_environment/run_residual_traces.sh [seeds] [parallel_jobs]

set -u
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

SEEDS="${1:-10}"
JOBS="${2:-2}"
PYTHON=/opt/anaconda3/envs/marlssa/bin/python
OUTPUT_DIR="$PROJECT_ROOT/results/residual_traces"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

# A log is only truncated when its own arm starts, so an arm still queued behind the job
# limit keeps the previous run's log and reads as finished. Clear them all up front.
rm -f "$LOG_DIR"/*.log

export PYTHONPATH="$PROJECT_ROOT"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

for policy in advanced_greedy frozen_cnnv2_ppo; do
  for dv in 200 5; do
    arm="${policy}_alongtrack${dv}mps_sigma10_${SEEDS}seeds"
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 2; done
    (
      "$PYTHON" -m real_environment.evaluation_runner \
        --policy "$policy" --condition maneuver --seeds "$SEEDS" \
        --maneuvering-fraction 0.20 --random-maneuvers \
        --direction-mode transverse \
        --delta-v-min-mps "$dv" --delta-v-max-mps "$dv" \
        --sigma-km 10.0 --record-nis --resume \
        --output "$OUTPUT_DIR/${arm}.p" > "$LOG_DIR/${arm}.log" 2>&1
      echo "finished $arm (exit $?)"
    ) &
  done
done
echo "Launched 4 residual-trace arms at ${SEEDS} seeds each."
wait
echo "Residual traces complete."
