#!/bin/bash
# Burn direction as its own factor, at sigma = 10 km.
#
# A transverse burn changes the semi-major axis and drifts secularly; radial and normal burns
# perturb the orbit periodically and stay bounded. Averaging them under one isotropic draw hides
# that, so each axis gets its own arm.
#
# The existing nullburn arms are the matched control for all three: magnitude_scale = 0 makes the
# direction irrelevant, and the cohort and burn times come from the same draw in every mode.
#
# Usage:  bash real_environment/run_direction_grid.sh [seeds] [parallel_jobs]

set -u
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

SEEDS="${1:-100}"
JOBS="${2:-6}"
PYTHON=/opt/anaconda3/envs/marlssa/bin/python
OUTPUT_DIR="$PROJECT_ROOT/results/noise_maneuver_grid"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

export PYTHONPATH="$PROJECT_ROOT"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

run_arm() {
  local policy="$1" direction="$2" arm="$3"
  "$PYTHON" -m real_environment.evaluation_runner \
    --policy "$policy" --condition maneuver --seeds "$SEEDS" \
    --maneuvering-fraction 0.20 --random-maneuvers --direction-mode "$direction" \
    --sigma-km 10.0 --record-nis --resume \
    --output "$OUTPUT_DIR/${arm}.p" > "$LOG_DIR/${arm}.log" 2>&1
  echo "finished $arm (exit $?)"
}

launched=0
for policy in advanced_greedy frozen_cnnv2_ppo; do
  for direction in radial transverse normal; do
    arm="${policy}_${direction}_sigma10_${SEEDS}seeds"
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 2; done
    run_arm "$policy" "$direction" "$arm" &
    launched=$((launched + 1))
  done
done
echo "Launched $launched direction arms, ${SEEDS} seeds each."
wait
echo "Direction grid complete."
