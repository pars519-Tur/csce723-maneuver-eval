#!/bin/bash
# The finite-burn arm: the same delta-v spread over a burn duration instead of applied at once.
#
# Electric propulsion does not deliver its delta-v in an instant, and a spread burn produces a
# smaller, later-arriving offset, so it is the harder detection case. The plan is otherwise the
# treatment's own: same cohort, same magnitudes, same burn start times, same isotropic directions.
# Only the burn model differs, so this arm pairs directly against the impulsive `random` arm and
# against the `nullburn` control.
#
# Usage:  bash real_environment/run_finite_burn_grid.sh [seeds] [duration_seconds] [parallel_jobs]

set -u
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

SEEDS="${1:-100}"
DURATION="${2:-1800}"
JOBS="${3:-2}"
PYTHON=/opt/anaconda3/envs/marlssa/bin/python
OUTPUT_DIR="$PROJECT_ROOT/results/noise_maneuver_grid"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

export PYTHONPATH="$PROJECT_ROOT"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

for policy in advanced_greedy frozen_cnnv2_ppo; do
  arm="${policy}_finite${DURATION}s_sigma10_${SEEDS}seeds"
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 2; done
  (
    "$PYTHON" -m real_environment.evaluation_runner \
      --policy "$policy" --condition maneuver --seeds "$SEEDS" \
      --maneuvering-fraction 0.20 --random-maneuvers \
      --burn-duration-seconds "$DURATION" \
      --sigma-km 10.0 --record-nis --resume \
      --output "$OUTPUT_DIR/${arm}.p" > "$LOG_DIR/${arm}.log" 2>&1
    echo "finished $arm (exit $?)"
  ) &
done
echo "Launched 2 finite-burn arms at ${DURATION} s, ${SEEDS} seeds each."
wait
echo "Finite-burn grid complete."
