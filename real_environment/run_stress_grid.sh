#!/bin/bash
# Stress arm: burns large enough to carry an object out of the sensor's field of view.
#
# At GEO a 4 degree field of view is about 1467 km wide at the half-angle, and displacement runs
# d = 0.00101 * dV * t km. With burn times drawn over 600-4800 s, escape needs roughly 300 to 1500
# m/s depending on how much coast is left. The 1-100 m/s primary arm therefore cannot produce
# custody loss at all, by geometry. This arm sweeps 100-2000 m/s so the escape threshold is
# crossed inside the sample and a dose-response through it can be measured.
#
# These are orbit-transfer scale burns, not station keeping. The arm is a robustness stress test
# of a frozen policy outside its training regime, and must be reported separately from the
# primary condition.
#
# The cohort and burn times are identical to the primary and control arms, so nullburn remains
# the matched control.

set -u
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1
SEEDS="${1:-100}"
JOBS="${2:-2}"
PYTHON=/opt/anaconda3/envs/marlssa/bin/python
OUTPUT_DIR="$PROJECT_ROOT/results/noise_maneuver_grid"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
export PYTHONPATH="$PROJECT_ROOT" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

for policy in advanced_greedy frozen_cnnv2_ppo; do
  arm="${policy}_stress100to2000_sigma10_${SEEDS}seeds"
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 2; done
  (
    "$PYTHON" -m real_environment.evaluation_runner \
      --policy "$policy" --condition maneuver --seeds "$SEEDS" \
      --maneuvering-fraction 0.20 --random-maneuvers \
      --delta-v-min-mps 100 --delta-v-max-mps 2000 \
      --sigma-km 10.0 --record-nis --resume \
      --output "$OUTPUT_DIR/${arm}.p" > "$LOG_DIR/${arm}.log" 2>&1
    echo "finished $arm (exit $?)"
  ) &
done
echo "Launched 2 stress arms, ${SEEDS} seeds each."
wait
echo "Stress grid complete."
