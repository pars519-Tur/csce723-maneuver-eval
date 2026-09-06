#!/bin/bash
# Full factorial run behind the measurement-noise question.
#
#   measurement noise  sigma in {0, 10} km   0 is stock Smith, 10 matches the R that
#                                            SSAmulti_v2.py:306 already assumes
#   maneuver           {nullburn, random}    nullburn reuses the random plan with zero
#                                            magnitude, so cohort, burn times and burn
#                                            directions match the treatment exactly
#   policy             {advanced_greedy, frozen_cnnv2_ppo}
#
# Both maneuver arms tag the identical RSO cohort for a given seed, so every comparison is
# paired. Smith's files are never touched; noise and burns are injected by the wrapper.
#
# Usage:  bash real_environment/run_noise_maneuver_grid.sh [seeds] [parallel_jobs]
# Re-running resumes each arm from its own checkpoint.

set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

SEEDS="${1:-100}"
JOBS="${2:-4}"
PYTHON=/opt/anaconda3/envs/marlssa/bin/python
OUTPUT_DIR="$PROJECT_ROOT/results/noise_maneuver_grid"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

export PYTHONPATH="$PROJECT_ROOT"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

run_arm() {
  local policy="$1" sigma="$2" maneuver="$3" arm="$4"
  # The null-burn control reuses the treatment's random plan with zero magnitude, so the
  # tagged cohort, burn times and burn directions match and only the delta-v differs.
  local maneuver_flags="--random-maneuvers --delta-v-scale 0.0"
  if [ "$maneuver" = "random" ]; then
    maneuver_flags="--random-maneuvers"
  fi
  "$PYTHON" -m real_environment.evaluation_runner \
    --policy "$policy" --condition maneuver --seeds "$SEEDS" \
    --maneuvering-fraction 0.20 $maneuver_flags \
    --sigma-km "$sigma" --record-nis --resume \
    --output "$OUTPUT_DIR/${arm}.p" > "$LOG_DIR/${arm}.log" 2>&1
  echo "finished $arm (exit $?)"
}

launched=0
for policy in advanced_greedy frozen_cnnv2_ppo; do
  for sigma in 0.0 10.0; do
    sigma_tag=$(printf 'sigma%02.0f' "$sigma")
    for maneuver in nullburn random; do
      arm="${policy}_${maneuver}_${sigma_tag}_${SEEDS}seeds"
      # macOS ships bash 3.2, which has no `wait -n`, so poll the running-job count.
      while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 2; done
      run_arm "$policy" "$sigma" "$maneuver" "$arm" &
      launched=$((launched + 1))
    done
  done
done

echo "Launched $launched arms, ${SEEDS} seeds each, ${JOBS} at a time."
wait
echo "Grid complete. Outputs in $OUTPUT_DIR"
