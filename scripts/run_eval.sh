#!/usr/bin/env bash
# Run one case's evaluation sequence, detached, with a timestamped log.
#
#   scripts/run_eval.sh case04    0
#   scripts/run_eval.sh case05    1     # cuboid test on case_06 AND case_07
#   scripts/run_eval.sh cylinder  1
#
# Every invocation is checked afterwards for the "Loaded best validation model from"
# line; without it the outputs came from randomly initialised weights and are garbage.
set -uo pipefail

WHAT="${1:?usage: run_eval.sh <case04|case05|cylinder> <gpu_index>}"
GPU="${2:?usage: run_eval.sh <case04|case05|cylinder> <gpu_index>}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source /home/visharma/miniconda3/etc/profile.d/conda.sh
conda activate dem-dyngnet

mkdir -p logs
STAMP="$(date +%Y%m%d_%H%M%S)"
echo "$STAMP" > "logs/eval_${WHAT}.stamp"
LOG="logs/eval_${WHAT}_${STAMP}.log"
ERR="logs/eval_${WHAT}_${STAMP}.err"

export CUDA_VISIBLE_DEVICES="${GPU}"
export MPLBACKEND=Agg

run() {
  echo ">>> RUN: python -u $*"
  python -u "$@"
  echo ">>> EXIT=$? for: $*"
}

{
  echo "=== run_eval.sh ${WHAT} on GPU ${GPU} ==="
  case "$WHAT" in
    case04)
      run main_dem_simple.py --mode test --plot --save_data
      run main_dem_simple.py --mode benchmark_sphere_collisions --plot
      run main_dem_simple.py --mode benchmark_wall_collisions --plot
      ;;
    case05)
      run main_dem_hard.py --mode test --plot --save_data
      ;;
    cylinder)
      run main_dem_hard.py --mode cylinder --plot --save_data
      ;;
    *) echo "unknown target: $WHAT" >&2; exit 2 ;;
  esac
  echo "=== finished ==="
} 2> "${ERR}" | gawk '{print strftime("[%F %T]"), $0; fflush()}' > "${LOG}"
