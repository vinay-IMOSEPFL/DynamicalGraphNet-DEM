#!/usr/bin/env bash
# Launch one training run, fully detached, with a timestamped log.
#
#   scripts/run_training.sh case04 0
#   scripts/run_training.sh case05 1
#
# Arg 1: case04 | case05
# Arg 2: CUDA device index
#
# stdout is timestamped line-by-line into logs/train_<case>_<stamp>.log.
# stderr (tqdm progress bars, tracebacks) goes to the matching .err file so the
# carriage-return progress noise never corrupts the parseable log.
set -uo pipefail

CASE="${1:?usage: run_training.sh <case04|case05> <gpu_index>}"
GPU="${2:?usage: run_training.sh <case04|case05> <gpu_index>}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

case "$CASE" in
  case04) SCRIPT=main_dem_simple.py ;;
  case05) SCRIPT=main_dem_hard.py ;;
  *) echo "unknown case: $CASE" >&2; exit 2 ;;
esac

# shellcheck disable=SC1091
source /home/visharma/miniconda3/etc/profile.d/conda.sh
conda activate dem-dyngnet

mkdir -p logs
STAMP="$(date +%Y%m%d_%H%M%S)"
echo "$STAMP" > "logs/train_${CASE}.stamp"

LOG="logs/train_${CASE}_${STAMP}.log"
ERR="logs/train_${CASE}_${STAMP}.err"

{
  echo "=== run_training.sh ${CASE} on GPU ${GPU} ==="
  echo "host=$(hostname) pid=$$ script=${SCRIPT}"
  nvidia-smi --id="${GPU}" --query-gpu=name,driver_version,memory.total --format=csv,noheader
  CUDA_VISIBLE_DEVICES="${GPU}" MPLBACKEND=Agg python -u "${SCRIPT}" --mode train 2> "${ERR}"
  echo "PYTHON_EXIT=$?"
  echo "=== finished ==="
} | gawk '{print strftime("[%F %T]"), $0; fflush()}' > "${LOG}"
