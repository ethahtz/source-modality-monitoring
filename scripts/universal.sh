#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#
# Generic job wrapper: runs one script from src/ with the arguments it is given.
#
#   ./scripts/universal.sh prompt_evaluation.py --model_name ... --dataset ...
#
# Resource requests (partition, GPUs, time, memory) are passed by the caller on
# the sbatch command line; see scripts/config.sh. To get failure emails, add
#   #SBATCH --mail-type=FAIL
#   #SBATCH --mail-user=you@example.edu
# to this file.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${WORK_DIR:-$REPO_ROOT}"
PYTHON="${PYTHON:-python}"

# Load a CUDA module if the cluster provides one; harmless elsewhere.
if command -v module >/dev/null 2>&1; then
  module load cuda 2>/dev/null || true
fi

cd "$REPO_ROOT/src"

echo "[universal.sh] host=$(hostname) work_dir=$WORK_DIR"
echo "[universal.sh] $PYTHON $*"
exec "$PYTHON" "$@"
