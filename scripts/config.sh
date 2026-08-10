#!/bin/bash
# Shared configuration for the experiment launchers.
#
# Everything here can be overridden from the environment, e.g.
#   WORK_DIR=/scratch/$USER/smm PARTITION=gpu ./scripts/run_behavioral.sh
#
# The launchers submit one Slurm job per configuration via universal.sh. If you
# are not on a Slurm cluster, set SUBMIT="bash" to run the jobs serially in the
# foreground instead of calling sbatch.

# Repository root and where results/ will be written.
WORK_DIR="${WORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Python interpreter. Point this at the environment built from requirements.txt.
PYTHON="${PYTHON:-python}"

# Slurm resources. TIME and GPUS are per-model; see gpus_for() below.
PARTITION="${PARTITION:-gpu}"
TIME="${TIME:-5:00:00}"
MEM="${MEM:-32G}"
LOG_DIR="${LOG_DIR:-$WORK_DIR/slurm_logs}"

# How to launch a job: "sbatch" on a cluster, "bash" to run locally.
SUBMIT="${SUBMIT:-sbatch}"

# Fixed experimental constants from the paper.
SEED="${SEED:-42}"
VERSION="${VERSION:-1}"
SPLIT="${SPLIT:-test}"

# The 11 VLMs evaluated in Section 3.
ALL_MODELS=(
  "Qwen/Qwen2.5-VL-3B-Instruct"
  "Qwen/Qwen2.5-VL-7B-Instruct"
  "Qwen/Qwen2.5-VL-32B-Instruct"
  "google/gemma-3-4b-it"
  "google/gemma-3-12b-it"
  "google/gemma-3-27b-it"
  "OpenGVLab/InternVL3-8B-hf"
  "OpenGVLab/InternVL3-14B-hf"
  "llava-hf/llava-1.5-7b-hf"
  "llava-hf/llava-onevision-qwen2-7b-ov-hf"
  "Salesforce/instructblip-vicuna-7b"
)

# The three focus models used for the mechanistic analyses (Sections 4 and 5).
FOCUS_MODELS=(
  "Qwen/Qwen2.5-VL-32B-Instruct"
  "google/gemma-3-12b-it"
  "OpenGVLab/InternVL3-14B-hf"
)

DATASETS=("flickr30k" "mscoco")
ORDERS=("icq" "ciq")          # image-first / caption-first
MODALITIES=("image" "text")   # target modality

# GPU count per model, matching what the paper runs were given.
gpus_for() {
  case "$1" in
    Qwen/Qwen2.5-VL-32B-Instruct) echo 2 ;;
    google/gemma-3-27b-it)        echo 2 ;;
    OpenGVLab/InternVL3-14B-hf)   echo 2 ;;
    google/gemma-3-12b-it)        echo 1 ;;
    *)                            echo 1 ;;
  esac
}

# Submit one job. Usage: submit <job-name> <script.py> [args...]
# Set DRY_RUN=1 to print what would be submitted without running anything.
submit() {
  local name="$1"; shift
  local script="$1"; shift
  local gpus; gpus=$(gpus_for "${MODEL:-}")
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry-run gpu:$gpus] $script $*"
    return 0
  fi
  mkdir -p "$LOG_DIR"
  if [[ "$SUBMIT" == "sbatch" ]]; then
    sbatch --job-name="$name" \
           --output="$LOG_DIR/${name}_%j.out" \
           --partition="$PARTITION" \
           --gres="gpu:$gpus" \
           --time="$TIME" \
           --mem="$MEM" \
           "$(dirname "${BASH_SOURCE[0]}")/universal.sh" "$script" "$@"
  else
    echo "[run] $script $*"
    WORK_DIR="$WORK_DIR" PYTHON="$PYTHON" \
      bash "$(dirname "${BASH_SOURCE[0]}")/universal.sh" "$script" "$@"
  fi
}
