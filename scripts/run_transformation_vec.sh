#!/bin/bash
# Section 5 -- learned transformation vectors that induce source misattribution.
#
# Trains two vectors (delta_1, delta_2) added to either the marker spans or the
# content spans at a given relative layer depth, optimized to make the model
# report the non-queried modality. Produces Figures 8, 14, 15 and 16.
#
#   ./scripts/run_transformation_vec.sh train
#   ./scripts/run_transformation_vec.sh eval
#
# Interventions: marker and content are the conditions of interest;
# baseline_first and baseline_last are the Appendix L.2 position baselines.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

STAGE="${1:-train}"
INTERVENTIONS="${INTERVENTIONS:-marker content baseline_first baseline_last}"
LAYER_DEPTHS="${LAYER_DEPTHS:-0.0 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1.0}"
TRAIN_SEEDS="${TRAIN_SEEDS:-42 43 44}"   # 3 random seeds, as in the paper
DATASET="${DATASET:-mscoco}"             # vectors are trained on MSCOCO train
MODELS=("${FOCUS_MODELS[@]}")

for MODEL in "${MODELS[@]}"; do
  export MODEL
  short="${MODEL##*/}"
  for interv in $INTERVENTIONS; do
    for depth in $LAYER_DEPTHS; do
      for seed in $TRAIN_SEEDS; do
        if [[ "$STAGE" == "train" ]]; then
          submit "train_${short}_${interv}_d${depth}_s${seed}" \
            train_transformation_vec.py \
            --model_name "$MODEL" --dataset "$DATASET" --work_dir "$WORK_DIR" \
            --seed "$seed" --span_type "$interv" --layer_depth "$depth"
        else
          submit "eval_${short}_${interv}_d${depth}_s${seed}" \
            eval_transformation_vec.py \
            --model_name "$MODEL" --dataset "$DATASET" --work_dir "$WORK_DIR" \
            --train_seed "$seed" --intervention "$interv" --layer_depth "$depth"
        fi
      done
    done
  done
done
