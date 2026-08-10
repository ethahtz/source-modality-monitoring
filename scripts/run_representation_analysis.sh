#!/bin/bash
# Section 4.2 / Appendix H -- distributional separation of image vs text tokens.
#
# Measures within- and cross-modality cosine similarity at the embedding layer
# and fits a linear probe (3-fold CV) to classify token modality, with a
# shuffled-label control. Produces Table 6.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

SPAN_TYPES="${SPAN_TYPES:-content}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"
MODELS=("${ALL_MODELS[@]}")

for MODEL in "${MODELS[@]}"; do
  export MODEL
  short="${MODEL##*/}"
  for dataset in "${DATASETS[@]}"; do
    for span in $SPAN_TYPES; do
      submit "repr_${short}_${dataset}_${span}" \
        representation_analysis.py \
        --seed "$SEED" --work_dir "$WORK_DIR" \
        --model_name "$MODEL" --dataset "$dataset" \
        --span_type "$span" --num_samples "$NUM_SAMPLES" --layer_idx 0
    done
  done
done
