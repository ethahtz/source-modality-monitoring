#!/bin/bash
# Sections 3 and 4.3 -- target-modality retrieval under each marker condition.
#
# Produces Figures 2, 4, 5, 12 and 13. Marker conditions follow Table 1:
#   none   unperturbed, original modality markers
#   remove both image and caption markers deleted
#   swap   image and caption markers exchanged
#
# Usage:
#   ./scripts/run_behavioral.sh                 # all 11 models, condition "none"
#   CONDITIONS="none remove swap" MODELS_SET=focus ./scripts/run_behavioral.sh
#   PROMPT_FORMATS="image_caption image_text image_document" ./scripts/run_behavioral.sh

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

CONDITIONS="${CONDITIONS:-none}"
PROMPT_FORMATS="${PROMPT_FORMATS:-image_caption}"
MODELS_SET="${MODELS_SET:-all}"
INPUT_TYPES="${INPUT_TYPES:-inconsistent}"

if [[ "$MODELS_SET" == "focus" ]]; then MODELS=("${FOCUS_MODELS[@]}"); else MODELS=("${ALL_MODELS[@]}"); fi

for MODEL in "${MODELS[@]}"; do
  export MODEL
  short="${MODEL##*/}"
  for dataset in "${DATASETS[@]}"; do
    for cond in $CONDITIONS; do
      for fmt in $PROMPT_FORMATS; do
        for input_type in $INPUT_TYPES; do
          for order in "${ORDERS[@]}"; do
            for modality in "${MODALITIES[@]}"; do
              submit "behv_${short}_${dataset}_${cond}_${fmt}_${order}_${modality}" \
                prompt_evaluation.py \
                --seed "$SEED" --work_dir "$WORK_DIR" --version "$VERSION" \
                --model_name "$MODEL" --dataset "$dataset" --split "$SPLIT" \
                --prompt_format "$fmt" --input_type "$input_type" \
                --modality_to_report "$modality" --order "$order" \
                --modify_inputs "$cond"
            done
          done
        done
      done
    done
  done
done
