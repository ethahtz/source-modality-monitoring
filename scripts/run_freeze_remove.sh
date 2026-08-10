#!/bin/bash
# Section 4.4 -- the freeze-remove (Frz-RM) intervention (Figures 6 and 11).
#
# Collects contextualized content-token activations from a clean run with
# markers intact, then patches them into a run with the markers removed. Tests
# whether marker information has propagated into content-token representations.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

PROMPT_FORMATS="${PROMPT_FORMATS:-image_caption}"
MODELS=("${FOCUS_MODELS[@]}")

for MODEL in "${MODELS[@]}"; do
  export MODEL
  short="${MODEL##*/}"
  for dataset in "${DATASETS[@]}"; do
    for fmt in $PROMPT_FORMATS; do
      for order in "${ORDERS[@]}"; do
        for modality in "${MODALITIES[@]}"; do
          submit "frz_${short}_${dataset}_${fmt}_${order}_${modality}" \
            freeze_content_remove.py \
            --seed "$SEED" --work_dir "$WORK_DIR" --version "$VERSION" \
            --model_name "$MODEL" --dataset "$dataset" --split "$SPLIT" \
            --prompt_format "$fmt" --input_type inconsistent \
            --modality_to_report "$modality" --order "$order"
        done
      done
    done
  done
done
