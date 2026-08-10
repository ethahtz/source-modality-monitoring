#!/bin/bash
# Section 4.1 -- purely symbolic binding with arbitrary labels (Figure 3).
#
# Replaces the modality markers with content-free labels, so success requires
# using the symbols as pure indexing devices. Both label orders are run so that
# the assignment of label to modality is counterbalanced.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

# "LabelA LabelB" means LabelA binds image content, LabelB binds caption content.
LABEL_PAIRS=${LABEL_PAIRS:-"Alpha:Beta Beta:Alpha Dax:Wug Wug:Dax"}
MODELS=("${FOCUS_MODELS[@]}")

for MODEL in "${MODELS[@]}"; do
  export MODEL
  short="${MODEL##*/}"
  for dataset in "${DATASETS[@]}"; do
    for pair in $LABEL_PAIRS; do
      l1="${pair%%:*}"; l2="${pair##*:}"
      for order in "${ORDERS[@]}"; do
        for modality in "${MODALITIES[@]}"; do
          submit "arb_${short}_${dataset}_${l1}_${l2}_${order}_${modality}" \
            prompt_evaluation_arbitrary_labels.py \
            --seed "$SEED" --work_dir "$WORK_DIR" --version "$VERSION" \
            --model_name "$MODEL" --dataset "$dataset" --split "$SPLIT" \
            --input_type inconsistent --modality_to_report "$modality" \
            --order "$order" --label_1 "$l1" --label_2 "$l2"
        done
      done
    done
  done
done
