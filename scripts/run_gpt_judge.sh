#!/bin/bash
# LLM-judge scoring of saved model outputs (Section 3.3, Appendices C, D and F).
#
# Each raw run under results/ is scored by GPT-5.4-mini, which decides whether a
# response is grounded in the image, the caption, or neither. Judged files are
# written to results_gpt_eval/ mirroring the input layout.
#
# Requires OPENAI_API_KEY (or OPENROUTER_API_KEY with --openrouter) in .env.
# This stage is CPU-only -- it just calls the API.
#
# Usage:
#   ./scripts/run_gpt_judge.sh                          # judge everything under results/
#   JUDGE_SCRIPT=gpt_judge_evaluation_arb.py ./scripts/run_gpt_judge.sh
#   RESULTS_SUBDIR=behavioral_evaluation/modification_none ./scripts/run_gpt_judge.sh

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

JUDGE_SCRIPT="${JUDGE_SCRIPT:-gpt_judge_evaluation.py}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-behavioral_evaluation}"
USE_OPENROUTER="${USE_OPENROUTER:-0}"

extra=()
[[ "$USE_OPENROUTER" == "1" ]] && extra+=(--openrouter)

search_root="$WORK_DIR/results/$RESULTS_SUBDIR"
if [[ ! -d "$search_root" ]]; then
  echo "No raw results at $search_root -- run the evaluation scripts first." >&2
  exit 1
fi

count=0
while IFS= read -r -d '' input_json; do
  count=$((count + 1))
  echo "[judge $count] ${input_json#$WORK_DIR/}"
  "$PYTHON" "$WORK_DIR/src/$JUDGE_SCRIPT" \
    --work_dir "$WORK_DIR" --input_json "$input_json" "${extra[@]}"
done < <(find "$search_root" -name "*.json" -print0 | sort -z)

echo "judged $count file(s)"
