#!/usr/bin/env bash
# Clean already-generated LLM prompt descriptions for the Leonardo outputs.
set -euo pipefail

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN=1
fi

HERE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HERE_REPO"
# shellcheck disable=SC1091
source leonardo/env.sh

ARGS=()
if [ "$DRY_RUN" -eq 1 ]; then
  ARGS+=(--dry-run)
fi

python leonardo/login/07_clean_llm_descriptions.py --root "$OUT_ROOT" --dataset female "${ARGS[@]}"
python leonardo/login/07_clean_llm_descriptions.py --root "$OUT_ROOT" --dataset male "${ARGS[@]}"
python leonardo/login/07_clean_llm_descriptions.py --root "$OUT_ROOT" --dataset greek_tts "${ARGS[@]}"

python leonardo/login/07_clean_llm_descriptions.py \
  --root "$NAMED_OUT_ROOT" \
  --dataset multi \
  --speaker-names-json "$(named_speaker_names_json_for multi)" \
  "${ARGS[@]}"
python leonardo/login/07_clean_llm_descriptions.py \
  --root "$NAMED_OUT_ROOT" \
  --dataset multi_v2 \
  --speaker-names-json "$(named_speaker_names_json_for multi_v2)" \
  "${ARGS[@]}"
