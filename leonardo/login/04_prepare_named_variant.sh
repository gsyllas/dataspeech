#!/usr/bin/env bash
# Build a named-speaker variant into $NAMED_OUT_ROOT.
#
# This keeps the existing anonymous/gender-aware outputs in $OUT_ROOT intact.
# Use `multi` for the older multi_speaker_combined source, or `multi_v2` for
# the Greek-data bundle.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/../env.sh"

activate_conda_env

export OUT_ROOT="$NAMED_OUT_ROOT"
DATASET="${1:-multi}"
case "$DATASET" in
  multi|multi_v2) ;;
  *) echo "usage: $0 [multi|multi_v2]" >&2; exit 2 ;;
esac

SPEAKER_NAMES_JSON="$(named_speaker_names_json_for "$DATASET")"

echo "[named-build] OUT_ROOT=$OUT_ROOT"
if [ "$DATASET" = "multi_v2" ]; then
  echo "[named-build] multi_v2 root=$GREEK_DATA_ROOT"
  echo "[named-build] multi_v2 sources=${MULTI_V2_COMPONENTS:-commonVoice_greek_clean_with_speaker_ids,cs10_greek_dataset,greek_male_3.5h,greek_female_tts,greek_male_tts}"
else
  echo "[named-build] multi source=$MULTI_SPEAKER_DIR"
fi
echo "[named-build] min speaker hours=$MULTI_MIN_SPEAKER_HOURS"
echo "[named-build] speaker names JSON=$SPEAKER_NAMES_JSON"

python "$HERE/03_build_hf_dataset.py" \
  --dataset "$DATASET" \
  --multi-min-speaker-hours "$MULTI_MIN_SPEAKER_HOURS" \
  --speaker-names-json "$SPEAKER_NAMES_JSON"

echo "[named-build] DONE. Submit with:"
echo "              bash \"$REPO_ROOT/leonardo/slurm/submit_named.sh\" \"$DATASET\""
