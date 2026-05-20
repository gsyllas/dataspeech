#!/usr/bin/env bash
# Build the named-speaker variant into $NAMED_OUT_ROOT.
#
# This keeps the existing anonymous/gender-aware outputs in $OUT_ROOT intact.
# The input is only multi_speaker_combined: it already contains the standalone
# female/male speakers plus the additional speakers. It is filtered to speakers
# with at least $MULTI_MIN_SPEAKER_HOURS hours and writes a speaker_id -> name
# JSON for the named prompt stages.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/../env.sh"

activate_conda_env

export OUT_ROOT="$NAMED_OUT_ROOT"

echo "[named-build] OUT_ROOT=$OUT_ROOT"
echo "[named-build] multi source=$MULTI_SPEAKER_DIR"
echo "[named-build] min speaker hours=$MULTI_MIN_SPEAKER_HOURS"
echo "[named-build] speaker names JSON=$NAMED_MULTI_SPEAKER_NAMES_JSON"

python "$HERE/03_build_hf_dataset.py" \
  --dataset multi \
  --multi-min-speaker-hours "$MULTI_MIN_SPEAKER_HOURS" \
  --speaker-names-json "$NAMED_MULTI_SPEAKER_NAMES_JSON"

echo "[named-build] DONE. Submit with:"
echo "              bash \"$REPO_ROOT/leonardo/slurm/submit_named.sh\""
