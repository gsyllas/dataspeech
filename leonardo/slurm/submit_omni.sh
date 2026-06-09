#!/usr/bin/env bash
# Submit the STANDALONE Qwen2.5-Omni prompt path (stage 45).
#
# This path does not use the tag pipeline (no annotate / bins / deterministic).
# It only needs 01_hf_dataset to exist. It hands the audio to Qwen2.5-Omni and
# writes 04c_prompts_omni next to the other prompt stages.
#
# Usage:
#   bash leonardo/slurm/submit_omni.sh greek_female_tts
#   bash leonardo/slurm/submit_omni.sh greek_male_tts
#   bash leonardo/slurm/submit_omni.sh greek_tts        # both of the above
#   bash leonardo/slurm/submit_omni.sh female
#   bash leonardo/slurm/submit_omni.sh male
#   bash leonardo/slurm/submit_omni.sh multi_v2         # uses NAMED_OUT_ROOT
#
# Override the model / prompt before submitting, e.g.:
#   export OMNI_MODEL_ID="Qwen/Qwen2.5-Omni-3B"
#   export OMNI_PROMPT_STYLE="rich"
set -euo pipefail

HERE_REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$HERE_REPO"
# shellcheck disable=SC1091
source leonardo/env.sh

target="${1:-}"
case "$target" in
  female|male|greek_female_tts|greek_male_tts) DATASETS=("$target") ;;
  greek_tts) DATASETS=(greek_female_tts greek_male_tts) ;;
  multi|multi_v2)
    DATASETS=("$target")
    # Named bundles live under NAMED_OUT_ROOT; the SLURM job honors OUT_ROOT.
    export OUT_ROOT="$NAMED_OUT_ROOT"
    ;;
  *)
    echo "usage: $0 {female|male|greek_female_tts|greek_male_tts|greek_tts|multi|multi_v2}" >&2
    exit 2
    ;;
esac

mkdir -p leonardo/logs

for ds in "${DATASETS[@]}"; do
  IN_DIR="$(hf_dataset_dir_for "$ds")"
  if [ ! -d "$IN_DIR" ]; then
    echo "[submit-omni] missing HF dataset for $ds at $IN_DIR" >&2
    echo "[submit-omni] build it first on a login node:" >&2
    if [ "$ds" = "multi_v2" ] || [ "$ds" = "multi" ]; then
      echo "              bash leonardo/login/04_prepare_named_variant.sh $ds" >&2
    else
      echo "              python leonardo/login/03_build_hf_dataset.py --dataset $ds" >&2
    fi
    exit 1
  fi
done

for ds in "${DATASETS[@]}"; do
  echo
  echo "=== submitting Omni path for DATASET=$ds (OUT_ROOT=$OUT_ROOT) ==="
  jid=$(sbatch --parsable \
    --job-name="dspch-omni-$ds" \
    --export=ALL,DATASET="$ds",OUT_ROOT="$OUT_ROOT" \
    leonardo/slurm/45_prompt_omni.slurm)
  echo "  45_omni -> $jid"
done

echo
echo "Submitted. Watch with: squeue -u \$USER"
