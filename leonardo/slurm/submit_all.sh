#!/usr/bin/env bash
# Submit the full pipeline (annotate -> bins -> deterministic prompts -> LLM prompts)
# for one or both datasets, chaining stages via --dependency=afterok.
#
# Usage:
#   bash leonardo/slurm/submit_all.sh female
#   bash leonardo/slurm/submit_all.sh male
#   bash leonardo/slurm/submit_all.sh both
#   bash leonardo/slurm/submit_all.sh greek_female_tts
#   bash leonardo/slurm/submit_all.sh greek_male_tts
#   bash leonardo/slurm/submit_all.sh greek_tts
#
# Optional flags via env vars:
#   SKIP_STAGE_10=1   skip annotation (use existing tags)
#   SKIP_STAGE_20=1   skip bin mapping
#   SKIP_STAGE_30=1   skip deterministic prompt creation
#   SKIP_STAGE_40=1   skip LLM prompt creation
set -euo pipefail

HERE_REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$HERE_REPO"
# shellcheck disable=SC1091
source leonardo/env.sh

target="${1:-}"
case "$target" in
  female|male|greek_female_tts|greek_male_tts) DATASETS=("$target") ;;
  both)        DATASETS=(female male) ;;
  greek_tts)   DATASETS=(greek_female_tts greek_male_tts) ;;
  *)           echo "usage: $0 {female|male|both|greek_female_tts|greek_male_tts|greek_tts}" >&2; exit 2 ;;
esac

# Sanity: input dataset(s) must exist on disk.
for ds in "${DATASETS[@]}"; do
  IN_DIR="$(hf_dataset_dir_for "$ds")"
  if [ ! -d "$IN_DIR" ]; then
    echo "[submit] missing HF dataset for $ds at $IN_DIR" >&2
    echo "[submit] run on a login node first:" >&2
    echo "         python leonardo/login/03_build_hf_dataset.py --dataset $ds" >&2
    exit 1
  fi
done

mkdir -p leonardo/logs

submit_chain() {
  local ds="$1"
  local prev=""
  local jid=""

  echo
  echo "=== submitting chain for DATASET=$ds ==="

  # Stage 10: annotate (GPU)
  if [ "${SKIP_STAGE_10:-0}" != "1" ]; then
    if [ -n "$prev" ]; then dep="--dependency=afterok:$prev"; else dep=""; fi
    jid=$(sbatch --parsable $dep \
      --job-name="dspch-annotate-$ds" \
      --export=ALL,DATASET="$ds" \
      leonardo/slurm/10_annotate.slurm)
    echo "  10_annotate  -> $jid"
    prev="$jid"
  fi

  # Stage 20: metadata_to_text
  if [ "${SKIP_STAGE_20:-0}" != "1" ]; then
    if [ -n "$prev" ]; then dep="--dependency=afterok:$prev"; else dep=""; fi
    jid=$(sbatch --parsable $dep \
      --job-name="dspch-bins-$ds" \
      --export=ALL,DATASET="$ds" \
      leonardo/slurm/20_metadata_to_text.slurm)
    echo "  20_bins      -> $jid"
    prev="$jid"
  fi

  # Stage 30 and 40 can run in parallel after stage 20 (they both depend on text-tags).
  local after_bins="$prev"
  local jid_det="" jid_llm=""

  if [ "${SKIP_STAGE_30:-0}" != "1" ]; then
    if [ -n "$after_bins" ]; then dep="--dependency=afterok:$after_bins"; else dep=""; fi
    jid_det=$(sbatch --parsable $dep \
      --job-name="dspch-det-$ds" \
      --export=ALL,DATASET="$ds" \
      leonardo/slurm/30_prompt_deterministic.slurm)
    echo "  30_det       -> $jid_det"
  fi

  if [ "${SKIP_STAGE_40:-0}" != "1" ]; then
    if [ -n "$after_bins" ]; then dep="--dependency=afterok:$after_bins"; else dep=""; fi
    jid_llm=$(sbatch --parsable $dep \
      --job-name="dspch-llm-$ds" \
      --export=ALL,DATASET="$ds" \
      leonardo/slurm/40_prompt_llm.slurm)
    echo "  40_llm       -> $jid_llm"
  fi
}

for ds in "${DATASETS[@]}"; do
  submit_chain "$ds"
done

echo
echo "All chains submitted. Watch with: squeue -u \$USER"
