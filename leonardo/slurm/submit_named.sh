#!/usr/bin/env bash
# Submit the named-speaker variant into $NAMED_OUT_ROOT.
#
# Usage:
#   bash leonardo/slurm/submit_named.sh
#   bash leonardo/slurm/submit_named.sh multi
#   bash leonardo/slurm/submit_named.sh multi_v2
#
# Optional flags via env vars:
#   SKIP_STAGE_10=1   skip annotation (use existing named tags)
#   SKIP_STAGE_20=1   skip bin mapping
#   SKIP_STAGE_30=1   skip named deterministic prompt creation
#   SKIP_STAGE_40=1   skip named LLM prompt creation
set -euo pipefail

HERE_REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$HERE_REPO"
# shellcheck disable=SC1091
source leonardo/env.sh

export OUT_ROOT="$NAMED_OUT_ROOT"

target="${1:-multi}"
case "$target" in
  multi|multi_v2) DATASETS=("$target") ;;
  *)              echo "usage: $0 [multi|multi_v2]" >&2; exit 2 ;;
esac

for ds in "${DATASETS[@]}"; do
  IN_DIR="$(hf_dataset_dir_for "$ds")"
  if [ ! -d "$IN_DIR" ]; then
    echo "[submit-named] missing HF dataset for $ds at $IN_DIR" >&2
    echo "[submit-named] run on a login node first:" >&2
    echo "               bash leonardo/login/04_prepare_named_variant.sh $ds" >&2
    exit 1
  fi
done

mkdir -p leonardo/logs

submit_chain() {
  local ds="$1"
  local prev=""
  local dep=""
  local jid=""

  echo
  echo "=== submitting named chain for DATASET=$ds ==="
  echo "    OUT_ROOT=$OUT_ROOT"

  if [ "${SKIP_STAGE_10:-0}" != "1" ]; then
    if [ -n "$prev" ]; then dep="--dependency=afterok:$prev"; else dep=""; fi
    jid=$(sbatch --parsable $dep \
      --job-name="dspch-named-annotate-$ds" \
      --export=ALL,DATASET="$ds" \
      leonardo/slurm/10_annotate.slurm)
    echo "  10_annotate  -> $jid"
    prev="$jid"
  fi

  if [ "${SKIP_STAGE_20:-0}" != "1" ]; then
    if [ -n "$prev" ]; then dep="--dependency=afterok:$prev"; else dep=""; fi
    jid=$(sbatch --parsable $dep \
      --job-name="dspch-named-bins-$ds" \
      --export=ALL,DATASET="$ds" \
      leonardo/slurm/20_metadata_to_text.slurm)
    echo "  20_bins      -> $jid"
    prev="$jid"
  fi

  local after_bins="$prev"

  if [ "${SKIP_STAGE_30:-0}" != "1" ]; then
    if [ -n "$after_bins" ]; then dep="--dependency=afterok:$after_bins"; else dep=""; fi
    jid=$(sbatch --parsable $dep \
      --job-name="dspch-named-det-$ds" \
      --export=ALL,DATASET="$ds" \
      leonardo/slurm/30_prompt_named_deterministic.slurm)
    echo "  30_named_det -> $jid"
  fi

  if [ "${SKIP_STAGE_40:-0}" != "1" ]; then
    if [ -n "$after_bins" ]; then dep="--dependency=afterok:$after_bins"; else dep=""; fi
    jid=$(sbatch --parsable $dep \
      --job-name="dspch-named-llm-$ds" \
      --export=ALL,DATASET="$ds" \
      leonardo/slurm/40_prompt_named_llm.slurm)
    echo "  40_named_llm -> $jid"
  fi
}

for ds in "${DATASETS[@]}"; do
  submit_chain "$ds"
done

echo
echo "All named chains submitted. Watch with: squeue -u \$USER"
