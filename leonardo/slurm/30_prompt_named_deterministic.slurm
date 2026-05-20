#!/bin/bash
#SBATCH --job-name=dspch-named-det
#SBATCH --account=EUHPC_D29_081
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=leonardo/logs/%x_%j.out
#SBATCH --error=leonardo/logs/%x_%j.err
#
# Stage 30, named variant: deterministic prompt creation with speaker names.
set -euo pipefail
module purge
module load cuda/12.2

HERE_REPO="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
cd "$HERE_REPO"
# shellcheck disable=SC1091
source leonardo/env.sh

: "${DATASET:?DATASET must be set to 'female', 'male', or 'multi'}"
activate_conda_env
set_offline_mode

IN_DIR="$(text_tags_dir_for "$DATASET")"
OUT_DIR="$(prompts_det_dir_for "$DATASET")"
mkdir -p "$OUT_DIR" "leonardo/logs"

PROMPT_ARGS=(--is_new_speaker_prompt)
case "$DATASET" in
  female)
    PROMPT_ARGS+=(--is_single_speaker --speaker_name "$NAMED_FEMALE_SPEAKER_NAME")
    ;;
  male)
    PROMPT_ARGS+=(--is_single_speaker --speaker_name "$NAMED_MALE_SPEAKER_NAME")
    ;;
  multi)
    if [ ! -f "$NAMED_MULTI_SPEAKER_NAMES_JSON" ]; then
      echo "[named-det] missing speaker names JSON: $NAMED_MULTI_SPEAKER_NAMES_JSON" >&2
      echo "[named-det] run leonardo/login/04_prepare_named_variant.sh first" >&2
      exit 1
    fi
    PROMPT_ARGS+=(--speaker_id_column "speaker_id" --speaker_ids_to_name_json "$NAMED_MULTI_SPEAKER_NAMES_JSON")
    ;;
  *)
    echo "[named-det] unknown DATASET=$DATASET" >&2
    exit 2
    ;;
esac

echo "[named-det] dataset: $DATASET"
echo "[named-det] input:   $IN_DIR"
echo "[named-det] output:  $OUT_DIR"
echo "[named-det] names:   $NAMED_MULTI_SPEAKER_NAMES_JSON"

if [ ! -d "$IN_DIR" ]; then
  echo "[named-det] missing text-tags input at $IN_DIR" >&2
  exit 1
fi

python ./scripts/run_prompt_creation_deterministic.py \
  "${PROMPT_ARGS[@]}" \
  --dataset_name "$IN_DIR" \
  --output_dir "$OUT_DIR" \
  --overwrite_output_dir \
  --preprocessing_num_workers "$PREPROC_WORKERS"

echo "[named-det] DONE: $OUT_DIR"
