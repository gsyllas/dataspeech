#!/bin/bash
#SBATCH --job-name=dspch-named-llm
#SBATCH --account=EUHPC_D29_081
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=leonardo/logs/%x_%j.out
#SBATCH --error=leonardo/logs/%x_%j.err
#
# Stage 40, named variant: LLM prompt creation with speaker names.
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
OUT_DIR="$(prompts_llm_dir_for "$DATASET")"
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
      echo "[named-llm] missing speaker names JSON: $NAMED_MULTI_SPEAKER_NAMES_JSON" >&2
      echo "[named-llm] run leonardo/login/04_prepare_named_variant.sh first" >&2
      exit 1
    fi
    PROMPT_ARGS+=(--speaker_id_column "speaker_id" --speaker_ids_to_name_json "$NAMED_MULTI_SPEAKER_NAMES_JSON")
    ;;
  *)
    echo "[named-llm] unknown DATASET=$DATASET" >&2
    exit 2
    ;;
esac

MODEL_EXTRA_ARGS=()
case "${LLM_TORCH_COMPILE,,}" in
  1|true|yes|on) MODEL_EXTRA_ARGS+=(--torch_compile) ;;
esac
case "${LLM_TRUST_REMOTE_CODE,,}" in
  1|true|yes|on) MODEL_EXTRA_ARGS+=(--trust_remote_code) ;;
esac
case "${LLM_USE_HF_TOKEN,,}" in
  1|true|yes|on) MODEL_EXTRA_ARGS+=(--token true) ;;
  *) MODEL_EXTRA_ARGS+=(--token false) ;;
esac

echo "[named-llm] dataset:  $DATASET"
echo "[named-llm] input:    $IN_DIR"
echo "[named-llm] output:   $OUT_DIR"
echo "[named-llm] model:    $LLM_MODEL_ID"
echo "[named-llm] names:    $NAMED_MULTI_SPEAKER_NAMES_JSON"
echo "[named-llm] compile:  $LLM_TORCH_COMPILE"
echo "[named-llm] use HF token: $LLM_USE_HF_TOKEN"
nvidia-smi -L || true

if [ ! -d "$IN_DIR" ]; then
  echo "[named-llm] missing text-tags input at $IN_DIR" >&2
  exit 1
fi

accelerate launch --num_processes=1 \
  ./scripts/run_prompt_creation.py \
  "${PROMPT_ARGS[@]}" \
  --dataset_name "$IN_DIR" \
  --output_dir "$OUT_DIR" \
  --overwrite_output_dir \
  --model_name_or_path "$LLM_MODEL_ID" \
  --torch_dtype "bfloat16" \
  --attn_implementation "sdpa" \
  "${MODEL_EXTRA_ARGS[@]}" \
  --per_device_eval_batch_size "$LLM_EVAL_BATCH_SIZE" \
  --dataloader_num_workers 4 \
  --preprocessing_num_workers "$PREPROC_WORKERS"

echo "[named-llm] DONE: $OUT_DIR"
