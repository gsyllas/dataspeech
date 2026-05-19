#!/bin/bash
#SBATCH --job-name=dspch-llm
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
# Stage 40: LLM-based prompt creation with Llama-3.1-8B-Instruct
# (or whatever LLM_MODEL_ID points to). Loads the model from
# the prefilled HF cache; no internet needed.
set -euo pipefail
module purge
module load cuda/12.2

HERE_REPO="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
cd "$HERE_REPO"
# shellcheck disable=SC1091
source leonardo/env.sh

: "${DATASET:?DATASET must be set to 'female' or 'male'}"
activate_conda_env
set_offline_mode

IN_DIR="$(text_tags_dir_for "$DATASET")"
OUT_DIR="$(prompts_llm_dir_for "$DATASET")"
mkdir -p "$OUT_DIR" "leonardo/logs"

echo "[llm] dataset:  $DATASET"
echo "[llm] input:    $IN_DIR"
echo "[llm] output:   $OUT_DIR"
echo "[llm] model:    $LLM_MODEL_ID"
echo "[llm] compile:  $LLM_TORCH_COMPILE"
echo "[llm] trust remote code: $LLM_TRUST_REMOTE_CODE"
echo "[llm] use HF token: $LLM_USE_HF_TOKEN"
nvidia-smi -L || true

if [ ! -d "$IN_DIR" ]; then
  echo "[llm] missing text-tags input at $IN_DIR" >&2
  exit 1
fi

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

# A100 64GB fits Llama-3.1-8B-Instruct in bf16 with batch=64; no need for 4-bit.
# Pass speaker_id_column so the script knows the dataset has multiple speakers;
# we don't map ids -> names (no speaker_ids_to_name_json), so descriptions stay
# anonymous but gender/pitch/etc. are still included.
accelerate launch --num_processes=1 \
  ./scripts/run_prompt_creation.py \
  --is_new_speaker_prompt \
  --dataset_name "$IN_DIR" \
  --output_dir "$OUT_DIR" \
  --overwrite_output_dir \
  --model_name_or_path "$LLM_MODEL_ID" \
  --torch_dtype "bfloat16" \
  --attn_implementation "sdpa" \
  "${MODEL_EXTRA_ARGS[@]}" \
  --per_device_eval_batch_size "$LLM_EVAL_BATCH_SIZE" \
  --dataloader_num_workers 4 \
  --preprocessing_num_workers "$PREPROC_WORKERS" \
  --speaker_id_column "speaker_id"

echo "[llm] DONE: $OUT_DIR"
