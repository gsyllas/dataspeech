#!/bin/bash
#SBATCH --job-name=dspch-omni
#SBATCH --account=EUHPC_D29_081
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=leonardo/logs/%x_%j.out
#SBATCH --error=leonardo/logs/%x_%j.err
#
# Stage 45 (standalone): Qwen2.5-Omni prompt creation.
# Hands the raw audio + an instruction prompt to a multimodal audio-language
# model and writes the English `text_description` directly. Does NOT use the
# tag pipeline (no 02_tags / 03_text_tags). Reads 01_hf_dataset, writes
# 04c_prompts_omni. Loads the model from the prefilled HF cache (offline).
set -euo pipefail
module purge
module load cuda/12.2

HERE_REPO="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
cd "$HERE_REPO"
# shellcheck disable=SC1091
source leonardo/env.sh

: "${DATASET:?DATASET must be set, e.g. greek_female_tts, greek_male_tts, female, male, multi_v2}"
# Standalone path uses its own newer-transformers env (built by 10_setup_omni_env.sh).
activate_omni_conda_env
set_offline_mode

IN_DIR="$(hf_dataset_dir_for "$DATASET")"
OUT_DIR="$(prompts_omni_dir_for "$DATASET")"
mkdir -p "$OUT_DIR" "leonardo/logs"

echo "[omni] dataset:      $DATASET"
echo "[omni] input:        $IN_DIR"
echo "[omni] output:       $OUT_DIR"
echo "[omni] model:        $OMNI_MODEL_ID"
echo "[omni] prompt style: $OMNI_PROMPT_STYLE"
echo "[omni] gender hint:  $OMNI_USE_GENDER_HINT"
echo "[omni] max tok:      $OMNI_MAX_NEW_TOKENS"
echo "[omni] temp:         $OMNI_TEMPERATURE"
echo "[omni] retries:      $OMNI_NUM_RETRIES"
echo "[omni] target sr:    $OMNI_TARGET_SR"
nvidia-smi -L || true

if [ ! -d "$IN_DIR" ]; then
  echo "[omni] missing HF dataset input at $IN_DIR" >&2
  echo "[omni] build it first on a login node, e.g.:" >&2
  echo "       python leonardo/login/03_build_hf_dataset.py --dataset $DATASET" >&2
  exit 1
fi

OMNI_EXTRA_ARGS=()
case "${OMNI_USE_GENDER_HINT,,}" in
  1|true|yes|on) OMNI_EXTRA_ARGS+=(--use_gender_hint) ;;
  *)             OMNI_EXTRA_ARGS+=(--no_gender_hint) ;;
esac
case "${OMNI_TRUST_REMOTE_CODE,,}" in
  1|true|yes|on) OMNI_EXTRA_ARGS+=(--trust_remote_code) ;;
esac

# Single A100 64GB; Qwen2.5-Omni-7B in bf16 with the talker disabled fits with
# batch=1 audio inference. accelerate launch keeps device placement consistent.
accelerate launch --num_processes=1 \
  ./scripts/run_prompt_creation_omni.py \
  --dataset_name "$IN_DIR" \
  --output_dir "$OUT_DIR" \
  --overwrite_output_dir \
  --model_name_or_path "$OMNI_MODEL_ID" \
  --prompt_style "$OMNI_PROMPT_STYLE" \
  --torch_dtype "bfloat16" \
  --attn_implementation "sdpa" \
  --max_new_tokens "$OMNI_MAX_NEW_TOKENS" \
  --temperature "$OMNI_TEMPERATURE" \
  --num_retries "$OMNI_NUM_RETRIES" \
  --target_sampling_rate "$OMNI_TARGET_SR" \
  --save_steps "$OMNI_SAVE_STEPS" \
  "${OMNI_EXTRA_ARGS[@]}"

echo "[omni] DONE: $OUT_DIR"
