#!/bin/bash
#SBATCH --job-name=dspch-det
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
# Stage 30: deterministic prompt creation (no LLM). Fast, CPU-bound.
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
OUT_DIR="$(prompts_det_dir_for "$DATASET")"
mkdir -p "$OUT_DIR" "leonardo/logs"

echo "[det] dataset: $DATASET"
echo "[det] input:   $IN_DIR"
echo "[det] output:  $OUT_DIR"

if [ ! -d "$IN_DIR" ]; then
  echo "[det] missing text-tags input at $IN_DIR" >&2
  exit 1
fi

python ./scripts/run_prompt_creation_deterministic.py \
  --is_new_speaker_prompt \
  --dataset_name "$IN_DIR" \
  --output_dir "$OUT_DIR" \
  --overwrite_output_dir \
  --preprocessing_num_workers "$PREPROC_WORKERS"

echo "[det] DONE: $OUT_DIR"
