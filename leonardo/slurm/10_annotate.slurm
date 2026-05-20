#!/bin/bash
#SBATCH --job-name=dspch-annotate
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
# Stage 10: run main.py to compute pitch, SNR, reverb, speaking rate, SI-SDR
# on the HF dataset already built by 03_build_hf_dataset.py.
#
# Submit with:
#   DATASET=female sbatch leonardo/slurm/10_annotate.slurm
#   DATASET=male   sbatch leonardo/slurm/10_annotate.slurm
#   DATASET=multi  sbatch leonardo/slurm/10_annotate.slurm
set -euo pipefail
module purge
module load cuda/12.2

# --- env ---
HERE_REPO="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
cd "$HERE_REPO"
# shellcheck disable=SC1091
source leonardo/env.sh

: "${DATASET:?DATASET must be set to 'female', 'male', or 'multi'}"
activate_conda_env
set_offline_mode

IN_DIR="$(hf_dataset_dir_for "$DATASET")"
OUT_DIR="$(tags_dir_for "$DATASET")"
mkdir -p "$OUT_DIR" "leonardo/logs"

echo "[annotate] dataset:   $DATASET"
echo "[annotate] input:     $IN_DIR"
echo "[annotate] output:    $OUT_DIR"
echo "[annotate] HF_HOME:   $HF_HOME"
echo "[annotate] TORCH_HOME:$TORCH_HOME"
nvidia-smi -L || true

if [ ! -d "$IN_DIR" ]; then
  echo "[annotate] missing input dataset at $IN_DIR" >&2
  echo "[annotate] run leonardo/login/03_build_hf_dataset.py on the login node first" >&2
  exit 1
fi

python main.py "$IN_DIR" \
  --output_dir "$OUT_DIR" \
  --audio_column_name "audio" \
  --text_column_name "text" \
  --cpu_num_workers "$CPU_NUM_WORKERS" \
  --batch_size 4 \
  --penn_batch_size 4096 \
  --num_workers_per_gpu_for_pitch 1 \
  --num_workers_per_gpu_for_snr 1 \
  --num_workers_per_gpu_for_squim 1 \
  --apply_squim_quality_estimation

echo "[annotate] DONE: $OUT_DIR"
