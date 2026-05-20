#!/bin/bash
#SBATCH --job-name=dspch-bins
#SBATCH --account=EUHPC_D29_081
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=leonardo/logs/%x_%j.out
#SBATCH --error=leonardo/logs/%x_%j.err
#
# Stage 20: map continuous tags -> text bins using the precomputed
# v02_bin_edges + v02_text_bins from the repo (so we don't need to
# recompute bin edges from one dataset).
#
# CPU only in practice, but we keep --gres=gpu:1 here because the
# boost partition requires it for the boost queue. Set
# SLURM_GRES=none in env.sh if your queue allows it.
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

IN_DIR="$(tags_dir_for "$DATASET")"
OUT_DIR="$(text_tags_dir_for "$DATASET")"
mkdir -p "$OUT_DIR" "leonardo/logs"

echo "[bins] dataset:   $DATASET"
echo "[bins] input:     $IN_DIR"
echo "[bins] output:    $OUT_DIR"
echo "[bins] edges:     $BIN_EDGES_PATH"
echo "[bins] textbins:  $TEXT_BINS_PATH"

if [ ! -d "$IN_DIR" ]; then
  echo "[bins] missing annotation output at $IN_DIR" >&2
  echo "[bins] did stage 10_annotate finish successfully?" >&2
  exit 1
fi

python ./scripts/metadata_to_text.py \
  "$IN_DIR" \
  --output_dir "$OUT_DIR" \
  --cpu_num_workers "$CPU_NUM_WORKERS" \
  --path_to_bin_edges "$BIN_EDGES_PATH" \
  --path_to_text_bins "$TEXT_BINS_PATH" \
  --apply_squim_quality_estimation

echo "[bins] DONE: $OUT_DIR"
