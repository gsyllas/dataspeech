#!/usr/bin/env bash
# Login-node only: bootstrap miniforge (inside the repo if $HOME is full) and
# create the dataspeech conda env at $CONDA_ENV_PREFIX. Idempotent: skips
# steps that are already done.
#
# Usage:
#   bash leonardo/login/00_setup_conda_env.sh
#
# Run on a login node (compute nodes have no internet).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/../env.sh"

mkdir -p "$REPO_ROOT/.conda" "$CACHE_ROOT" "$REPO_ROOT/leonardo/logs"

# ---- 1. miniforge -----------------------------------------------------------
MINIFORGE_DIR="$REPO_ROOT/.conda/miniforge"
if [ ! -x "$MINIFORGE_DIR/bin/conda" ]; then
  if command -v conda >/dev/null 2>&1 && [ -n "${CONDA_EXE:-}" ]; then
    echo "[setup] using existing conda at $CONDA_EXE (will still create env inside repo)"
  else
    echo "[setup] installing miniforge into $MINIFORGE_DIR"
    INSTALLER="$REPO_ROOT/.conda/miniforge_installer.sh"
    curl -fsSL -o "$INSTALLER" \
      "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    bash "$INSTALLER" -b -p "$MINIFORGE_DIR"
    rm -f "$INSTALLER"
  fi
fi

# Locate conda.sh now that miniforge (or system conda) is available.
if [ -f "$MINIFORGE_DIR/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  echo "[setup] conda not found after install" >&2
  exit 1
fi

# ---- 2. env -----------------------------------------------------------------
if [ ! -d "$CONDA_ENV_PREFIX" ]; then
  echo "[setup] creating env at $CONDA_ENV_PREFIX"
  # Python 3.10: known good with pyannote.audio + brouhaha-vad + penn.
  # espeak-ng pulled from conda-forge so phonemizer works without sudo.
  conda create -y -p "$CONDA_ENV_PREFIX" \
    -c conda-forge -c nvidia \
    python=3.10 \
    espeak-ng \
    libsndfile \
    sox \
    ffmpeg \
    pip \
    git
else
  echo "[setup] env already exists at $CONDA_ENV_PREFIX (reusing)"
fi

conda activate "$CONDA_ENV_PREFIX"

# ---- 3. pip dependencies ----------------------------------------------------
# Pin torch to a CUDA-12 wheel that matches `module load cuda/12.2` on compute
# nodes. cu121 wheels are forward compatible with cu122 runtimes.
echo "[setup] installing pytorch (cu121 wheels)"
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu121 \
  "torch==2.3.1" "torchaudio==2.3.1"

# bitsandbytes for 4-bit LLM loading on A100.
echo "[setup] installing dataspeech requirements + bitsandbytes"
pip install -r "$REPO_ROOT/requirements.txt"
pip install "bitsandbytes>=0.43.1" "accelerate>=0.30" "soundfile" "pandas"

# Sanity check: GPU-side imports should at least be importable on a login node
# (they'll fail to actually run kernels without a GPU, which is fine here).
python - <<'PY'
import torch, torchaudio, transformers, datasets, accelerate, penn, phonemizer
import pyannote.audio  # noqa
import brouhaha       # noqa
print("[setup] torch:", torch.__version__,
      "torchaudio:", torchaudio.__version__,
      "transformers:", transformers.__version__,
      "datasets:", datasets.__version__)
print("[setup] CUDA available (login node, may be False):", torch.cuda.is_available())
PY

echo "[setup] DONE. Activate with:"
echo "        source \"$REPO_ROOT/leonardo/env.sh\" && activate_conda_env"
