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

# ---- 1. miniforge (always install in repo, never reuse system conda) -------
# Reusing $HOME's miniconda pulls in its .condarc (typically with `defaults`),
# which has shadowed conda-forge packages like espeak-ng in the past. Use a
# fresh miniforge with conda-forge as the ONLY channel.
MINIFORGE_DIR="$REPO_ROOT/.conda/miniforge"
if [ ! -x "$MINIFORGE_DIR/bin/conda" ]; then
  echo "[setup] installing miniforge into $MINIFORGE_DIR"
  INSTALLER="$REPO_ROOT/.conda/miniforge_installer.sh"
  curl -fsSL -o "$INSTALLER" \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash "$INSTALLER" -b -p "$MINIFORGE_DIR"
  rm -f "$INSTALLER"
else
  echo "[setup] miniforge already present at $MINIFORGE_DIR (reusing)"
fi

# shellcheck disable=SC1091
source "$MINIFORGE_DIR/etc/profile.d/conda.sh"

# Make sure we never read $HOME/.condarc (which may have `defaults` etc.) for
# this shell. CONDARC=/dev/null forces conda to ignore user config files.
export CONDARC=/dev/null

# ---- 2. env -----------------------------------------------------------------
if [ ! -d "$CONDA_ENV_PREFIX" ] || [ ! -x "$CONDA_ENV_PREFIX/bin/python" ]; then
  echo "[setup] creating env at $CONDA_ENV_PREFIX"
  # Python 3.10: known good with pyannote.audio + brouhaha-vad + penn.
  # espeak-ng pulled from conda-forge so phonemizer works without sudo.
  # --override-channels guarantees we ignore any inherited channel config.
  "$MINIFORGE_DIR/bin/conda" create -y -p "$CONDA_ENV_PREFIX" \
    --override-channels -c conda-forge -c nvidia \
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
echo "[setup] checking espeak-ng on PATH:"
command -v espeak-ng || { echo "[setup] espeak-ng binary not found in env" >&2; exit 1; }
python - <<'PY'
import torch, torchaudio, transformers, datasets, accelerate, penn, phonemizer
import pyannote.audio  # noqa
import brouhaha       # noqa
from phonemizer.backend import EspeakBackend
EspeakBackend(language="el")  # fails fast if espeak-ng isn't reachable
print("[setup] torch:", torch.__version__,
      "torchaudio:", torchaudio.__version__,
      "transformers:", transformers.__version__,
      "datasets:", datasets.__version__)
print("[setup] CUDA available (login node, may be False):", torch.cuda.is_available())
PY

echo "[setup] DONE. Activate with:"
echo "        source \"$REPO_ROOT/leonardo/env.sh\" && activate_conda_env"
