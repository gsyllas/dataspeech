#!/usr/bin/env bash
# Login-node only: create a DEDICATED conda env for the standalone Qwen-Omni
# prompt path at $OMNI_CONDA_ENV_PREFIX (default $REPO_ROOT/.conda/omni).
#
# This env is separate from the tag-pipeline env on purpose: Qwen2.5/3-Omni
# needs a newer transformers than pyannote/penn/brouhaha tolerate, so we do not
# touch the main env. Idempotent: reuses miniforge and skips existing steps.
#
# Usage:
#   bash leonardo/login/10_setup_omni_env.sh
#
# Then cache the model and submit:
#   source leonardo/env.sh && activate_omni_conda_env
#   python leonardo/login/09_cache_omni_model.py
#   bash leonardo/slurm/submit_omni.sh greek_female_tts
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/../env.sh"

mkdir -p "$REPO_ROOT/.conda" "$CACHE_ROOT" "$REPO_ROOT/leonardo/logs"

# ---- 1. reuse the repo miniforge (built by 00_setup_conda_env.sh) ----------
MINIFORGE_DIR="$REPO_ROOT/.conda/miniforge"
if [ ! -x "$MINIFORGE_DIR/bin/conda" ]; then
  echo "[omni-setup] miniforge not found at $MINIFORGE_DIR" >&2
  echo "[omni-setup] run leonardo/login/00_setup_conda_env.sh first" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
export CONDARC=/dev/null

# ---- 2. create the omni env -------------------------------------------------
# No espeak-ng / phonemizer needed here: this path never runs the tag pipeline.
if [ ! -d "$OMNI_CONDA_ENV_PREFIX" ] || [ ! -x "$OMNI_CONDA_ENV_PREFIX/bin/python" ]; then
  echo "[omni-setup] creating env at $OMNI_CONDA_ENV_PREFIX"
  "$MINIFORGE_DIR/bin/conda" create -y -p "$OMNI_CONDA_ENV_PREFIX" \
    --override-channels -c conda-forge \
    python=3.10 \
    libsndfile sox ffmpeg \
    pip git
else
  echo "[omni-setup] env already exists at $OMNI_CONDA_ENV_PREFIX (reusing)"
fi

conda activate "$OMNI_CONDA_ENV_PREFIX"

# ---- 3. pip dependencies ----------------------------------------------------
OMNI_CONSTRAINTS="$HERE/omni-constraints.txt"
TORCH_VERSION="2.5.1"
# torchvision 0.20.1 pairs with torch 2.5.1. It is required because
# Qwen2_5OmniProcessor builds a video sub-processor at load time (even though we
# never pass video), which needs the torchvision backend to import.
TORCHVISION_VERSION="0.20.1"

echo "[omni-setup] installing pytorch (cu121 wheels)"
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu121 \
  "torch==$TORCH_VERSION" "torchaudio==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"

# Transformers spec: 4.52 <= v < 5 for Qwen2.5-Omni.
# MUST stay <5: transformers 5.x imports torch.float8_e8m0fnu at load time,
# which needs torch>=2.7. Leonardo is pinned to torch 2.5.1 (newest cu121
# wheel), so 5.x cannot import. Qwen3-Omni (needs transformers>=4.57 / 5.x) is
# therefore not runnable on this stack — use Qwen2.5-Omni here.
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers>=4.52,<5}"
echo "[omni-setup] installing $TRANSFORMERS_SPEC + audio deps (pinned by $OMNI_CONSTRAINTS)"
pip install -c "$OMNI_CONSTRAINTS" \
  "$TRANSFORMERS_SPEC" \
  "accelerate>=0.30" \
  "datasets[audio]" \
  "soundfile" \
  "librosa" \
  "qwen-omni-utils" \
  "pandas"
pip check || true

# ---- 4. sanity check --------------------------------------------------------
python - <<'PY'
import numpy
if int(numpy.__version__.split(".")[0]) >= 2:
    raise RuntimeError(
        f"numpy {numpy.__version__} >= 2.0; the cu121 torch wheels expect numpy 1.x."
    )
import torch, torchaudio, transformers, datasets, accelerate
print("[omni-setup] torch:", torch.__version__,
      "transformers:", transformers.__version__,
      "datasets:", datasets.__version__)
import os
if int(transformers.__version__.split(".")[0]) >= 5:
    raise RuntimeError(
        f"transformers {transformers.__version__} (5.x) imports torch.float8_e8m0fnu, "
        "which needs torch>=2.7. This stack is pinned to torch 2.5.1 (cu121). "
        "Reinstall a 4.x: pip install -c leonardo/login/omni-constraints.txt 'transformers<5'"
    )
model_id = os.environ.get("OMNI_MODEL_ID", "Qwen/Qwen2.5-Omni-7B")
name = model_id.lower()
if "qwen3" in name or "omni3" in name:
    raise RuntimeError(
        f"{model_id!r} needs transformers>=4.57 (5.x) and torch>=2.7, which is not "
        "available as a cu121 wheel on Leonardo. Use a Qwen2.5-Omni model instead "
        "(e.g. Qwen/Qwen2.5-Omni-7B or Qwen/Qwen2.5-Omni-3B)."
    )
if not hasattr(transformers, "Qwen2_5OmniForConditionalGeneration"):
    raise RuntimeError(
        f"transformers {transformers.__version__} lacks Qwen2_5OmniForConditionalGeneration; "
        "need transformers>=4.52,<5."
    )
print(f"[omni-setup] Omni classes available for {model_id}")
print("[omni-setup] CUDA available (login node, may be False):", torch.cuda.is_available())
PY

echo "[omni-setup] DONE. Use it with:"
echo "        source \"$REPO_ROOT/leonardo/env.sh\" && activate_omni_conda_env"
echo "        python leonardo/login/09_cache_omni_model.py"
