"""Pre-fetch the Qwen2.5-Omni model for the standalone Omni prompt path.

Compute nodes on Leonardo have no internet, so the Omni weights + processor
must be cached on a login node first. Run once per repo clone (and again if you
change OMNI_MODEL_ID), from the dedicated Omni env:

    source leonardo/env.sh
    bash leonardo/login/10_setup_omni_env.sh   # one-time: builds .conda/omni
    activate_omni_conda_env
    python leonardo/login/09_cache_omni_model.py

The standalone Omni path needs a newer transformers than the tag pipeline,
which is exactly why it lives in its own env. If the import check below fails,
(re)build that env (Qwen3-Omni needs transformers>=4.57):

    TRANSFORMERS_SPEC='transformers>=4.57' bash leonardo/login/10_setup_omni_env.sh

This is separate from 01_cache_models.py on purpose: the Omni model is large
(~20 GB for the 7B) and not everyone running the tag pipeline needs it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _info(msg: str) -> None:
    print(f"[cache-omni] {msg}", flush=True)


def normalize_cache_env() -> None:
    """Keep downloads inside this repo unless CACHE_ROOT is explicit."""
    cache_root = Path(os.environ.get("CACHE_ROOT", REPO_ROOT / "cache")).resolve()
    expected = {
        "HF_HOME": cache_root / "hf",
        "HF_HUB_CACHE": cache_root / "hf",
        "HUGGINGFACE_HUB_CACHE": cache_root / "hf",
        "TRANSFORMERS_CACHE": cache_root / "hf",
    }
    for name, path in expected.items():
        old = os.environ.get(name)
        new = str(path)
        if old and Path(old).resolve() != path:
            _info(f"ignoring inherited {name}={old}; using {new}")
        os.environ[name] = new


def check_transformers() -> None:
    import transformers

    model_id = os.environ.get("OMNI_MODEL_ID", "Qwen/Qwen2.5-Omni-7B")
    name = model_id.lower()
    if "qwen3" in name or "omni3" in name:
        cls, need = "Qwen3OmniMoeForConditionalGeneration", "transformers>=4.57"
    else:
        cls, need = "Qwen2_5OmniForConditionalGeneration", "transformers>=4.52"
    if not hasattr(transformers, cls):
        raise SystemExit(
            f"transformers {transformers.__version__} lacks {cls} for {model_id!r} "
            f"(need {need}). Rebuild the Omni env:\n"
            f"    TRANSFORMERS_SPEC='{need}' bash leonardo/login/10_setup_omni_env.sh"
        )


def cache_omni() -> None:
    model_id = os.environ.get("OMNI_MODEL_ID", "Qwen/Qwen2.5-Omni-7B")
    _info(f"downloading Omni weights + processor for {model_id}")
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=model_id,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.model",
            "*.txt",
            "tokenizer*",
            "merges.txt",
            "vocab.json",
            "preprocessor_config.json",
            "processor_config.json",
            "chat_template.*",
            "generation_config.json",
            "spk_dict.pt",
        ],
    )
    _info(f"  -> {path}")


def main() -> int:
    normalize_cache_env()
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    _info(f"HF_HOME = {os.environ['HF_HOME']}")
    check_transformers()
    cache_omni()
    _info("Omni model cached. The stage 45 SLURM job can now run offline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
