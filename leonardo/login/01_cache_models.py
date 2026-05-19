"""Pre-fetch every model dataspeech needs onto local cache, on a login node.

Compute nodes on Leonardo have no internet, so HF / torch / penn must be able
to find everything offline. Run this once per repo clone:

    source leonardo/env.sh && activate_conda_env
    huggingface-cli login    # only needed if you want a gated LLM
    python leonardo/login/01_cache_models.py

Idempotent. Re-run any time you change LLM_MODEL_ID.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _info(msg: str) -> None:
    print(f"[cache] {msg}", flush=True)


def cache_brouhaha() -> None:
    repo = os.environ.get("BROUHAHA_REPO", "ylacombe/brouhaha-best")
    _info(f"downloading {repo}/best.ckpt (SNR + reverb)")
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo, filename="best.ckpt")
    _info(f"  -> {path}")


def cache_squim() -> None:
    _info("downloading torchaudio SQUIM_OBJECTIVE (SI-SDR / PESQ / STOI)")
    from torchaudio.pipelines import SQUIM_OBJECTIVE

    model = SQUIM_OBJECTIVE.get_model()
    _info(f"  loaded {type(model).__name__}")


def cache_penn() -> None:
    _info("triggering penn pitch model download (FCNF0++)")
    import numpy as np
    import torch
    import penn

    # 1 s of silence is enough to make penn lazy-init its bundled checkpoint.
    sr = 16_000
    dummy = torch.zeros(1, sr)
    try:
        penn.from_audio(
            dummy,
            sr,
            hopsize=0.01,
            fmin=30.0,
            fmax=1000.0,
            checkpoint=None,
            batch_size=1024,
            center="half-hop",
            interp_unvoiced_at=0.065,
            gpu=None,
        )
    except Exception as e:  # noqa: BLE001
        # penn sometimes errors on all-silence; the model is what we care about.
        _info(f"  penn warmup raised {type(e).__name__}: {e} (ok if model file was downloaded)")
    # Locate the cached checkpoint so the user sees where it lives.
    pkg_dir = Path(penn.__file__).resolve().parent
    found = list(pkg_dir.rglob("*.pt")) + list(pkg_dir.rglob("*.ckpt"))
    if found:
        _info(f"  penn assets: {[str(p) for p in found[:3]]}")
    _ = np  # silence linters


def cache_llm() -> None:
    model_id = os.environ.get("LLM_MODEL_ID", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    _info(f"downloading LLM weights + tokenizer for {model_id}")
    from huggingface_hub import snapshot_download

    # snapshot_download is more forgiving than from_pretrained for big repos.
    path = snapshot_download(
        repo_id=model_id,
        allow_patterns=[
            "*.json",
            "*.model",
            "*.safetensors",
            "*.txt",
            "tokenizer*",
            "special_tokens_map.json",
            "generation_config.json",
        ],
    )
    _info(f"  -> {path}")


def main() -> int:
    # Caches must be set BEFORE importing torch / hf_hub.
    required = ["HF_HOME", "TORCH_HOME"]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        print(
            f"[cache] missing env vars {missing}; source leonardo/env.sh first",
            file=sys.stderr,
        )
        return 1

    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)

    _info(f"HF_HOME    = {os.environ['HF_HOME']}")
    _info(f"TORCH_HOME = {os.environ['TORCH_HOME']}")

    cache_brouhaha()
    cache_squim()
    cache_penn()
    cache_llm()

    _info("all model caches populated. Compute jobs can now run offline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
