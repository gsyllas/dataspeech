# Leonardo HPC pipeline for dataspeech

End-to-end pipeline for tagging the two TTS datasets on Leonardo:

- `/leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/female_normalized/`
- `/leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/male_normalized/`

Each one has a `metadata.csv` (columns: `filename, speaker_id,
transcription, transcription_original, origin_dataset, gender`) and a
`wavs/` directory.

The compute nodes have **no internet**, so everything has to be pre-staged
on a login node: conda env, HF models, torch hub checkpoints, penn pitch
model, and the HF dataset itself. After that the SLURM jobs run fully
offline (`HF_HUB_OFFLINE=1` etc.).

## Layout

```
leonardo/
├── env.sh                                # Single source of truth for paths/cfg
├── login/                                # Run on a LOGIN node (has internet)
│   ├── 00_setup_conda_env.sh             # Build conda env INSIDE the repo
│   ├── 01_cache_models.py                # Pre-download brouhaha/squim/penn/LLM
│   ├── 02_probe_audio.py                 # Sample-rate / channel / duration stats
│   └── 03_build_hf_dataset.py            # CSV + wavs/ -> save_to_disk DatasetDict
├── slurm/                                # Submitted to COMPUTE nodes
│   ├── 10_annotate.slurm                 # main.py (pitch/SNR/SI-SDR/rate)
│   ├── 20_metadata_to_text.slurm         # bin -> text mapping
│   ├── 30_prompt_deterministic.slurm     # deterministic descriptions
│   ├── 40_prompt_llm.slurm               # LLM (Llama-3.1-8B-Instruct) descriptions
│   └── submit_all.sh                     # Chain stages with --dependency=afterok
└── logs/                                 # SLURM stdout/stderr land here
```

## Prerequisites

The conda env, all caches, and outputs live in `$REPO_ROOT` (i.e. inside
the cloned repo) — **not** in `$HOME` (Leonardo home quota is tiny).
Clone the repo somewhere on `/leonardo_work/...`. Example:

```bash
cd /leonardo_work/EUHPC_D29_081/gsyllas0
git clone -b leonardo-slurm <your fork URL> dataspeech
cd dataspeech
```

If you need a gated HF model (Llama-3.1-8B-Instruct is gated), run
`huggingface-cli login` on the login node once before stage 01.

## One-time setup (login node)

```bash
cd /leonardo_work/EUHPC_D29_081/gsyllas0/dataspeech
source leonardo/env.sh                       # paths only, no env activation yet
bash leonardo/login/00_setup_conda_env.sh    # ~10 min; installs miniforge + deps
activate_conda_env                           # now your shell has the env

# OPTIONAL: log in once if you want Llama (gated). Token is stored in HF_HOME.
huggingface-cli login

python leonardo/login/01_cache_models.py     # ~10 min; downloads all weights
python leonardo/login/02_probe_audio.py      # tells us actual sample rate / channels
python leonardo/login/03_build_hf_dataset.py --dataset both
```

`02_probe_audio.py` is purely informational — it lets us see what the
audio looks like before main.py runs for hours. If anything is non-mono
or unexpectedly high sample rate, that's fine: main.py internally
resamples to 16k for pitch/SNR/SQUIM.

## Submit jobs (login node)

```bash
# Submit the full pipeline for both datasets, chained with afterok deps:
bash leonardo/slurm/submit_all.sh both

# Or just one:
bash leonardo/slurm/submit_all.sh female
bash leonardo/slurm/submit_all.sh male

# Or one stage at a time:
DATASET=female sbatch leonardo/slurm/10_annotate.slurm
DATASET=female sbatch --dependency=afterok:<jobid> leonardo/slurm/20_metadata_to_text.slurm
```

Stage 30 (deterministic) and stage 40 (LLM) both depend on stage 20 and
run **in parallel** after it finishes. Stage 40 is the long one; stage
30 typically finishes in minutes.

## Outputs

Each stage writes to `$OUT_ROOT/<dataset>/<stage>/` as a
`save_to_disk` DatasetDict:

```
$OUT_ROOT/<female|male>/
  01_hf_dataset/                 # built by 03_build_hf_dataset.py
  02_tags/                       # main.py output (pitch/SNR/SI-SDR/etc)
  03_text_tags/                  # metadata_to_text.py output (text bins)
  04a_prompts_deterministic/     # deterministic descriptions
  04b_prompts_llm/               # Llama-3.1 generated descriptions
```

Hub upload is NOT done from compute nodes (no internet). If you want to
push results to the Hub, do it from a login node after the jobs finish:

```bash
python - <<'PY'
from datasets import load_from_disk
ds = load_from_disk("$OUT_ROOT/female/04b_prompts_llm")
ds.push_to_hub("syllasgiorgos/female-tts-descriptions")
PY
```

## Knobs

Edit `leonardo/env.sh` or override on the command line:

- `LLM_MODEL_ID` — change LLM (default `meta-llama/Meta-Llama-3.1-8B-Instruct`).
- `LLM_TORCH_COMPILE` — set `0` for non-Llama/Gemma models such as Qwen or Mistral.
- `LLM_TRUST_REMOTE_CODE` — set `1` only for models that require custom Hub code.
- `LLM_USE_HF_TOKEN` — set `1` for gated/private models; keep `0` for public models.
- `LLM_EVAL_BATCH_SIZE` — bump to 96+ if VRAM allows; 64 is conservative.
- `CPU_NUM_WORKERS` / `PREPROC_WORKERS` — boost node has 32 cores, default 8.
- `DATA_ROOT`, `FEMALE_DIR`, `MALE_DIR` — point elsewhere if data moves.
- `OUT_ROOT` — change where outputs land.
- `SKIP_STAGE_10=1` etc. for `submit_all.sh` — re-run downstream stages
  without redoing earlier ones.

## What changed in the upstream code

Four upstream scripts were patched to also accept a local `save_to_disk`
directory (in addition to a Hub dataset name), so the same scripts run
offline on compute nodes:

- `main.py`
- `scripts/metadata_to_text.py`
- `scripts/run_prompt_creation.py`
- `scripts/run_prompt_creation_deterministic.py`

Each patch only adds a `load_from_disk` fallback when the dataset arg
is a directory containing `dataset_dict.json` / `dataset_info.json`.
Hub paths still work exactly as before.
