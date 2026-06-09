# Parler-TTS Dataset Manifest

Use these `save_to_disk` DatasetDict paths from the Parler-TTS training repo.

## Recommended Training Dataset

Use this as the main named-speaker training dataset:

```text
/leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/dataspeech_out_named/multi_v2/04b_prompts_llm
```

Notes:

- Source root: `/leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/greekData`
- Sources: `commonVoice_greek_clean_with_speaker_ids`, `cs10_greek_dataset`, `greek_male_3.5h`, `greek_female_tts`, `greek_male_tts`
- Prompts: named-speaker LLM prompts generated with Qwen.
- Speaker names: `/leonardo_work/EUHPC_D29_081/gsyllas0/dataspeech/leonardo/config/multi_v2_speaker_names.json`
- Speaker IDs are source-prefixed in `multi_v2` to avoid collisions.

## Fallback Dataset

Use this only if you want deterministic descriptions instead of LLM descriptions:

```text
/leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/dataspeech_out_named/multi_v2/04a_prompts_deterministic
```

## Anonymous Greek TTS Ablations

These are standalone anonymous/gender-aware outputs for the two pipe-metadata TTS datasets:

```text
/leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/dataspeech_out/greek_female_tts/04b_prompts_llm
/leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/dataspeech_out/greek_male_tts/04b_prompts_llm
/leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/dataspeech_out/greek_female_tts/04a_prompts_deterministic
/leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/dataspeech_out/greek_male_tts/04a_prompts_deterministic
```

## Qwen2.5-Omni Standalone Descriptions

Alternative descriptions written by Qwen2.5-Omni directly from the audio
(no tag pipeline). These keep the audio column and add a `_omni_failed` flag
column (rows where Omni did not return a valid English description have an
empty `text_description`).

```text
/leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/dataspeech_out/greek_female_tts/04c_prompts_omni
/leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/dataspeech_out/greek_male_tts/04c_prompts_omni
/leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/dataspeech_out_named/multi_v2/04c_prompts_omni
```

## Build Commands

```bash
source leonardo/env.sh
activate_conda_env

# Build the anonymous standalone HF inputs.
python leonardo/login/03_build_hf_dataset.py --dataset greek_tts

# Build the named combined HF input and speaker-name JSON.
bash leonardo/login/04_prepare_named_variant.sh multi_v2

# Submit anonymous standalone prompt pipelines.
bash leonardo/slurm/submit_all.sh greek_tts

# Submit named multi_v2 prompt pipeline.
export LLM_MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
export LLM_TORCH_COMPILE=0
export LLM_USE_HF_TOKEN=0
bash leonardo/slurm/submit_named.sh multi_v2

# Standalone Qwen2.5-Omni descriptions (audio -> text, no tag pipeline).
# One-time: build the dedicated Omni env, then cache the model on a login node.
bash leonardo/login/10_setup_omni_env.sh
activate_omni_conda_env
python leonardo/login/09_cache_omni_model.py
bash leonardo/slurm/submit_omni.sh greek_tts
bash leonardo/slurm/submit_omni.sh multi_v2

# After it finishes, eyeball quality + the _omni_failed rate:
python leonardo/login/11_inspect_omni_outputs.py --dataset greek_tts
python leonardo/login/11_inspect_omni_outputs.py --root "$NAMED_OUT_ROOT" --dataset multi_v2
```

## Quick Check

```bash
python - <<'PY'
from datasets import load_from_disk

paths = [
    "/leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/dataspeech_out_named/multi_v2/04b_prompts_llm",
    "/leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/dataspeech_out_named/multi_v2/04a_prompts_deterministic",
    "/leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/dataspeech_out/greek_female_tts/04b_prompts_llm",
    "/leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/dataspeech_out/greek_male_tts/04b_prompts_llm",
]

for path in paths:
    ds = load_from_disk(path)
    print(path, {split: len(ds[split]) for split in ds})
PY
```
