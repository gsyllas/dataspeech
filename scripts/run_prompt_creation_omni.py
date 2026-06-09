"""Standalone prompt creation with Qwen2.5-Omni (audio -> English description).

This is an ALTERNATIVE to the tag-based pipeline (`run_prompt_creation.py`).
Instead of computing acoustic tags and asking a text LLM to phrase them, this
script hands the *raw audio* plus an instruction prompt to a multimodal
audio-language model (Qwen2.5-Omni) and lets the model write the
`text_description` directly by listening to the clip.

Input: a `save_to_disk` DatasetDict (or Hub dataset) that still contains the
audio column, e.g. the pipeline's `01_hf_dataset` stage. No tags required.

Output: the same DatasetDict with two added columns:
    text_description  - the English speech-style description (str)
    _omni_failed      - True if Omni never produced a valid English
                        description (in which case text_description is "")

The input speech here is Greek, but every description must be English. The
script enforces that with a strict system prompt, an English-only / anti-garbage
validator, and up to `--num_retries` resampled attempts per clip. Rows that
still fail validation are left empty and flagged rather than filled with
hallucinated or foreign-language text.

Example (single GPU):

    accelerate launch --num_processes=1 \
      scripts/run_prompt_creation_omni.py \
      --dataset_name "$OUT_ROOT/greek_female_tts/01_hf_dataset" \
      --output_dir   "$OUT_ROOT/greek_female_tts/04c_prompts_omni" \
      --model_name_or_path "Qwen/Qwen2.5-Omni-7B" \
      --overwrite_output_dir
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from datasets import Audio, DatasetDict, load_dataset, load_from_disk
from tqdm import tqdm


logger = logging.getLogger("run_prompt_creation_omni")


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
# The system prompt is the hard guardrail: English only, never transcribe,
# never name the spoken language. It is prepended to every conversation.
SYSTEM_PROMPT = (
    "You are an expert audio annotator that builds text-to-speech datasets. "
    "You listen to a short speech recording and describe only how the speaker "
    "sounds and the quality of the recording. You ALWAYS write in English, no "
    "matter what language is spoken in the recording. You NEVER transcribe, "
    "translate, quote, or summarise the words that are spoken, and you NEVER "
    "mention the language being spoken or the meaning of the speech. You reply "
    "with the description only, never with any other commentary."
)

# Parler-TTS style user prompt (the option chosen for this dataset). Covers the
# same attributes as the tag-based pipeline so the output is a drop-in.
USER_PROMPT_PARLER = (
    "Listen to the attached speech clip and write a single concise English "
    "description of how the speaker sounds, suitable as a text-to-speech style "
    "prompt.\n"
    "Describe these characteristics:\n"
    "- the speaker's gender (man / woman)\n"
    "- the pitch of the voice (low / moderate / high)\n"
    "- the speaking rate (slow / moderate / fast)\n"
    "- how expressive or monotone the delivery is\n"
    "- the recording quality (clean vs noisy, close vs distant/roomy)\n"
    "Rules:\n"
    "- Write in English only. Output one or two short sentences, nothing else.\n"
    "- Start immediately with the description (for example, \"A man speaks ...\").\n"
    "- Do NOT transcribe or mention what is being said, and do NOT mention the "
    "language being spoken.\n"
    "- Do NOT use labels, lists, quotes, or markdown. Return only the "
    "description sentence(s)."
)

# Richer / perceptual variant (timbre + emotion). Kept for easy A/B testing.
USER_PROMPT_RICH = (
    "Listen to the attached speech clip and write a single concise English "
    "description of how the speaker sounds.\n"
    "Describe: the speaker's gender, the pitch of the voice, the speaking rate, "
    "how expressive or monotone the delivery is, the recording quality (clean "
    "vs noisy, close vs distant/roomy), and the overall vocal timbre or tone "
    "(for example warm, bright, breathy, tense, or cheerful).\n"
    "Rules:\n"
    "- Write in English only. Output one or two short sentences, nothing else.\n"
    "- Start immediately with the description (for example, \"A woman ...\").\n"
    "- Do NOT transcribe or mention what is being said, and do NOT mention the "
    "language being spoken.\n"
    "- Do NOT use labels, lists, quotes, or markdown."
)

# Minimal / voice-only variant (no recording-quality guesses).
USER_PROMPT_MINIMAL = (
    "Listen to the clip and write one short English sentence describing the "
    "speaker's voice and delivery.\n"
    "Describe only: gender, pitch (low / moderate / high), speaking rate (slow "
    "/ moderate / fast), and whether the delivery is expressive or monotone.\n"
    "Rules:\n"
    "- Write in English only. One sentence, nothing else.\n"
    "- Start with \"A man\" or \"A woman\".\n"
    "- Do NOT transcribe or mention the language being spoken.\n"
    "- Do NOT use labels, lists, quotes, or markdown."
)

PROMPT_STYLES = {
    "parler": USER_PROMPT_PARLER,
    "rich": USER_PROMPT_RICH,
    "minimal": USER_PROMPT_MINIMAL,
}


# --------------------------------------------------------------------------- #
# Output cleaning + validation (English-only / anti-garbage guardrails)
# --------------------------------------------------------------------------- #
# Scripts that immediately disqualify a description (we want English/Latin).
_NON_LATIN_RANGES = (
    (0x0370, 0x03FF),  # Greek
    (0x1F00, 0x1FFF),  # Greek extended
    (0x0400, 0x04FF),  # Cyrillic
    (0x0500, 0x052F),  # Cyrillic supplement
    (0x0590, 0x05FF),  # Hebrew
    (0x0600, 0x06FF),  # Arabic
    (0x0900, 0x097F),  # Devanagari
    (0x3040, 0x30FF),  # Hiragana + Katakana
    (0x3400, 0x9FFF),  # CJK
    (0xAC00, 0xD7AF),  # Hangul
)

# Refusals / meta-commentary / transcription leaks we never want to keep.
_BAD_SUBSTRINGS = (
    "i'm sorry",
    "i am sorry",
    "i cannot",
    "i can't",
    "i can not",
    "i'm unable",
    "i am unable",
    "as an ai",
    "as a language model",
    "i don't have",
    "i do not have",
    "unfortunately",
    "the audio file",
    "the recording says",
    "the speaker says",
    "transcription",
    "transcript:",
    "translates to",
    "in greek",
    "the language",
    "the words spoken",
    "it sounds like the speaker is saying",
)

# A real description should mention at least one voice/recording attribute.
_VOICE_KEYWORDS = (
    "speak",
    "voice",
    "speaker",
    "deliver",
    "tone",
    "pitch",
    "pace",
    "man",
    "woman",
    "male",
    "female",
    "recording",
    "monoton",
    "express",
)

_CHAT_TOKEN_RE = re.compile(r"<\|/?(?:assistant|user|system|im_start|im_end|endoftext)\|>")
_TAG_RE = re.compile(r"</?\w+[^>]*>")
# A leading role word, possibly on its own line ("assistant\n", "system:").
_ROLE_PREFIX_RE = re.compile(r"^\s*(?:assistant|user|system)\b[\s:.\-]*", re.IGNORECASE)
# A lead-in clause that ends in a colon ("Sure, here is the description:").
_LEADIN_RE = re.compile(
    r"^[^:]{0,80}\b(?:sure|certainly|of course|okay|ok|here\s+(?:is|are|'s)|"
    r"description|response|answer)\b[^:]{0,40}:\s*",
    re.IGNORECASE,
)


def clean_omni_text(text: Optional[str]) -> str:
    """Strip chat-template scraps, role words, labels and surrounding punctuation."""
    if text is None:
        return ""
    text = str(text).replace("\x00", " ").strip()
    text = _CHAT_TOKEN_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    text = _ROLE_PREFIX_RE.sub("", text.strip())
    # If the model added trailing notes on new lines, keep just the first line
    # when it already looks like a full sentence; otherwise join everything.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1 and len(lines[0]) >= 20 and lines[0].rstrip()[-1:] in ".!?":
        text = lines[0]
    elif lines:
        text = " ".join(lines)
    text = _LEADIN_RE.sub("", text, count=1)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n\"'`*_:-")


def _has_non_latin(text: str) -> bool:
    for ch in text:
        code = ord(ch)
        for lo, hi in _NON_LATIN_RANGES:
            if lo <= code <= hi:
                return True
    return False


def is_english(text: str) -> bool:
    """True if `text` looks like English / Latin script (no Greek, CJK, ...)."""
    letters = [ch for ch in text if ch.isalpha()]
    if len(letters) < 8:
        return False
    if _has_non_latin(text):
        return False
    ascii_letters = sum(1 for ch in letters if ord(ch) < 128)
    return (ascii_letters / len(letters)) >= 0.9


def is_valid_description(text: str) -> bool:
    """English, plausible length, on-topic, and not a refusal/transcription."""
    if not text:
        return False
    if not (15 <= len(text) <= 600):
        return False
    if not is_english(text):
        return False
    lowered = text.lower()
    if any(bad in lowered for bad in _BAD_SUBSTRINGS):
        return False
    if not any(kw in lowered for kw in _VOICE_KEYWORDS):
        return False
    return True


# --------------------------------------------------------------------------- #
# Gender hint
# --------------------------------------------------------------------------- #
def normalize_gender(value) -> Optional[str]:
    """Map a dataset gender value to 'man' / 'woman', or None if unknown."""
    if value is None:
        return None
    token = str(value).strip().lower()
    if token in {"male", "man", "m", "masculine"}:
        return "man"
    if token in {"female", "woman", "f", "feminine"}:
        return "woman"
    return None


def build_user_prompt(base_prompt: str, gender_word: Optional[str]) -> str:
    if gender_word:
        return (
            f"{base_prompt}\n"
            f"Context you can rely on: the speaker is a {gender_word}. "
            f"Use this gender and do not contradict it."
        )
    return base_prompt


# --------------------------------------------------------------------------- #
# Audio helpers
# --------------------------------------------------------------------------- #
def to_mono_float32(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim > 1:
        # (channels, samples) or (samples, channels) -> average to mono.
        axis = 0 if arr.shape[0] < arr.shape[-1] else -1
        arr = arr.mean(axis=axis)
    return arr.astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# Checkpointing (plain-string results, resumable)
# --------------------------------------------------------------------------- #
_CKPT_RE = re.compile(r"^omni-checkpoint-(\d+)\.json$")


def load_resume(split_dir: Path) -> List[dict]:
    if not split_dir.is_dir():
        return []
    checkpoints = [(int(m.group(1)), p) for p in split_dir.glob("omni-checkpoint-*.json")
                   if (m := _CKPT_RE.match(p.name))]
    if not checkpoints:
        return []
    _, newest = max(checkpoints, key=lambda x: x[0])
    with newest.open("r", encoding="utf-8") as f:
        results = json.load(f)
    logger.info("Resuming %s from %d completed rows", split_dir, len(results))
    return results


def save_checkpoint(split_dir: Path, results: List[dict]) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    tmp = split_dir / f"omni-checkpoint-{len(results)}.json.tmp"
    final = split_dir / f"omni-checkpoint-{len(results)}.json"
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    tmp.replace(final)
    # Keep only the newest checkpoint.
    for old in split_dir.glob("omni-checkpoint-*.json"):
        if old != final and _CKPT_RE.match(old.name):
            old.unlink()


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
# (model class, processor class) candidates by Omni generation. The same
# conversation / generate(return_audio=False) / disable_talker API works for
# both, so the only difference is which class names exist in transformers.
_OMNI_CLASS_CANDIDATES = {
    "qwen3": [
        ("Qwen3OmniMoeForConditionalGeneration", "Qwen3OmniMoeProcessor"),
        ("Qwen2_5OmniForConditionalGeneration", "Qwen2_5OmniProcessor"),
    ],
    "qwen2_5": [
        ("Qwen2_5OmniForConditionalGeneration", "Qwen2_5OmniProcessor"),
        ("Qwen3OmniMoeForConditionalGeneration", "Qwen3OmniMoeProcessor"),
    ],
}


def _resolve_omni_classes(model_name: str):
    """Pick the right Omni model/processor classes for the requested model id."""
    import transformers as tf

    name = model_name.lower()
    family = "qwen3" if ("qwen3" in name or "omni3" in name) else "qwen2_5"
    for model_cls, proc_cls in _OMNI_CLASS_CANDIDATES[family]:
        if hasattr(tf, model_cls) and hasattr(tf, proc_cls):
            return getattr(tf, model_cls), getattr(tf, proc_cls)
    raise SystemExit(
        f"transformers {tf.__version__} has no Omni classes for {model_name!r}.\n"
        "Qwen2.5-Omni needs transformers>=4.52; Qwen3-Omni needs transformers>=4.57.\n"
        "Set up the dedicated env first:\n"
        "    bash leonardo/login/10_setup_omni_env.sh\n"
        "or upgrade in place:\n"
        "    pip install -U 'transformers>=4.57' accelerate soundfile librosa"
    )


def _neutralize_torch_load_check() -> None:
    """Allow torch.load of Qwen's trusted spk_dict.pt on torch<2.6.

    transformers>=4.57 hard-blocks torch.load unless torch>=2.6 (CVE-2025-32434).
    Leonardo is pinned to torch 2.5.1 (no cu121 wheel exists for 2.6+). The only
    torch.load this path triggers is Qwen2.5-Omni's official `spk_dict.pt`
    (talker voice presets), loaded with weights_only=True from the offline cache
    during from_pretrained -- and we disable the talker immediately after, so the
    speaker data is never used. Neutralise that single check for this trusted file.
    """
    noop = lambda *a, **k: None  # noqa: E731
    patched = []
    try:
        import transformers.utils.import_utils as iu
        iu.check_torch_load_is_safe = noop
        patched.append("utils.import_utils")
    except Exception:  # noqa: BLE001
        pass
    try:
        import transformers.utils as tu
        if hasattr(tu, "check_torch_load_is_safe"):
            tu.check_torch_load_is_safe = noop
            patched.append("utils")
    except Exception:  # noqa: BLE001
        pass
    # The call site binds the name as a module global, so patch it there too.
    try:
        from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as mq
        if hasattr(mq, "check_torch_load_is_safe"):
            mq.check_torch_load_is_safe = noop
            patched.append("qwen2_5_omni.modeling")
    except Exception:  # noqa: BLE001
        pass
    if patched:
        logger.info("Neutralised torch.load safety check for trusted spk_dict.pt (%s).",
                    ", ".join(patched))


def load_model_and_processor(args):
    model_cls, processor_cls = _resolve_omni_classes(args.model_name_or_path)
    _neutralize_torch_load_check()

    torch_dtype = getattr(torch, args.torch_dtype) if args.torch_dtype not in ("auto", None) else "auto"
    logger.info("Loading Omni model %s via %s (dtype=%s)",
                args.model_name_or_path, model_cls.__name__, args.torch_dtype)
    model = model_cls.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    ).eval()
    # We only want text descriptions; drop the speech "talker" to save VRAM.
    if hasattr(model, "disable_talker"):
        model.disable_talker()
        logger.info("Disabled Omni talker (text-only generation).")

    processor = processor_cls.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    return model, processor


def generate_once(model, processor, audio: np.ndarray, sampling_rate: int,
                  user_prompt: str, do_sample: bool, temperature: float,
                  max_new_tokens: int) -> str:
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "audio", "audio": audio},
            {"type": "text", "text": user_prompt},
        ]},
    ]
    chat_text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        text=chat_text,
        audio=[audio],
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
    )
    inputs = inputs.to(model.device)
    # BatchFeature.to(dtype) casts only the float tensors (audio features) to the
    # model dtype and leaves integer input_ids untouched, matching the Omni docs.
    model_dtype = getattr(model, "dtype", None)
    if model_dtype is not None:
        try:
            inputs = inputs.to(model_dtype)
        except (TypeError, ValueError):
            pass

    gen_kwargs = dict(max_new_tokens=max_new_tokens, return_audio=False)
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        gen_kwargs.update(do_sample=False)

    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)
    # When the talker is disabled / return_audio=False, generate returns only
    # the text ids; some versions still return a (text_ids, audio) tuple.
    if isinstance(output, (tuple, list)):
        output = output[0]
    trimmed = output[:, inputs["input_ids"].shape[1]:]
    decoded = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return decoded[0] if decoded else ""


def describe_clip(model, processor, audio: np.ndarray, sampling_rate: int,
                  user_prompt: str, args) -> Tuple[str, bool]:
    """Try greedy first, then resampled retries. Return (description, failed)."""
    last_raw = ""
    for attempt in range(args.num_retries + 1):
        do_sample = attempt > 0  # attempt 0 greedy; retries sample for variety.
        temperature = args.temperature + 0.1 * (attempt - 1 if attempt > 0 else 0)
        try:
            raw = generate_once(
                model, processor, audio, sampling_rate, user_prompt,
                do_sample=do_sample, temperature=min(temperature, 1.2),
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - never let one clip kill the run
            logger.warning("Generation error on attempt %d: %s", attempt, exc)
            continue
        cleaned = clean_omni_text(raw)
        last_raw = cleaned or last_raw
        if is_valid_description(cleaned):
            return cleaned, False
    # All attempts failed validation -> empty + flag (per chosen policy).
    logger.debug("No valid English description (last raw: %r)", last_raw[:120])
    return "", True


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #
def load_input_dataset(args) -> DatasetDict:
    local_save = (
        os.path.isdir(args.dataset_name)
        and os.path.isfile(os.path.join(args.dataset_name, "dataset_dict.json"))
    )
    if local_save:
        raw = load_from_disk(args.dataset_name)
        if not isinstance(raw, DatasetDict):
            raw = DatasetDict({"train": raw})
        if args.dataset_split_name:
            raw = DatasetDict({s: raw[s] for s in args.dataset_split_name.split("+")})
        return raw

    if args.dataset_split_name:
        raw = DatasetDict()
        for split in args.dataset_split_name.split("+"):
            raw[split] = load_dataset(
                args.dataset_name, args.dataset_config_name, split=split
            )
        return raw
    return load_dataset(args.dataset_name, args.dataset_config_name)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_name", required=True,
                   help="Path to a save_to_disk DatasetDict (with audio) or a Hub id.")
    p.add_argument("--output_dir", required=True, help="Where to save the result DatasetDict.")
    p.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--dataset_config_name", default=None)
    p.add_argument("--dataset_split_name", default=None,
                   help="'+'-separated split names; default: all splits.")
    p.add_argument("--audio_column_name", default="audio")
    p.add_argument("--gender_column_name", default="gender")
    p.add_argument("--text_description_column", default="text_description")
    p.add_argument("--prompt_style", choices=sorted(PROMPT_STYLES), default="parler")
    p.add_argument("--use_gender_hint", dest="use_gender_hint", action="store_true", default=True)
    p.add_argument("--no_gender_hint", dest="use_gender_hint", action="store_false")
    p.add_argument("--target_sampling_rate", type=int, default=16000)
    p.add_argument("--torch_dtype", default="bfloat16",
                   choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--attn_implementation", default="sdpa",
                   choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--trust_remote_code", action="store_true", default=False)
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--num_retries", type=int, default=2,
                   help="Resampled retries when an output fails the English check.")
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--max_eval_samples", type=int, default=None,
                   help="Cap rows per split (debugging).")
    p.add_argument("--overwrite_output_dir", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--push_to_hub", action="store_true", default=False)
    p.add_argument("--hub_dataset_id", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if args.push_to_hub and not args.hub_dataset_id:
        raise SystemExit("--push_to_hub requires --hub_dataset_id")

    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    if args.overwrite_output_dir and output_dir.is_dir():
        logger.info("Cleaning output dir from previous run: %s", output_dir)
        shutil.rmtree(output_dir)

    raw_datasets = load_input_dataset(args)
    if args.max_eval_samples is not None:
        for split in raw_datasets:
            n = min(args.max_eval_samples, len(raw_datasets[split]))
            raw_datasets[split] = raw_datasets[split].select(range(n))

    first_split = next(iter(raw_datasets))
    columns = set(raw_datasets[first_split].column_names)
    if args.audio_column_name not in columns:
        raise SystemExit(
            f"Audio column {args.audio_column_name!r} not found. Columns: {sorted(columns)}"
        )
    has_gender = args.gender_column_name in columns
    if args.use_gender_hint and not has_gender:
        logger.warning("Gender column %r missing; continuing without gender hints.",
                       args.gender_column_name)

    base_prompt = PROMPT_STYLES[args.prompt_style]
    model, processor = load_model_and_processor(args)

    # Fail-fast smoke test: run the real pipeline on one clip WITHOUT catching
    # errors, so a bad processor signature / env issue crashes here instead of
    # silently flagging every row as failed. Also prints one example to the log.
    smoke_view = raw_datasets[first_split].cast_column(
        args.audio_column_name, Audio(sampling_rate=args.target_sampling_rate)
    )
    if len(smoke_view) > 0:
        smoke_sample = smoke_view[0]
        smoke_gender = (
            normalize_gender(smoke_sample.get(args.gender_column_name))
            if (args.use_gender_hint and has_gender) else None
        )
        smoke_raw = generate_once(
            model, processor,
            to_mono_float32(smoke_sample[args.audio_column_name]["array"]),
            args.target_sampling_rate,
            build_user_prompt(base_prompt, smoke_gender),
            do_sample=False, temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
        logger.info("Smoke test (raw)   : %r", smoke_raw[:200])
        logger.info("Smoke test (clean) : %r", clean_omni_text(smoke_raw)[:200])

    result_splits = DatasetDict()
    for split, split_ds in raw_datasets.items():
        # Decode audio at the model's expected sampling rate on the fly. The
        # original (un-resampled) audio is preserved in the saved dataset.
        audio_view = split_ds.cast_column(
            args.audio_column_name, Audio(sampling_rate=args.target_sampling_rate)
        )

        split_dir = output_dir / split
        results: List[dict] = load_resume(split_dir)
        start = len(results)
        total = len(split_ds)

        progress = tqdm(total=total, initial=start, desc=f"omni:{split}")
        for idx in range(start, total):
            sample = audio_view[idx]
            audio = to_mono_float32(sample[args.audio_column_name]["array"])

            gender_word = None
            if args.use_gender_hint and has_gender:
                gender_word = normalize_gender(sample.get(args.gender_column_name))
            user_prompt = build_user_prompt(base_prompt, gender_word)

            description, failed = describe_clip(
                model, processor, audio, args.target_sampling_rate, user_prompt, args
            )
            results.append({"text_description": description, "_omni_failed": failed})
            progress.update(1)

            if len(results) % args.save_steps == 0 or len(results) == total:
                save_checkpoint(split_dir, results)
        progress.close()

        n_failed = sum(r["_omni_failed"] for r in results)
        logger.info("%s: %d rows, %d flagged empty (_omni_failed).", split, total, n_failed)

        # Attach the new columns to the original (audio-preserving) split.
        clean_ds = split_ds
        for col in (args.text_description_column, "_omni_failed"):
            if col in clean_ds.column_names:
                clean_ds = clean_ds.remove_columns(col)
        clean_ds = clean_ds.add_column(
            args.text_description_column, [r["text_description"] for r in results]
        )
        clean_ds = clean_ds.add_column(
            "_omni_failed", [r["_omni_failed"] for r in results]
        )
        result_splits[split] = clean_ds

    output_dir.mkdir(parents=True, exist_ok=True)
    result_splits.save_to_disk(str(output_dir))
    logger.info("Wrote Omni descriptions to %s", output_dir)

    if args.push_to_hub:
        result_splits.push_to_hub(
            args.hub_dataset_id,
            config_name=args.dataset_config_name or "default",
        )
        logger.info("Pushed to hub: %s", args.hub_dataset_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
