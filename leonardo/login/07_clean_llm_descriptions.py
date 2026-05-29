"""Clean small LLM decode artifacts from prompt descriptions.

This is intended for already-built prompt datasets such as:

    <OUT_ROOT>/greek_female_tts/04b_prompts_llm
    <OUT_ROOT>/greek_male_tts/04b_prompts_llm

The script writes a sibling backup before replacing the prompt dataset:

    04b_prompts_llm_before_clean
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

from datasets import DatasetDict, load_from_disk


DEFAULT_DATASETS = ("greek_female_tts", "greek_male_tts")
DESCRIPTION_START_PATTERNS = (
    re.compile(r"\bA\s+(?:female|male|woman|man|speaker)\b", re.IGNORECASE),
    re.compile(r"\bAn\s+(?:adult\s+)?(?:female|male|woman|man|speaker)\b", re.IGNORECASE),
    re.compile(r"\bThe\s+(?:female|male|woman|man)\s+speaker\b", re.IGNORECASE),
    re.compile(r"\bThe\s+(?:speaker|recording|voice|audio)\b", re.IGNORECASE),
    re.compile(r"\bIn\s+(?:a|an)\s+", re.IGNORECASE),
    re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?:'s)?\s+(?:voice\s+is|speaks|delivers|has|sounds)\b"),
)
DESCRIPTION_META_PREFIX = re.compile(
    r"\b(?:answer|description|following|generated|idioma|proporcion|responder|respuesta|kuvaus|seuraava)\b",
    re.IGNORECASE,
)
SUSPICIOUS_TEXT = re.compile(
    r"^\s*(?:[^A-Za-z0-9]{0,4}tica\b|opportunit|kommenttia|[:;,%>\"'<]|assistant|response|answer|"
    r"no\s+hay|respuesta|responder|puhuu)",
    re.IGNORECASE,
)


def clean_description(text: str) -> str:
    """Remove short prompt/chat-template scraps before the real description."""
    if text is None:
        return ""

    text = str(text).replace("\x00", "").strip()
    text = re.sub(r"<\|/?(?:assistant|user|system|im_start|im_end)\|>", " ", text)
    text = re.sub(r"</?\w+[^>]*>", " ", text)
    text = re.sub(r"^(?:assistant|response|answer)\s*[:\-]\s*", "", text, flags=re.IGNORECASE)

    lines = [line.strip(" \t\r\n\"'`:-,;!?%<>/=") for line in text.splitlines() if line.strip()]
    while len(lines) > 1 and len(lines[0]) <= 24:
        first = lines[0].lstrip(":-,.;!?\"'`%<>/= ")
        if _starts_description(first):
            break
        lines.pop(0)
    text = " ".join(lines) if lines else text

    earliest = None
    for pattern in DESCRIPTION_START_PATTERNS:
        match = pattern.search(text)
        if match and (earliest is None or match.start() < earliest):
            earliest = match.start()

    if earliest and earliest > 0:
        prefix = text[:earliest]
        if (
            "\n" in prefix
            or earliest <= 350
            or DESCRIPTION_META_PREFIX.search(prefix)
            or not re.search(r"[A-Za-z]{3,}", prefix)
        ):
            text = text[earliest:]

    return re.sub(r"\s+", " ", text).strip(" \t\r\n\"'`:-,;!?%<>/=")


def _starts_description(text: str) -> bool:
    return any(
        (match := pattern.search(text)) is not None and match.start() == 0
        for pattern in DESCRIPTION_START_PATTERNS
    )


def _preview(text: str, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def _looks_suspicious(text: str) -> bool:
    text = str(text).strip()
    if not text:
        return False
    if SUSPICIOUS_TEXT.search(text):
        return True
    head = text[:300]
    return bool(
        re.search(r"</?\w+[^>]*>", head)
        or re.search(r"\b(?:idioma|proporcion|respuesta|responder|kuvaus|seuraava|puhuu)\b", head, re.IGNORECASE)
    )


def _value(sample: dict, key: str) -> str:
    value = sample.get(key)
    return str(value).strip() if value is not None else ""


def _rate_phrase(rate: str) -> str:
    if not rate:
        return ""
    if rate == "moderate speed":
        return "at a moderate speed"
    if "slow" in rate:
        return rate
    if "fast" in rate:
        return f"at a {rate} pace"
    return f"at {rate}"


def fallback_description(sample: dict, speaker_names: dict[str, str] | None = None) -> str:
    name = ""
    if speaker_names and sample.get("speaker_id") is not None:
        name = speaker_names.get(str(sample.get("speaker_id")), "")

    gender = _value(sample, "gender")
    if name:
        subject = name
    elif gender in {"female", "male"}:
        subject = f"A {gender} speaker"
    else:
        subject = "A speaker"

    rate = _rate_phrase(_value(sample, "speaking_rate"))
    pitch = _value(sample, "pitch")
    monotony = _value(sample, "speech_monotony")
    noise = _value(sample, "sdr_noise") or _value(sample, "noise")
    reverberation = _value(sample, "reverberation")

    first = f"{subject} speaks"
    if rate:
        first += f" {rate}"
    if pitch:
        first += f" with a {pitch} voice"
    if monotony:
        first += f" and a {monotony} delivery"

    recording = []
    if noise:
        recording.append(f"has {noise}" if "noise" in noise else f"is {noise}")
    if reverberation:
        recording.append(f"is {reverberation}")
    if recording:
        return f"{first}. The recording {' and '.join(recording)}."
    return f"{first}."


def clean_sample(sample: dict, speaker_names: dict[str, str] | None = None) -> dict:
    original = sample.get("text_description", "")
    cleaned = clean_description(original)
    used_fallback = False
    if _looks_suspicious(cleaned) or (_looks_suspicious(original) and cleaned[:1].islower()):
        cleaned = fallback_description(sample, speaker_names)
        used_fallback = True
    return {"text_description": cleaned, "_used_fallback": used_fallback}


def _parse_datasets(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return DEFAULT_DATASETS

    datasets: list[str] = []
    for value in values:
        for item in value.replace(",", " ").split():
            if item == "greek_tts":
                datasets.extend(DEFAULT_DATASETS)
            else:
                datasets.append(item)

    seen = set()
    unique = []
    for dataset in datasets:
        if dataset not in seen:
            seen.add(dataset)
            unique.append(dataset)
    return tuple(unique)


def clean_dataset(
    root: Path,
    dataset: str,
    stage: str,
    speaker_names_json: Path | None = None,
    dry_run: bool = False,
    overwrite_backup: bool = False,
) -> None:
    prompt_dir = root / dataset / stage
    backup_dir = prompt_dir.with_name(f"{prompt_dir.name}_before_clean")
    tmp_dir = prompt_dir.with_name(f"{prompt_dir.name}_clean_tmp")

    if not prompt_dir.is_dir():
        raise SystemExit(f"missing prompt dataset: {prompt_dir}")

    loaded = load_from_disk(str(prompt_dir))
    if not isinstance(loaded, DatasetDict):
        raise SystemExit(f"expected DatasetDict at {prompt_dir}")

    cleaned = DatasetDict()
    total_changed = 0
    total_fallbacks = 0
    examples: list[tuple[str, int, str, str]] = []
    speaker_names = None
    if speaker_names_json:
        with speaker_names_json.open("r", encoding="utf-8") as f:
            speaker_names = json.load(f)

    for split, split_ds in loaded.items():
        if "text_description" not in split_ds.column_names:
            raise SystemExit(f"{prompt_dir}/{split} has no text_description column")

        originals = split_ds["text_description"]
        rewritten = []
        fallback_count = 0
        for sample in split_ds:
            result = clean_sample(sample, speaker_names)
            rewritten.append(result["text_description"])
            fallback_count += int(result["_used_fallback"])
        changed = [idx for idx, (old, new) in enumerate(zip(originals, rewritten)) if old != new]
        suspicious_before = sum(1 for text in originals if _looks_suspicious(text))
        suspicious_after = sum(1 for text in rewritten if _looks_suspicious(text))
        total_changed += len(changed)
        total_fallbacks += fallback_count

        for idx in changed[: max(0, 4 - len(examples))]:
            examples.append((split, idx, originals[idx], rewritten[idx]))

        if dry_run:
            pass
        elif changed:
            cleaned[split] = split_ds.map(
                lambda batch: {
                    "text_description": [
                        clean_sample(dict(zip(batch, values)), speaker_names)["text_description"]
                        for values in zip(*batch.values())
                    ]
                },
                batched=True,
                desc=f"Cleaning {dataset}/{stage}/{split}",
            )
        else:
            cleaned[split] = split_ds

        print(
            f"[clean-llm] {dataset}/{stage}/{split}: changed {len(changed)} / {len(split_ds)} rows; "
            f"suspicious before={suspicious_before}, after={suspicious_after}; fallbacks={fallback_count}"
        )

    for split, idx, before, after in examples:
        print(f"[clean-llm] example {dataset}/{split}[{idx}]")
        print(f"  before: {_preview(before)}")
        print(f"  after:  {_preview(after)}")

    if dry_run:
        print(f"[clean-llm] dry run only, not writing {prompt_dir}")
        return

    if total_changed == 0:
        print(f"[clean-llm] no changes needed: {prompt_dir}")
        return

    if backup_dir.exists():
        if overwrite_backup:
            shutil.rmtree(backup_dir)
        else:
            print(f"[clean-llm] keeping existing backup: {backup_dir}")
    if not backup_dir.exists():
        shutil.copytree(prompt_dir, backup_dir)
        print(f"[clean-llm] backup: {backup_dir}")

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    cleaned.save_to_disk(str(tmp_dir))
    shutil.rmtree(prompt_dir)
    shutil.move(str(tmp_dir), str(prompt_dir))
    print(f"[clean-llm] wrote cleaned dataset: {prompt_dir} (fallbacks={total_fallbacks})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=os.environ.get("OUT_ROOT"),
        help="Pipeline output root. Defaults to OUT_ROOT.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset to clean. Can be repeated or comma-separated. Use greek_tts for female+male.",
    )
    parser.add_argument("--stage", default="04b_prompts_llm", help="Prompt stage to clean.")
    parser.add_argument(
        "--speaker-names-json",
        type=Path,
        help="Optional speaker_id -> name map for named datasets.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing.")
    parser.add_argument(
        "--overwrite-backup",
        action="store_true",
        help="Replace an existing *_before_clean backup.",
    )
    args = parser.parse_args()

    if not args.root:
        raise SystemExit("no --root given and OUT_ROOT is not set")

    for dataset in _parse_datasets(args.dataset):
        clean_dataset(
            Path(args.root),
            dataset,
            args.stage,
            args.speaker_names_json,
            args.dry_run,
            args.overwrite_backup,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
