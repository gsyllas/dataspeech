"""Clean small LLM decode artifacts from prompt descriptions.

This is intended for already-built prompt datasets such as:

    <OUT_ROOT>/greek_female_tts/04b_prompts_llm
    <OUT_ROOT>/greek_male_tts/04b_prompts_llm

The script writes a sibling backup before replacing the prompt dataset:

    04b_prompts_llm_before_clean
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

from datasets import DatasetDict, load_from_disk


DEFAULT_DATASETS = ("greek_female_tts", "greek_male_tts")
DESCRIPTION_START_PATTERNS = (
    re.compile(r"\bA\s+(?:female|male|woman|man|speaker)\b", re.IGNORECASE),
    re.compile(r"\bAn\s+(?:adult\s+)?(?:female|male|woman|man|speaker)\b", re.IGNORECASE),
    re.compile(r"\bThe\s+(?:speaker|recording|voice|audio)\b", re.IGNORECASE),
    re.compile(r"\bIn\s+(?:a|an)\s+", re.IGNORECASE),
    re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+(?:speaks|delivers|has|sounds)\b"),
)


def clean_description(text: str) -> str:
    """Remove short prompt/chat-template scraps before the real description."""
    if text is None:
        return ""

    text = str(text).replace("\x00", "").strip()
    text = re.sub(r"<\|/?(?:assistant|user|system|im_start|im_end)\|>", " ", text)
    text = re.sub(r"^(?:assistant|response|answer)\s*[:\-]\s*", "", text, flags=re.IGNORECASE)

    lines = [line.strip(" \t\r\n\"'`") for line in text.splitlines() if line.strip()]
    while len(lines) > 1 and len(lines[0]) <= 24:
        first = lines[0].lstrip(":-,.;!?\"'` ")
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
        if "\n" in prefix or len(prefix) <= 80 or not re.search(r"[A-Za-z]{3,}", prefix):
            text = text[earliest:]

    return re.sub(r"\s+", " ", text).strip(" \t\r\n\"'`")


def _starts_description(text: str) -> bool:
    return any(
        (match := pattern.search(text)) is not None and match.start() == 0
        for pattern in DESCRIPTION_START_PATTERNS
    )


def _preview(text: str, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text[:limit] + ("..." if len(text) > limit else "")


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
    examples: list[tuple[str, int, str, str]] = []

    for split, split_ds in loaded.items():
        if "text_description" not in split_ds.column_names:
            raise SystemExit(f"{prompt_dir}/{split} has no text_description column")

        originals = split_ds["text_description"]
        rewritten = [clean_description(text) for text in originals]
        changed = [idx for idx, (old, new) in enumerate(zip(originals, rewritten)) if old != new]
        total_changed += len(changed)

        for idx in changed[: max(0, 4 - len(examples))]:
            examples.append((split, idx, originals[idx], rewritten[idx]))

        if dry_run:
            pass
        elif changed:
            cleaned[split] = split_ds.map(
                lambda batch: {"text_description": [clean_description(text) for text in batch["text_description"]]},
                batched=True,
                desc=f"Cleaning {dataset}/{stage}/{split}",
            )
        else:
            cleaned[split] = split_ds

        print(f"[clean-llm] {dataset}/{stage}/{split}: changed {len(changed)} / {len(split_ds)} rows")

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
    print(f"[clean-llm] wrote cleaned dataset: {prompt_dir}")


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
        clean_dataset(Path(args.root), dataset, args.stage, args.dry_run, args.overwrite_backup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
