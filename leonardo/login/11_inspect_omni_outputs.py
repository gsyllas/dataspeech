"""Eyeball the Qwen-Omni descriptions after a stage-45 run.

Prints, per split: row count, how many rows Omni flagged as failed
(`_omni_failed`), how many are empty, how many still look non-English (a
guardrail backstop), basic length stats, and a handful of random examples.

Run on a login node (needs only `datasets`):

    source leonardo/env.sh && activate_omni_conda_env   # or activate_conda_env
    python leonardo/login/11_inspect_omni_outputs.py --dataset greek_tts
    python leonardo/login/11_inspect_omni_outputs.py \
        --root "$NAMED_OUT_ROOT" --dataset multi_v2 --num-samples 12
"""

from __future__ import annotations

import argparse
import os
import random
import re
import statistics
from pathlib import Path

from datasets import load_from_disk


DEFAULT_DATASETS = ("greek_female_tts", "greek_male_tts")

# Non-Latin script ranges; presence of any of these means "not English".
_NON_LATIN_RANGES = (
    (0x0370, 0x03FF), (0x1F00, 0x1FFF),  # Greek
    (0x0400, 0x052F),                    # Cyrillic
    (0x0590, 0x05FF), (0x0600, 0x06FF),  # Hebrew, Arabic
    (0x3040, 0x30FF), (0x3400, 0x9FFF),  # Kana, CJK
    (0xAC00, 0xD7AF),                    # Hangul
)


def looks_non_english(text: str) -> bool:
    if not text:
        return False
    for ch in text:
        code = ord(ch)
        for lo, hi in _NON_LATIN_RANGES:
            if lo <= code <= hi:
                return True
    return False


def _parse_datasets(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return DEFAULT_DATASETS
    out: list[str] = []
    for value in values:
        for item in value.replace(",", " ").split():
            if item == "greek_tts":
                out.extend(DEFAULT_DATASETS)
            else:
                out.append(item)
    seen, unique = set(), []
    for d in out:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return tuple(unique)


def _preview(text: str, limit: int = 200) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def inspect(root: Path, dataset: str, stage: str, num_samples: int, rng: random.Random) -> None:
    prompt_dir = root / dataset / stage
    print(f"\n=== {dataset}/{stage} ===")
    if not prompt_dir.is_dir():
        print(f"  MISSING: {prompt_dir}")
        return

    ds = load_from_disk(str(prompt_dir))
    splits = ds.keys() if hasattr(ds, "keys") else {"<single>": ds}
    for split in splits:
        split_ds = ds[split] if hasattr(ds, "keys") else ds
        cols = split_ds.column_names
        if "text_description" not in cols:
            print(f"  {split}: no text_description column (cols={cols})")
            continue

        texts = split_ds["text_description"]
        n = len(texts)
        flagged = split_ds["_omni_failed"] if "_omni_failed" in cols else [False] * n
        n_failed = sum(bool(x) for x in flagged)
        n_empty = sum(1 for t in texts if not str(t).strip())
        n_non_en = sum(1 for t in texts if looks_non_english(t))
        lengths = [len(str(t)) for t in texts if str(t).strip()]
        mean_len = round(statistics.mean(lengths)) if lengths else 0

        print(
            f"  {split}: {n} rows | flagged _omni_failed={n_failed} "
            f"({100 * n_failed / max(n, 1):.1f}%) | empty={n_empty} | "
            f"non-English leak={n_non_en} | mean len(chars)={mean_len}"
        )
        if n_non_en:
            print("    WARNING: non-English text present despite guardrails; "
                  "inspect and consider raising OMNI_NUM_RETRIES.")

        # Show random NON-empty examples (the ones you actually care about).
        good_idx = [i for i in range(n) if str(texts[i]).strip()]
        sample_idx = rng.sample(good_idx, min(num_samples, len(good_idx))) if good_idx else []
        for i in sorted(sample_idx):
            gender = split_ds[i].get("gender") if "gender" in cols else None
            spk = split_ds[i].get("speaker_id") if "speaker_id" in cols else None
            tag = " [FAILED]" if flagged[i] else ""
            meta = ", ".join(p for p in (f"gender={gender}" if gender else "",
                                         f"spk={spk}" if spk else "") if p)
            print(f"    [{i}]{tag} {meta}")
            print(f"        {_preview(texts[i])}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.environ.get("OUT_ROOT"),
                        help="Output root. Defaults to OUT_ROOT. Use NAMED_OUT_ROOT for multi/multi_v2.")
    parser.add_argument("--dataset", action="append",
                        help="Dataset(s). Repeatable/comma-separated. 'greek_tts' = female+male.")
    parser.add_argument("--stage", default="04c_prompts_omni")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if not args.root:
        raise SystemExit("no --root given and OUT_ROOT is not set")

    rng = random.Random(args.seed)
    for dataset in _parse_datasets(args.dataset):
        inspect(Path(args.root), dataset, args.stage, args.num_samples, rng)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
