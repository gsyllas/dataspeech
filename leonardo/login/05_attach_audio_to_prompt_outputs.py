"""Attach the original audio column back to prompt datasets.

main.py intentionally drops the audio column while computing tags to avoid
rewriting large audio payloads. That is good for the annotation pipeline, but
Parler-TTS training wants the final prompt dataset to contain audio too.

This script joins:

    <root>/<dataset>/01_hf_dataset      (source audio)
    <root>/<dataset>/<prompt-stage>     (text descriptions)

by split and row order, then writes a sibling directory:

    <root>/<dataset>/<prompt-stage>_with_audio

Run on a login node after the prompt stages finish.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from datasets import Audio, DatasetDict, concatenate_datasets, load_from_disk


PROMPT_STAGES = ("04b_prompts_llm", "04a_prompts_deterministic")


def attach_audio(root: Path, dataset: str, stage: str, overwrite: bool = False) -> Path:
    src_dir = root / dataset / "01_hf_dataset"
    prompt_dir = root / dataset / stage
    out_dir = root / dataset / f"{stage}_with_audio"

    if not src_dir.is_dir():
        raise SystemExit(f"missing source dataset: {src_dir}")
    if not prompt_dir.is_dir():
        raise SystemExit(f"missing prompt dataset: {prompt_dir}")
    if out_dir.exists():
        if not overwrite:
            print(f"[attach-audio] exists, skipping: {out_dir}")
            return out_dir
        shutil.rmtree(out_dir)

    source = load_from_disk(str(src_dir))
    prompts = load_from_disk(str(prompt_dir))

    if not isinstance(source, DatasetDict) or not isinstance(prompts, DatasetDict):
        raise SystemExit("expected both source and prompt datasets to be DatasetDict objects")

    repaired = DatasetDict()
    for split, prompt_split in prompts.items():
        if split not in source:
            raise SystemExit(f"split {split!r} is in prompts but not in {src_dir}")
        source_split = source[split]
        if len(source_split) != len(prompt_split):
            raise SystemExit(
                f"split {split!r} length mismatch: source={len(source_split)} prompts={len(prompt_split)}"
            )
        if "audio" in prompt_split.column_names:
            repaired[split] = prompt_split
            print(f"[attach-audio] {dataset}/{stage}/{split}: prompt already has audio")
            continue
        if "audio" not in source_split.column_names:
            raise SystemExit(f"split {split!r} in {src_dir} has no audio column")

        # Keep paths/bytes lazy instead of decoding arrays while we join.
        audio_only = source_split.select_columns(["audio"]).cast_column("audio", Audio(decode=False))
        repaired[split] = concatenate_datasets([prompt_split, audio_only], axis=1)
        print(f"[attach-audio] {dataset}/{stage}/{split}: added audio to {len(prompt_split)} rows")

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    repaired.save_to_disk(str(out_dir))
    print(f"[attach-audio] wrote {out_dir}")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.environ.get("NAMED_OUT_ROOT") or os.environ.get("OUT_ROOT"),
                        help="Pipeline output root. Defaults to NAMED_OUT_ROOT, then OUT_ROOT.")
    parser.add_argument("--dataset", default="multi",
                        help="Dataset directory under root. Default: multi.")
    parser.add_argument("--stage", choices=[*PROMPT_STAGES, "all"], default="04b_prompts_llm",
                        help="Prompt stage to repair. Default: 04b_prompts_llm.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing *_with_audio output.")
    args = parser.parse_args()

    if not args.root:
        raise SystemExit("no --root given and neither NAMED_OUT_ROOT nor OUT_ROOT is set")

    root = Path(args.root)
    stages = PROMPT_STAGES if args.stage == "all" else (args.stage,)
    for stage in stages:
        attach_audio(root, args.dataset, stage, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
