"""Turn (metadata.csv + wavs/) into a HF datasets DatasetDict saved on disk.

CSV schema (as provided):
    filename, speaker_id, transcription, transcription_original,
    origin_dataset, gender

Output layout (save_to_disk format):
    $OUT_ROOT/<dataset>/01_hf_dataset/
        dataset_dict.json
        train/
            ...arrow shards...

Run on the login node after activating the conda env:

    source leonardo/env.sh && activate_conda_env
    python leonardo/login/03_build_hf_dataset.py --dataset female
    python leonardo/login/03_build_hf_dataset.py --dataset male
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _resolve_dataset_dir(name: str) -> Path:
    env_var = {"female": "FEMALE_DIR", "male": "MALE_DIR"}[name]
    value = os.environ.get(env_var)
    if not value:
        raise SystemExit(f"{env_var} not set; source leonardo/env.sh first")
    return Path(value)


def build(name: str, split: str = "train") -> None:
    import pandas as pd
    from datasets import Audio, Dataset, DatasetDict

    src = _resolve_dataset_dir(name)
    csv_name = os.environ.get("METADATA_CSV_NAME", "metadata.csv")
    wavs_subdir = os.environ.get("WAVS_SUBDIR", "wavs")
    out_root = os.environ.get("OUT_ROOT")
    if not out_root:
        raise SystemExit("OUT_ROOT not set; source leonardo/env.sh first")

    csv_path = src / csv_name
    wavs_dir = src / wavs_subdir
    out_dir = Path(out_root) / name / "01_hf_dataset"

    if not csv_path.is_file():
        raise SystemExit(f"missing CSV: {csv_path}")
    if not wavs_dir.is_dir():
        raise SystemExit(f"missing wavs dir: {wavs_dir}")

    print(f"[build] {name}: reading {csv_path}", flush=True)
    df = pd.read_csv(csv_path)

    expected = {"filename", "speaker_id", "transcription"}
    missing = expected - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing required columns: {missing} (have: {list(df.columns)})")

    # Resolve audio paths. The "filename" column may be a basename (e.g. "spk_001.wav")
    # or a relative path under wavs/; handle both.
    def to_audio_path(fn: str) -> str:
        p = Path(fn)
        if p.is_absolute():
            return str(p)
        candidate = wavs_dir / p
        if candidate.is_file():
            return str(candidate)
        # Fall back to looking under wavs/ assuming `fn` already starts with "wavs/".
        if p.parts and p.parts[0] == wavs_subdir:
            cand2 = src / p
            if cand2.is_file():
                return str(cand2)
        return str(candidate)  # let HF surface the error if it really is missing

    df["audio"] = df["filename"].astype(str).map(to_audio_path)

    # Rename transcription -> text so main.py defaults work without --rename_column.
    df = df.rename(columns={"transcription": "text"})

    # Backfill gender from folder name if missing/empty.
    folder_gender = "female" if name == "female" else "male"
    if "gender" not in df.columns:
        df["gender"] = folder_gender
    else:
        df["gender"] = df["gender"].fillna(folder_gender)
        df.loc[df["gender"].astype(str).str.strip() == "", "gender"] = folder_gender

    print(f"[build] {name}: {len(df)} rows, columns={list(df.columns)}", flush=True)

    # Sanity-check a handful of audio paths so users learn early if filename
    # mapping is wrong (rather than after main.py runs for an hour).
    missing_audio = [p for p in df["audio"].head(20) if not Path(p).is_file()]
    if missing_audio:
        print("[build] WARNING: some audio paths don't exist, first 5:", flush=True)
        for p in missing_audio[:5]:
            print(f"   {p}", flush=True)
        print("[build] (continuing — datasets.Audio will lazily decode)", flush=True)

    ds = Dataset.from_pandas(df, preserve_index=False)
    ds = ds.cast_column("audio", Audio())  # sample rate left native; main.py will resample.

    dd = DatasetDict({split: ds})
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[build] {name}: writing to {out_dir}", flush=True)
    dd.save_to_disk(str(out_dir))
    print(f"[build] {name}: done. Verify with `python -c \"from datasets import load_from_disk; "
          f"print(load_from_disk('{out_dir}'))\"`", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["female", "male", "both"], required=True)
    parser.add_argument("--split-name", default="train",
                        help="Name of the split inside the DatasetDict (default: train).")
    args = parser.parse_args()

    targets = ["female", "male"] if args.dataset == "both" else [args.dataset]
    for name in targets:
        build(name, split=args.split_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
