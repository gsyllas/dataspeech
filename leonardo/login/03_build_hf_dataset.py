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
    python leonardo/login/03_build_hf_dataset.py --dataset multi

For the named-speaker variant, prefer:

    bash leonardo/login/04_prepare_named_variant.sh
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path


def _resolve_dataset_dir(name: str) -> Path:
    env_var = {"female": "FEMALE_DIR", "male": "MALE_DIR", "multi": "MULTI_SPEAKER_DIR"}[name]
    value = os.environ.get(env_var)
    if not value:
        raise SystemExit(f"{env_var} not set; source leonardo/env.sh first")
    return Path(value)


FEMALE_NAMES = [
    "Maria",
    "Sofia",
    "Anna",
    "Katerina",
    "Dimitra",
    "Ioanna",
    "Georgia",
    "Vasiliki",
    "Christina",
    "Alexandra",
    "Niki",
    "Irini",
]

MALE_NAMES = [
    "Giorgos",
    "Kostas",
    "Dimitris",
    "Andreas",
    "Panagiotis",
    "Petros",
    "Christos",
    "Manolis",
    "Vasilis",
    "Spyros",
    "Stefanos",
    "Alexandros",
    "Michalis",
    "Antonis",
    "Ioannis",
    "Thanasis",
    "Pavlos",
    "Leonidas",
    "Theodoros",
    "Haris",
    "Aris",
    "Stavros",
    "Marios",
    "Ilias",
    "Apostolos",
    "Filippos",
    "Grigoris",
    "Sotiris",
    "Aggelos",
    "Lefteris",
    "Kyriakos",
    "Dionysis",
    "Paris",
    "Markos",
    "Alkis",
    "Fotis",
    "Achilleas",
    "Orestis",
    "Lambros",
    "Miltiadis",
    "Periklis",
    "Stelios",
    "Yannis",
    "Alekos",
    "Babis",
    "Makis",
    "Takis",
    "Miltos",
    "Sakis",
    "Lakis",
    "Nondas",
    "Pantelis",
    "Vangelis",
    "Minas",
    "Dionisios",
    "Nikiforos",
    "Lazaros",
    "Aimilios",
    "Damianos",
    "Evangelos",
    "Iasonas",
    "Linos",
    "Thodoris",
    "Rafail",
]


def _speaker_gender(speaker_key: str) -> str:
    key = speaker_key.lower()
    if key == "female" or key.startswith("cs10") or key.startswith("css10"):
        return "female"
    return "male"


def _load_multi_speaker_stats(src: Path) -> dict:
    stats_path = Path(os.environ.get("MULTI_SPEAKER_STATS_JSON", src / "dataset_stats.json"))
    if not stats_path.is_file():
        raise SystemExit(f"missing multi-speaker stats JSON: {stats_path}")
    with stats_path.open() as f:
        stats = json.load(f)
    per_speaker = stats.get("per_speaker")
    if not isinstance(per_speaker, dict):
        raise SystemExit(f"{stats_path} does not contain a per_speaker object")
    return per_speaker


def _eligible_multi_speakers(src: Path, min_hours: float) -> dict:
    per_speaker = _load_multi_speaker_stats(src)
    eligible = {
        speaker_key: info
        for speaker_key, info in per_speaker.items()
        if float(info.get("duration_hours", 0.0)) >= min_hours
    }
    if not eligible:
        raise SystemExit(f"no multi-speaker speakers found with >= {min_hours} hours")
    return eligible


def _assign_speaker_names(eligible: dict) -> dict:
    seed = int(os.environ.get("NAMED_SPEAKER_NAME_SEED", "1337"))
    rng = random.Random(seed)
    standalone_female_name = os.environ.get("NAMED_FEMALE_SPEAKER_NAME", "Eleni")
    standalone_male_name = os.environ.get("NAMED_MALE_SPEAKER_NAME", "Nikos")

    reserved = {standalone_female_name, standalone_male_name}
    female_names = [name for name in FEMALE_NAMES if name not in reserved]
    male_names = [name for name in MALE_NAMES if name not in reserved]
    rng.shuffle(female_names)
    rng.shuffle(male_names)

    assigned = {}
    female_i = 0
    male_i = 0
    for speaker_key, info in sorted(eligible.items(), key=lambda item: int(item[1].get("speaker_id", 10**9))):
        gender = _speaker_gender(speaker_key)
        if speaker_key == "female":
            name = standalone_female_name
        elif speaker_key == "male":
            name = standalone_male_name
        elif gender == "female":
            if female_i >= len(female_names):
                raise SystemExit("not enough unique female names in 03_build_hf_dataset.py")
            name = female_names[female_i]
            female_i += 1
        else:
            if male_i >= len(male_names):
                raise SystemExit("not enough unique male names in 03_build_hf_dataset.py")
            name = male_names[male_i]
            male_i += 1
        assigned[speaker_key] = name
    return assigned


def _filter_multi_speaker_df(df, src: Path, min_hours: float, speaker_names_json: str | None):
    eligible = _eligible_multi_speakers(src, min_hours)
    names_by_key = _assign_speaker_names(eligible)

    id_to_key = {str(info.get("speaker_id")): speaker_key for speaker_key, info in eligible.items()}
    eligible_values = set(id_to_key) | set(eligible)
    speaker_values = df["speaker_id"].astype(str)
    before = len(df)
    df = df[speaker_values.isin(eligible_values)].copy()
    after = len(df)
    if after == 0:
        raise SystemExit(
            "multi-speaker filter produced 0 rows; expected metadata speaker_id "
            "to match either dataset_stats speaker_id values or per_speaker keys"
        )

    def speaker_key_for_value(value: object) -> str | None:
        value = str(value)
        return id_to_key.get(value, value if value in eligible else None)

    key_by_observed_value = {
        value: speaker_key_for_value(value)
        for value in sorted(df["speaker_id"].astype(str).unique())
    }
    name_map = {
        value: names_by_key[key]
        for value, key in key_by_observed_value.items()
        if key is not None
    }
    gender_map = {
        value: _speaker_gender(key)
        for value, key in key_by_observed_value.items()
        if key is not None
    }

    df["gender"] = df["speaker_id"].astype(str).map(gender_map).fillna(df.get("gender", ""))

    if speaker_names_json:
        path = Path(speaker_names_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(name_map, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"[build] multi: wrote speaker-name map to {path}", flush=True)

    kept_summary = [
        f"{key}={names_by_key[key]}({float(info.get('duration_hours', 0.0)):.2f}h)"
        for key, info in sorted(eligible.items(), key=lambda item: int(item[1].get("speaker_id", 10**9)))
    ]
    print(f"[build] multi: kept {len(eligible)} speakers >= {min_hours}h: {', '.join(kept_summary)}", flush=True)
    print(f"[build] multi: filtered {before} -> {after} rows", flush=True)
    return df


def build(name: str, split: str = "train", multi_min_speaker_hours: float = 1.0,
          speaker_names_json: str | None = None) -> None:
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
    if name == "multi":
        df = _filter_multi_speaker_df(df, src, multi_min_speaker_hours, speaker_names_json)
    else:
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
    parser.add_argument("--dataset", choices=["female", "male", "multi", "both", "all"], required=True)
    parser.add_argument("--split-name", default="train",
                        help="Name of the split inside the DatasetDict (default: train).")
    parser.add_argument("--multi-min-speaker-hours", type=float,
                        default=float(os.environ.get("MULTI_MIN_SPEAKER_HOURS", "1.0")),
                        help="For --dataset multi/all, keep speakers with at least this many hours.")
    parser.add_argument("--speaker-names-json",
                        default=os.environ.get("NAMED_MULTI_SPEAKER_NAMES_JSON"),
                        help="For --dataset multi/all, write speaker_id -> display-name JSON here.")
    args = parser.parse_args()

    if args.dataset == "both":
        targets = ["female", "male"]
    elif args.dataset == "all":
        targets = ["female", "male", "multi"]
    else:
        targets = [args.dataset]
    for name in targets:
        build(
            name,
            split=args.split_name,
            multi_min_speaker_hours=args.multi_min_speaker_hours,
            speaker_names_json=args.speaker_names_json if name == "multi" else None,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
