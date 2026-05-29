"""Turn local Leonardo sources into HF datasets DatasetDict objects.

Legacy CSV schema:
    filename, speaker_id, transcription, transcription_original,
    origin_dataset, gender

Greek pipe TTS schema:
    wav_id|text|speaker|source|duration_s

Greek HF sources:
    commonVoice_greek_clean_with_speaker_ids, cs10_greek_dataset, and
    greek_male_3.5h are expected to be local HF parquet dataset repos or
    save_to_disk datasets.

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
    python leonardo/login/03_build_hf_dataset.py --dataset greek_tts

For the named-speaker variant, prefer:

    bash leonardo/login/04_prepare_named_variant.sh multi_v2
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path


def _resolve_dataset_dir(name: str) -> Path:
    env_var = {
        "female": "FEMALE_DIR",
        "male": "MALE_DIR",
        "multi": "MULTI_SPEAKER_DIR",
        "commonVoice_greek_clean_with_speaker_ids": "COMMONVOICE_GREEK_DIR",
        "cs10_greek_dataset": "CS10_GREEK_DIR",
        "greek_male_3.5h": "GREEK_MALE_35H_DIR",
        "greek_female_tts": "GREEK_FEMALE_TTS_DIR",
        "greek_male_tts": "GREEK_MALE_TTS_DIR",
        "multi_v2": "GREEK_DATA_ROOT",
    }[name]
    value = os.environ.get(env_var)
    if not value:
        raise SystemExit(f"{env_var} not set; source leonardo/env.sh first")
    return Path(value)


GREEK_HF_DATASETS = {
    "commonVoice_greek_clean_with_speaker_ids",
    "cs10_greek_dataset",
    "greek_male_3.5h",
}
GREEK_PIPE_TTS_DATASETS = {
    "greek_female_tts": "female",
    "greek_male_tts": "male",
}
MULTI_V2_COMPONENTS = [
    "commonVoice_greek_clean_with_speaker_ids",
    "cs10_greek_dataset",
    "greek_male_3.5h",
    "greek_female_tts",
    "greek_male_tts",
]
TEXT_COLUMN_CANDIDATES = (
    "text",
    "transcription_normalised",
    "transcription_normalized",
    "normalized_text",
    "transcription",
    "sentence",
    "transcript",
    "transcription_original",
)
SPEAKER_COLUMN_CANDIDATES = (
    "speaker_id",
    "speaker",
    "speaker_name",
    "client_id",
    "reader_id",
)
SOURCE_COLUMN_CANDIDATES = ("source", "origin_dataset", "dataset")
DURATION_COLUMN_CANDIDATES = ("duration_s", "duration", "duration_seconds")
NORMALIZED_COLUMNS = [
    "audio",
    "text",
    "speaker_id",
    "gender",
    "origin_dataset",
    "source",
    "duration_s",
    "source_speaker_id",
]


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

GREEK_SURNAMES = [
    "Papadopoulos",
    "Papadopoulou",
    "Georgiou",
    "Nikolaou",
    "Dimitriou",
    "Konstantinou",
    "Ioannou",
    "Pappas",
    "Karagiannis",
    "Vasileiou",
    "Antoniou",
    "Anagnostou",
    "Athanasiou",
    "Christou",
    "Alexiou",
    "Markou",
    "Stavrou",
    "Lambrou",
    "Sotiriou",
    "Petrou",
    "Oikonomou",
    "Makris",
    "Mavridis",
    "Theodorou",
    "Panagiotou",
    "Katsaros",
    "Raptis",
    "Michailidis",
    "Zervas",
    "Galanis",
    "Kouris",
    "Lykos",
    "Rallis",
    "Fotiou",
]


def _make_name_pool(gender: str, reserved: set[str], rng: random.Random) -> list[str]:
    first_names = FEMALE_NAMES if gender == "female" else MALE_NAMES
    simple = [name for name in first_names if name not in reserved]
    compound = [
        f"{first} {surname}"
        for first in first_names
        for surname in GREEK_SURNAMES
        if f"{first} {surname}" not in reserved
    ]
    rng.shuffle(simple)
    rng.shuffle(compound)
    return simple + compound


def _pick_name(pool: list[str], index: int, gender: str) -> str:
    if index < len(pool):
        return pool[index]
    fallback = FEMALE_NAMES if gender == "female" else MALE_NAMES
    return f"{fallback[index % len(fallback)]} {index + 1:03d}"


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
    female_names = _make_name_pool("female", reserved, rng)
    male_names = _make_name_pool("male", reserved, rng)

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
            name = _pick_name(female_names, female_i, "female")
            female_i += 1
        else:
            name = _pick_name(male_names, male_i, "male")
            male_i += 1
        assigned[speaker_key] = name
    return assigned


def _multi_speaker_maps(src: Path, min_hours: float, speaker_names_json: str | None):
    eligible = _eligible_multi_speakers(src, min_hours)
    names_by_key = _assign_speaker_names(eligible)
    id_to_key = {str(info.get("speaker_id")): speaker_key for speaker_key, info in eligible.items()}

    # Include both forms because the saved DatasetDict may store speaker_id as
    # the numeric id (e.g. 5), while the stats JSON is keyed by names
    # (e.g. cv_speaker_0). run_prompt_creation.py also stringifies speaker_id.
    name_map = {}
    gender_map = {}
    for speaker_key, info in eligible.items():
        values = {speaker_key, str(info.get("speaker_id"))}
        for value in values:
            name_map[value] = names_by_key[speaker_key]
            gender_map[value] = _speaker_gender(speaker_key)

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
    return eligible, name_map, gender_map


def _filter_multi_speaker_df(df, src: Path, min_hours: float, speaker_names_json: str | None):
    _, name_map, gender_map = _multi_speaker_maps(src, min_hours, speaker_names_json)
    eligible_values = set(name_map)
    speaker_values = df["speaker_id"].astype(str)
    before = len(df)
    df = df[speaker_values.isin(eligible_values)].copy()
    after = len(df)
    if after == 0:
        raise SystemExit(
            "multi-speaker filter produced 0 rows; expected metadata speaker_id "
            "to match either dataset_stats speaker_id values or per_speaker keys"
        )

    df["gender"] = df["speaker_id"].astype(str).map(gender_map).fillna(df.get("gender", ""))
    print(f"[build] multi: filtered {before} -> {after} rows", flush=True)
    return df


def _build_multi_from_saved_dataset(src: Path, out_dir: Path, min_hours: float,
                                    speaker_names_json: str | None) -> None:
    from datasets import DatasetDict, load_from_disk

    _, name_map, gender_map = _multi_speaker_maps(src, min_hours, speaker_names_json)
    eligible_values = set(name_map)

    print(f"[build] multi: loading saved dataset from {src}", flush=True)
    raw = load_from_disk(str(src))
    if not isinstance(raw, DatasetDict):
        raw = DatasetDict({"train": raw})

    processed = DatasetDict()
    for split_name, ds in raw.items():
        if "speaker_id" not in ds.column_names:
            raise SystemExit(f"multi split {split_name!r} missing speaker_id column: {ds.column_names}")
        if "text" not in ds.column_names and "transcription" in ds.column_names:
            ds = ds.rename_column("transcription", "text")

        before = len(ds)
        ds = ds.filter(
            lambda speaker_id: str(speaker_id) in eligible_values,
            input_columns=["speaker_id"],
        )
        after = len(ds)
        if after == 0:
            raise SystemExit(
                f"multi split {split_name!r} filter produced 0 rows; expected speaker_id "
                "to match dataset_stats speaker_id values or per_speaker keys"
            )

        ds = ds.map(
            lambda speaker_ids: {"gender": [gender_map.get(str(value), "") for value in speaker_ids]},
            batched=True,
            input_columns=["speaker_id"],
        )
        processed[split_name] = ds
        print(f"[build] multi/{split_name}: filtered {before} -> {after} rows", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[build] multi: writing to {out_dir}", flush=True)
    processed.save_to_disk(str(out_dir))
    print(f"[build] multi: done. Verify with `python -c \"from datasets import load_from_disk; "
          f"print(load_from_disk('{out_dir}'))\"`", flush=True)


def _collapse_ws(value) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _first_existing(columns, candidates) -> str | None:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def _to_float_or_none(value):
    if value is None:
        return float("nan")
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _normalize_gender_value(value, default_gender: str) -> str:
    text = str(value or "").strip().lower()
    female_values = {"female", "f", "woman", "women", "girl", "female_01"}
    male_values = {"male", "m", "man", "men", "boy", "male_01"}
    if text in female_values or text.startswith("female"):
        return "female"
    if text in male_values or text.startswith("male"):
        return "male"
    return default_gender


def _default_gender_for_dataset(name: str) -> str:
    key = name.lower()
    if "female" in key or key.startswith("cs10") or key.startswith("css10"):
        return "female"
    return "male"


def _load_source_gender_payload(name: str) -> dict:
    if name != "commonVoice_greek_clean_with_speaker_ids":
        return {}

    path_value = os.environ.get("COMMONVOICE_GREEK_GENDER_JSON", "")
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.is_file():
        print(f"[build] {name}: no speaker gender map at {path}; using dataset/default genders", flush=True)
        return {}

    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} must be a JSON object")

    if isinstance(payload.get("speaker_genders"), dict):
        speaker_genders = payload["speaker_genders"]
    else:
        speaker_genders = {
            key: value for key, value in payload.items()
            if isinstance(value, str) and key not in {"gender_column", "speaker_id_column"}
        }
    payload["speaker_genders"] = {
        str(speaker_id): _normalize_gender_value(gender, "male")
        for speaker_id, gender in speaker_genders.items()
    }
    payload["row_genders"] = {
        str(key): _normalize_gender_value(gender, "male")
        for key, gender in payload.get("row_genders", {}).items()
    }
    payload["conflicting_speakers"] = {
        str(key): value for key, value in payload.get("conflicting_speakers", {}).items()
    }
    print(
        f"[build] {name}: loaded {len(payload['speaker_genders'])} speaker genders "
        f"and {len(payload['row_genders'])} row genders from {path}",
        flush=True,
    )
    return payload


def _load_hf_dataset_dict(name: str, src: Path):
    from datasets import DatasetDict, load_dataset, load_from_disk

    if not src.is_dir():
        raise SystemExit(f"{name}: missing HF dataset directory: {src}")

    if (src / "dataset_dict.json").is_file():
        print(f"[build] {name}: loading save_to_disk dataset from {src}", flush=True)
        raw = load_from_disk(str(src))
    else:
        data_dir = src / "data"
        parquet_files = sorted(data_dir.glob("*.parquet")) if data_dir.is_dir() else []
        if not parquet_files:
            raise SystemExit(
                f"{name}: expected either dataset_dict.json or parquet files under {data_dir}"
            )
        data_files = defaultdict(list)
        for parquet_file in parquet_files:
            split_name = parquet_file.name.split("-", 1)[0]
            data_files[split_name].append(str(parquet_file))
        print(f"[build] {name}: loading local parquet dataset from {src}", flush=True)
        raw = load_dataset("parquet", data_files=dict(data_files))

    if not isinstance(raw, DatasetDict):
        raw = DatasetDict({"train": raw})
    return raw


def _normalize_hf_split(ds, name: str, prefix_speaker_ids: bool):
    from datasets import Audio

    if "audio" not in ds.column_names:
        raise SystemExit(f"{name}: expected an audio column, got {ds.column_names}")

    text_col = _first_existing(ds.column_names, TEXT_COLUMN_CANDIDATES)
    if text_col is None:
        raise SystemExit(f"{name}: expected one of {TEXT_COLUMN_CANDIDATES}, got {ds.column_names}")
    if text_col != "text":
        ds = ds.rename_column(text_col, "text")

    speaker_col = _first_existing(ds.column_names, SPEAKER_COLUMN_CANDIDATES)
    gender_col = "gender" if "gender" in ds.column_names else None
    source_col = _first_existing(ds.column_names, SOURCE_COLUMN_CANDIDATES)
    duration_col = _first_existing(ds.column_names, DURATION_COLUMN_CANDIDATES)
    origin_col = "origin_dataset" if "origin_dataset" in ds.column_names else None
    default_gender = _default_gender_for_dataset(name)
    gender_payload = _load_source_gender_payload(name)
    speaker_gender_map = gender_payload.get("speaker_genders", {})
    row_gender_map = gender_payload.get("row_genders", {})
    row_gender_key_columns = gender_payload.get("row_gender_key_columns", [])
    conflicting_speakers = set(gender_payload.get("conflicting_speakers", {}))

    def add_normalized_columns(batch):
        n = len(batch["text"])
        raw_speakers = batch[speaker_col] if speaker_col else [name] * n
        raw_genders = batch[gender_col] if gender_col else [default_gender] * n
        raw_sources = batch[source_col] if source_col else [name] * n
        raw_origins = batch[origin_col] if origin_col else [name] * n
        raw_durations = batch[duration_col] if duration_col else [None] * n

        genders = []
        source_speakers = []
        row_key_available = row_gender_map and all(column in batch for column in row_gender_key_columns)
        for index, (raw_speaker_value, raw_gender) in enumerate(zip(raw_speakers, raw_genders)):
            raw_speaker = _collapse_ws(raw_speaker_value) or name
            row_gender = None
            if row_key_available:
                row_key = "\t".join(_collapse_ws(batch[column][index]) for column in row_gender_key_columns)
                row_gender = row_gender_map.get(row_key)
            mapped_gender = (
                row_gender
                or speaker_gender_map.get(str(raw_speaker))
                or speaker_gender_map.get(f"{name}:{raw_speaker}")
            )
            gender = _normalize_gender_value(mapped_gender or raw_gender, default_gender)
            genders.append(gender)
            if raw_speaker in conflicting_speakers and row_gender:
                source_speakers.append(f"{raw_speaker}:{gender}")
            else:
                source_speakers.append(raw_speaker)

        if prefix_speaker_ids:
            speaker_ids = [f"{name}:{speaker}" for speaker in source_speakers]
        else:
            speaker_ids = source_speakers

        return {
            "text": [_collapse_ws(value) for value in batch["text"]],
            "source_speaker_id": source_speakers,
            "speaker_id": speaker_ids,
            "gender": genders,
            "origin_dataset": [_collapse_ws(value) or name for value in raw_origins],
            "source": [_collapse_ws(value) or name for value in raw_sources],
            "duration_s": [_to_float_or_none(value) for value in raw_durations],
        }

    ds = ds.map(add_normalized_columns, batched=True, desc=f"Normalizing {name}")
    ds = ds.cast_column("audio", Audio())
    missing = [column for column in NORMALIZED_COLUMNS if column not in ds.column_names]
    if missing:
        raise SystemExit(f"{name}: normalized split is missing columns {missing}")
    return ds.select_columns(NORMALIZED_COLUMNS)


def _normalize_hf_dataset_dict(name: str, raw, prefix_speaker_ids: bool):
    from datasets import DatasetDict

    processed = DatasetDict()
    for split_name, ds in raw.items():
        before = len(ds)
        processed[split_name] = _normalize_hf_split(ds, name, prefix_speaker_ids)
        print(f"[build] {name}/{split_name}: normalized {before} rows", flush=True)
    return processed


def _pipe_tts_dataset_dict(name: str, src: Path, split: str, prefix_speaker_ids: bool):
    import pandas as pd
    from datasets import Audio, Dataset, DatasetDict

    csv_path = src / os.environ.get("METADATA_CSV_NAME", "metadata.csv")
    wavs_dir = src / os.environ.get("WAVS_SUBDIR", "wavs")
    if not csv_path.is_file():
        raise SystemExit(f"{name}: missing pipe metadata CSV: {csv_path}")
    if not wavs_dir.is_dir():
        raise SystemExit(f"{name}: missing wavs dir: {wavs_dir}")

    print(f"[build] {name}: reading pipe metadata from {csv_path}", flush=True)
    df = pd.read_csv(
        csv_path,
        sep="|",
        header=None,
        names=["wav_id", "text", "speaker", "source", "duration_s"],
        encoding="utf-8",
        dtype={"wav_id": str, "text": str, "speaker": str, "source": str},
    )
    gender = GREEK_PIPE_TTS_DATASETS[name]

    def to_audio_path(wav_id: str) -> str:
        rel = Path(str(wav_id))
        if rel.suffix.lower() != ".wav":
            rel = rel.with_suffix(".wav")
        return str(wavs_dir / rel)

    df["audio"] = df["wav_id"].map(to_audio_path)
    df["text"] = df["text"].map(_collapse_ws)
    df["source_speaker_id"] = df["speaker"].map(lambda value: _collapse_ws(value) or f"{gender}_01")
    if prefix_speaker_ids:
        df["speaker_id"] = df["source_speaker_id"].map(lambda value: f"{name}:{value}")
    else:
        df["speaker_id"] = df["source_speaker_id"]
    df["gender"] = gender
    df["origin_dataset"] = name
    df["source"] = df["source"].map(lambda value: _collapse_ws(value) or name)
    df["duration_s"] = df["duration_s"].map(_to_float_or_none)

    missing_audio = [p for p in df["audio"].head(20) if not Path(p).is_file()]
    if missing_audio:
        print(f"[build] {name}: WARNING first audio paths missing:", flush=True)
        for p in missing_audio[:5]:
            print(f"   {p}", flush=True)

    ds = Dataset.from_pandas(df[NORMALIZED_COLUMNS], preserve_index=False)
    ds = ds.cast_column("audio", Audio())
    dd = DatasetDict({split: ds})
    print(f"[build] {name}/{split}: normalized {len(ds)} rows", flush=True)
    return dd


def _normalized_source_dataset(name: str, split: str, prefix_speaker_ids: bool):
    src = _resolve_dataset_dir(name)
    if name in GREEK_PIPE_TTS_DATASETS:
        return _pipe_tts_dataset_dict(name, src, split, prefix_speaker_ids)
    if name in GREEK_HF_DATASETS:
        raw = _load_hf_dataset_dict(name, src)
        return _normalize_hf_dataset_dict(name, raw, prefix_speaker_ids)
    raise SystemExit(f"{name}: no Greek source loader configured")


def _write_speaker_names_json_from_dataset(dataset, speaker_names_json: str | None, label: str) -> None:
    if not speaker_names_json:
        return

    seed = int(os.environ.get("NAMED_SPEAKER_NAME_SEED", "1337"))
    rng = random.Random(seed)
    reserved = {
        os.environ.get("NAMED_FEMALE_SPEAKER_NAME", "Eleni"),
        os.environ.get("NAMED_MALE_SPEAKER_NAME", "Nikos"),
    }
    pools = {
        "female": _make_name_pool("female", reserved, rng),
        "male": _make_name_pool("male", reserved, rng),
    }
    counters = {"female": 0, "male": 0}
    speaker_genders = {}

    for split_name, split in dataset.items():
        if "speaker_id" not in split.column_names or "gender" not in split.column_names:
            raise SystemExit(f"{label}/{split_name}: expected speaker_id and gender columns")
        for row in split.select_columns(["speaker_id", "gender"]):
            speaker_id = str(row["speaker_id"])
            speaker_genders.setdefault(speaker_id, _normalize_gender_value(row["gender"], "male"))

    name_map = {}
    for speaker_id, gender in sorted(speaker_genders.items()):
        gender = "female" if gender == "female" else "male"
        index = counters[gender]
        name_map[speaker_id] = _pick_name(pools[gender], index, gender)
        counters[gender] += 1

    path = Path(speaker_names_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(name_map, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[build] {label}: wrote {len(name_map)} speaker names to {path}", flush=True)


def _split_order(split_names) -> list[str]:
    preferred = ["train", "eval", "validation", "test"]
    seen = set(split_names)
    ordered = [split for split in preferred if split in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def _build_multi_v2(out_dir: Path, split: str, speaker_names_json: str | None) -> None:
    from datasets import DatasetDict, concatenate_datasets

    by_split = defaultdict(list)
    for component in MULTI_V2_COMPONENTS:
        dd = _normalized_source_dataset(component, split, prefix_speaker_ids=True)
        for split_name, ds in dd.items():
            by_split[split_name].append(ds)

    processed = DatasetDict()
    for split_name in _split_order(by_split):
        processed[split_name] = concatenate_datasets(by_split[split_name])
        print(
            f"[build] multi_v2/{split_name}: concatenated "
            f"{len(by_split[split_name])} source split(s), {len(processed[split_name])} rows",
            flush=True,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_speaker_names_json_from_dataset(processed, speaker_names_json, "multi_v2")
    print(f"[build] multi_v2: writing to {out_dir}", flush=True)
    processed.save_to_disk(str(out_dir))
    print(f"[build] multi_v2: done. Verify with `python -c \"from datasets import load_from_disk; "
          f"print(load_from_disk('{out_dir}'))\"`", flush=True)


def _build_greek_source(name: str, out_dir: Path, split: str, speaker_names_json: str | None) -> None:
    dd = _normalized_source_dataset(name, split, prefix_speaker_ids=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    if speaker_names_json:
        _write_speaker_names_json_from_dataset(dd, speaker_names_json, name)
    print(f"[build] {name}: writing to {out_dir}", flush=True)
    dd.save_to_disk(str(out_dir))
    print(f"[build] {name}: done. Verify with `python -c \"from datasets import load_from_disk; "
          f"print(load_from_disk('{out_dir}'))\"`", flush=True)


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

    if name == "multi_v2":
        _build_multi_v2(out_dir, split, speaker_names_json)
        return

    if name in GREEK_HF_DATASETS or name in GREEK_PIPE_TTS_DATASETS:
        _build_greek_source(name, out_dir, split, speaker_names_json)
        return

    if name == "multi" and (src / "dataset_dict.json").is_file():
        _build_multi_from_saved_dataset(src, out_dir, multi_min_speaker_hours, speaker_names_json)
        return

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
    parser.add_argument(
        "--dataset",
        choices=[
            "female",
            "male",
            "multi",
            "commonVoice_greek_clean_with_speaker_ids",
            "cs10_greek_dataset",
            "greek_male_3.5h",
            "greek_female_tts",
            "greek_male_tts",
            "multi_v2",
            "both",
            "all",
            "greek_tts",
            "greek_sources",
        ],
        required=True,
    )
    parser.add_argument("--split-name", default="train",
                        help="Name of the split inside the DatasetDict (default: train).")
    parser.add_argument("--multi-min-speaker-hours", type=float,
                        default=float(os.environ.get("MULTI_MIN_SPEAKER_HOURS", "1.0")),
                        help="For --dataset multi/all, keep speakers with at least this many hours.")
    parser.add_argument("--speaker-names-json",
                        default=None,
                        help="For named multi builds, write speaker_id -> display-name JSON here.")
    args = parser.parse_args()

    if args.dataset == "both":
        targets = ["female", "male"]
    elif args.dataset == "all":
        targets = ["female", "male", "multi"]
    elif args.dataset == "greek_tts":
        targets = ["greek_female_tts", "greek_male_tts"]
    elif args.dataset == "greek_sources":
        targets = [*MULTI_V2_COMPONENTS, "multi_v2"]
    else:
        targets = [args.dataset]
    for name in targets:
        if name == "multi":
            speaker_names_json = args.speaker_names_json or os.environ.get("NAMED_MULTI_SPEAKER_NAMES_JSON")
        elif name == "multi_v2":
            speaker_names_json = args.speaker_names_json or os.environ.get("NAMED_MULTI_V2_SPEAKER_NAMES_JSON")
        else:
            speaker_names_json = None
        build(
            name,
            split=args.split_name,
            multi_min_speaker_hours=args.multi_min_speaker_hours,
            speaker_names_json=speaker_names_json,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
