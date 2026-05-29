"""Build a CommonVoice speaker_id -> gender JSON.

Use this when one CommonVoice Greek dataset has gender labels, and the matching
training dataset has audio plus speaker IDs. The script matches rows by stable
text/file columns, aggregates gender counts per speaker, keeps row-level gender
labels for mixed speaker IDs, and writes a JSON consumed by 03_build_hf_dataset.py.

Run on a login node after activating the Leonardo conda env:

    source leonardo/env.sh && activate_conda_env
    python leonardo/login/06_build_commonvoice_gender_map.py
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_KEY_COLUMNS = (
    "file_name",
    "transcription_normalised",
    "transcription_normalized",
    "transcription",
    "text",
)


def _load_dataset_dict(path: Path):
    from datasets import DatasetDict, load_dataset, load_from_disk

    if (path / "dataset_dict.json").is_file():
        data = load_from_disk(str(path))
    else:
        data_dir = path / "data"
        parquet_files = sorted(data_dir.glob("*.parquet")) if data_dir.is_dir() else []
        if parquet_files:
            data_files = defaultdict(list)
            for parquet_file in parquet_files:
                split_name = parquet_file.name.split("-", 1)[0]
                data_files[split_name].append(str(parquet_file))
            data = load_dataset("parquet", data_files=dict(data_files))
        else:
            data = load_dataset(str(path))

    if not isinstance(data, DatasetDict):
        data = DatasetDict({"train": data})
    return data


def _normalize_gender(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"female", "f", "woman", "women"} or text.startswith("female"):
        return "female"
    if text in {"male", "m", "man", "men"} or text.startswith("male"):
        return "male"
    return ""


def _available_key_columns(gender_columns: list[str], speaker_columns: list[str], requested: str | None) -> list[str]:
    if requested:
        columns = [item.strip() for item in requested.split(",") if item.strip()]
    else:
        columns = [column for column in DEFAULT_KEY_COLUMNS if column in gender_columns and column in speaker_columns]
    if not columns:
        raise SystemExit(
            "could not infer matching columns; pass --key-columns, for example "
            "--key-columns file_name,transcription_normalised"
        )
    missing_gender = [column for column in columns if column not in gender_columns]
    missing_speaker = [column for column in columns if column not in speaker_columns]
    if missing_gender or missing_speaker:
        raise SystemExit(
            f"missing key columns: gender_dataset={missing_gender}, speaker_dataset={missing_speaker}"
        )
    return columns


def _row_key(row: dict, key_columns: list[str]) -> str:
    return "\t".join(str(row.get(column, "") or "").strip() for column in key_columns)


def _builder_key_columns(key_columns: list[str]) -> list[str]:
    return ["text" if column in {"transcription_normalised", "transcription_normalized"} else column
            for column in key_columns]


def _builder_row_key(row: dict, key_columns: list[str]) -> str:
    values = []
    for column in key_columns:
        if column in {"transcription_normalised", "transcription_normalized"}:
            values.append(str(row.get(column, "") or "").strip())
        else:
            values.append(str(row.get(column, "") or "").strip())
    return "\t".join(values)


def _iter_rows(dataset_dict):
    for split_name, split in dataset_dict.items():
        for row in split:
            yield split_name, row


def build_map(
    gender_dataset_path: Path,
    speaker_dataset_path: Path,
    output_json: Path,
    summary_txt: Path,
    key_columns_arg: str | None,
    gender_column: str,
    speaker_id_column: str,
) -> None:
    gender_dataset = _load_dataset_dict(gender_dataset_path)
    speaker_dataset = _load_dataset_dict(speaker_dataset_path)

    first_gender_split = gender_dataset[next(iter(gender_dataset))]
    first_speaker_split = speaker_dataset[next(iter(speaker_dataset))]
    gender_columns = list(first_gender_split.column_names)
    speaker_columns = list(first_speaker_split.column_names)

    if gender_column not in gender_columns:
        raise SystemExit(
            f"{gender_dataset_path} has no {gender_column!r} column. Columns: {gender_columns}"
        )
    if speaker_id_column not in speaker_columns:
        raise SystemExit(
            f"{speaker_dataset_path} has no {speaker_id_column!r} column. Columns: {speaker_columns}"
        )

    key_columns = _available_key_columns(gender_columns, speaker_columns, key_columns_arg)

    gender_by_key: dict[str, str] = {}
    row_genders: dict[str, str] = {}
    duplicate_keys = 0
    skipped_gender_rows = 0
    for _, row in _iter_rows(gender_dataset):
        gender = _normalize_gender(row.get(gender_column))
        if not gender:
            skipped_gender_rows += 1
            continue
        key = _row_key(row, key_columns)
        if key in gender_by_key and gender_by_key[key] != gender:
            duplicate_keys += 1
            continue
        gender_by_key[key] = gender
        row_genders[_builder_row_key(row, key_columns)] = gender

    speaker_counts: dict[str, Counter] = defaultdict(Counter)
    matched_rows = 0
    unmatched_rows = 0
    for _, row in _iter_rows(speaker_dataset):
        key = _row_key(row, key_columns)
        gender = gender_by_key.get(key)
        if not gender:
            unmatched_rows += 1
            continue
        speaker_id = str(row.get(speaker_id_column))
        speaker_counts[speaker_id][gender] += 1
        matched_rows += 1

    if not speaker_counts:
        raise SystemExit("no speaker genders were matched; check --key-columns and source datasets")

    speaker_genders = {}
    speaker_gender_counts = {}
    conflicts = {}
    for speaker_id, counts in sorted(speaker_counts.items()):
        speaker_gender_counts[speaker_id] = dict(counts)
        gender, _ = counts.most_common(1)[0]
        speaker_genders[speaker_id] = gender
        if len(counts) > 1:
            conflicts[speaker_id] = dict(counts)

    payload = {
        "source_gender_dataset": str(gender_dataset_path),
        "source_speaker_dataset": str(speaker_dataset_path),
        "key_columns": key_columns,
        "row_gender_key_columns": _builder_key_columns(key_columns),
        "gender_column": gender_column,
        "speaker_id_column": speaker_id_column,
        "matched_rows": matched_rows,
        "unmatched_rows": unmatched_rows,
        "skipped_gender_rows": skipped_gender_rows,
        "duplicate_conflicting_keys": duplicate_keys,
        "num_speakers": len(speaker_genders),
        "num_conflicting_speakers": len(conflicts),
        "speaker_genders": speaker_genders,
        "row_genders": row_genders,
        "speaker_gender_counts": speaker_gender_counts,
        "conflicting_speakers": conflicts,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    summary_txt.parent.mkdir(parents=True, exist_ok=True)
    with summary_txt.open("w", encoding="utf-8") as f:
        f.write(f"speaker gender map: {output_json}\n")
        f.write(f"gender dataset: {gender_dataset_path}\n")
        f.write(f"speaker dataset: {speaker_dataset_path}\n")
        f.write(f"key columns: {', '.join(key_columns)}\n")
        f.write(f"matched rows: {matched_rows}\n")
        f.write(f"unmatched rows: {unmatched_rows}\n")
        f.write(f"speakers: {len(speaker_genders)}\n")
        f.write(f"conflicting speakers: {len(conflicts)}\n")
        if conflicts:
            f.write("\nconflicts:\n")
            for speaker_id, counts in sorted(conflicts.items()):
                f.write(f"{speaker_id}: {counts}\n")

    print(f"[cv-genders] wrote {output_json}")
    print(f"[cv-genders] wrote {summary_txt}")
    print(f"[cv-genders] matched_rows={matched_rows} unmatched_rows={unmatched_rows}")
    print(f"[cv-genders] speakers={len(speaker_genders)} conflicts={len(conflicts)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    greek_root = Path(os.environ.get("GREEK_DATA_ROOT", "/leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/greekData"))
    parser.add_argument("--gender-dataset", type=Path, default=greek_root / "commonVoice_greek_clean_genders")
    parser.add_argument(
        "--speaker-dataset",
        type=Path,
        default=Path(os.environ.get("COMMONVOICE_GREEK_DIR", greek_root / "commonVoice_greek_clean_with_speaker_ids")),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            os.environ.get(
                "COMMONVOICE_GREEK_GENDER_JSON",
                "leonardo/config/commonvoice_greek_speaker_genders.json",
            )
        ),
    )
    parser.add_argument("--summary-txt", type=Path, default=None)
    parser.add_argument("--key-columns", default=None,
                        help="Comma-separated row matching columns. Default: common stable columns.")
    parser.add_argument("--gender-column", default="gender")
    parser.add_argument("--speaker-id-column", default="speaker_id")
    args = parser.parse_args()

    summary_txt = args.summary_txt or args.output_json.with_suffix(".txt")
    build_map(
        args.gender_dataset,
        args.speaker_dataset,
        args.output_json,
        summary_txt,
        args.key_columns,
        args.gender_column,
        args.speaker_id_column,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
