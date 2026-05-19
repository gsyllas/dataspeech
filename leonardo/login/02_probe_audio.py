"""Print sample-rate / channels / duration stats for the two datasets.

You answered "don't know, just probe" — run this on a login node after
activating the conda env. It samples up to N files from each dataset and
prints a histogram of sample rates + channel counts + duration spread.

Usage:
    source leonardo/env.sh && activate_conda_env
    python leonardo/login/02_probe_audio.py [--max-files 200]
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median


def probe_one(dataset_label: str, root: Path, csv_name: str, wavs_subdir: str, max_files: int) -> None:
    import soundfile as sf

    csv_path = root / csv_name
    wavs_dir = root / wavs_subdir
    print(f"\n=== {dataset_label} ===")
    print(f"  root:    {root}")
    print(f"  csv:     {csv_path} (exists={csv_path.is_file()})")
    print(f"  wavs/:   {wavs_dir} (exists={wavs_dir.is_dir()})")

    if not wavs_dir.is_dir():
        print(f"  [skip] {wavs_dir} not found")
        return

    files = sorted(p for p in wavs_dir.iterdir() if p.suffix.lower() in {".wav", ".flac", ".ogg", ".mp3"})
    print(f"  audio files in wavs/: {len(files)}")
    if not files:
        return

    sample = files if len(files) <= max_files else random.sample(files, max_files)
    srs: Counter = Counter()
    chs: Counter = Counter()
    durs: list[float] = []
    bad: list[str] = []
    for p in sample:
        try:
            info = sf.info(str(p))
            srs[info.samplerate] += 1
            chs[info.channels] += 1
            durs.append(info.duration)
        except Exception as e:  # noqa: BLE001
            bad.append(f"{p.name}: {e!r}")

    print(f"  probed {len(sample)} files ({len(bad)} unreadable)")
    print(f"  sample rates: {dict(srs)}")
    print(f"  channels:     {dict(chs)}")
    if durs:
        print(
            f"  duration s:   min={min(durs):.2f} median={median(durs):.2f} "
            f"mean={mean(durs):.2f} max={max(durs):.2f}"
        )
    if bad:
        print("  first unreadable:")
        for line in bad[:5]:
            print(f"    {line}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-files", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed)

    csv_name = os.environ.get("METADATA_CSV_NAME", "metadata.csv")
    wavs_subdir = os.environ.get("WAVS_SUBDIR", "wavs")

    for label, env_var in [("female", "FEMALE_DIR"), ("male", "MALE_DIR")]:
        root = os.environ.get(env_var)
        if not root:
            print(f"[probe] {env_var} not set; source leonardo/env.sh first", file=sys.stderr)
            return 1
        probe_one(label, Path(root), csv_name, wavs_subdir, args.max_files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
