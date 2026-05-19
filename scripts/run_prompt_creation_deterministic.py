import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from datasets import DatasetDict, load_dataset, load_from_disk
import shutil


def _is_save_to_disk_dir(path):
    # Only DatasetDict (not bare Dataset) is supported as a local path: the
    # downstream code iterates over named splits.
    return (
        isinstance(path, str)
        and os.path.isdir(path)
        and os.path.isfile(os.path.join(path, "dataset_dict.json"))
    )

logger = logging.getLogger(__name__)

@dataclass
class DataArguments:
    output_dir: str = field(metadata={"help": "Where to save the processed dataset"})
    dataset_name: str = field(default=None, metadata={"help": "Dataset name"})
    dataset_config_name: Optional[str] = field(default=None)
    dataset_split_name: Optional[str] = field(default=None)
    dataset_cache_dir: Optional[str] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    overwrite_cache: bool = field(default=False)
    preprocessing_num_workers: Optional[int] = field(default=None)
    push_to_hub: Optional[bool] = field(default=False)
    hub_dataset_id: Optional[str] = field(default=None)
    overwrite_output_dir: Optional[bool] = field(default=False)
    speaker_name: Optional[str] = field(default=None)
    is_single_speaker: Optional[bool] = field(default=False)
    is_new_speaker_prompt: Optional[bool] = field(default=False)
    speaker_id_column: Optional[str] = field(default=None)
    speaker_ids_to_name_json: Optional[str] = field(default=None)
    accent_column: Optional[str] = field(default=None)

def generate_description(sample, is_single_speaker=False, is_new_speaker_prompt=False, 
                        speaker_name=None, speaker_ids_to_name=None, speaker_id_column=None, 
                        accent_column=None):
    features = []
    
    if is_single_speaker and speaker_name:
        features.append(f"speaker: {speaker_name}")
    elif speaker_id_column and speaker_ids_to_name:
        speaker_id = str(sample.get(speaker_id_column))
        name = speaker_ids_to_name.get(speaker_id)
        if name:
            features.append(f"speaker: {name}")
    elif "gender" in sample:
        features.append(f"{sample['gender']}")

    core_features = ["reverberation", "sdr_noise" if "sdr_noise" in sample else "noise", 
                    "speech_monotony", "speaking_rate", "pitch"]
    
    for feature in core_features:
        if feature in sample:
            features.append(f"{sample[feature]}")
            
    if accent_column and accent_column in sample:
        features.append(f"{sample[accent_column]}")
        
    return ", ".join(features)

def main():
    from transformers import HfArgumentParser
    parser = HfArgumentParser(DataArguments)
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        data_args = parser.parse_args_into_dataclasses()[0]

    if data_args.overwrite_output_dir and os.path.exists(data_args.output_dir):
        shutil.rmtree(data_args.output_dir)

    if _is_save_to_disk_dir(data_args.dataset_name):
        raw_datasets = load_from_disk(data_args.dataset_name)
        if data_args.dataset_split_name:
            raw_datasets = DatasetDict({
                split: raw_datasets[split]
                for split in data_args.dataset_split_name.split("+")
            })
    elif data_args.dataset_split_name:
        raw_datasets = DatasetDict()
        for split in data_args.dataset_split_name.split("+"):
            raw_datasets[split] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=split,
                cache_dir=data_args.dataset_cache_dir,
                num_proc=data_args.preprocessing_num_workers,
            )
    else:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=data_args.dataset_cache_dir,
            num_proc=data_args.preprocessing_num_workers,
        )

    EXPECTED_COLUMNS = {"gender", "pitch", "noise", "reverberation", "speech_monotony", "speaking_rate"}
    if data_args.is_single_speaker:
        EXPECTED_COLUMNS = {"noise", "reverberation", "speech_monotony", "speaking_rate"}
        
    if data_args.is_new_speaker_prompt:
        EXPECTED_COLUMNS.remove("noise")
        EXPECTED_COLUMNS.add("sdr_noise")

    raw_datasets_features = set(raw_datasets[next(iter(raw_datasets))].features.keys())
    if not EXPECTED_COLUMNS.issubset(raw_datasets_features):
        missing_columns = EXPECTED_COLUMNS - raw_datasets_features
        raise ValueError(f"Missing columns {missing_columns}")

    speaker_ids_to_name = {}
    if data_args.speaker_ids_to_name_json:
        import json
        with open(data_args.speaker_ids_to_name_json, "r") as f:
            speaker_ids_to_name = json.load(f)

    def process_sample(sample):
        sample["text_description"] = generate_description(
            sample,
            is_single_speaker=data_args.is_single_speaker,
            is_new_speaker_prompt=data_args.is_new_speaker_prompt,
            speaker_name=data_args.speaker_name,
            speaker_ids_to_name=speaker_ids_to_name,
            speaker_id_column=data_args.speaker_id_column,
            accent_column=data_args.accent_column
        )
        return sample

    processed_datasets = raw_datasets.map(
        process_sample,
        num_proc=data_args.preprocessing_num_workers,
        desc="Generating descriptions"
    )

    processed_datasets.save_to_disk(data_args.output_dir)
    if data_args.push_to_hub:
        processed_datasets.push_to_hub(data_args.hub_dataset_id)

if __name__ == "__main__":
    main()