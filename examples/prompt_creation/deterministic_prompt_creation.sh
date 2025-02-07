# python ./scripts/run_prompt_creation_deterministic.py \
#   --is_new_speaker_prompt \
#   --dataset_name "syllasgiorgos/greek_male_3.5h-text-tags" \
#   --output_dir "./tmp_dataset" \
#   --dataset_config_name "default" \
#   --push_to_hub \
#   --hub_dataset_id "greek_male_3.5h-descriptions-deterministic" \
#   --preprocessing_num_workers 2 

python ./scripts/run_prompt_creation_deterministic.py \
  --is_new_speaker_prompt \
  --dataset_name "syllasgiorgos/graham-green-end-of-relationship-text-tags" \
  --output_dir "./tmp_dataset1" \
  --dataset_config_name "default" \
  --push_to_hub \
  --hub_dataset_id "graham-green-end-of-relationship-descriptions-deterministic" \
  --preprocessing_num_workers 2 
  
  python ./scripts/run_prompt_creation_deterministic.py \
  --is_new_speaker_prompt \
  --dataset_name "syllasgiorgos/cs10_greek_dataset_gender-text-tags" \
  --output_dir "./tmp_dataset2" \
  --dataset_config_name "default" \
  --push_to_hub \
  --hub_dataset_id "cs10_greek_dataset_gender-descriptions-deterministic" \
  --preprocessing_num_workers 2 

  python ./scripts/run_prompt_creation_deterministic.py \
  --is_new_speaker_prompt \
  --dataset_name "syllasgiorgos/commonVoice_greek_clean_speakers_genders-text-tags" \
  --output_dir "./tmp_dataset3" \
  --dataset_config_name "default" \
  --push_to_hub \
  --hub_dataset_id "commonVoice_greek_clean_speakers_genders-descriptions-deterministic" \
  --preprocessing_num_workers 2 