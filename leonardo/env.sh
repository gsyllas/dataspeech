#!/usr/bin/env bash
# Shared config for all dataspeech leonardo scripts.
# Sourced by login-node helpers and by SLURM jobs.
#
# Override any of these by exporting them before sourcing, or by editing
# this file. Per-job runtime overrides (e.g. DATASET=female sbatch ...)
# are honored via the `${VAR:-default}` pattern.

set -u

# ---- account / partition --------------------------------------------------
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-EUHPC_D29_081}"
export SLURM_PARTITION="${SLURM_PARTITION:-boost_usr_prod}"

# ---- repo / cache / data --------------------------------------------------
# REPO_ROOT is the absolute path to the cloned dataspeech repo on /leonardo_work.
# It must NOT be in $HOME (no quota). Default: parent dir of this file's parent.
if [ -z "${REPO_ROOT:-}" ]; then
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
export REPO_ROOT

# Conda env lives INSIDE the repo (per user requirement: no $HOME space).
export CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-$REPO_ROOT/.conda/dataspeech}"

# All HF / torch / penn caches live next to the repo on /leonardo_work.
# Some login-node shells already export HF_HOME / TORCH_HOME from old projects;
# ignore those inherited values so this pipeline does not fill the wrong quota.
# To intentionally move every cache, set CACHE_ROOT before sourcing env.sh.
export CACHE_ROOT="${CACHE_ROOT:-$REPO_ROOT/cache}"
_dataspeech_note_ignored_cache_var() {
  local name="$1"
  local value="${!name:-}"
  local expected="$2"
  if [ -n "$value" ] && [ "$value" != "$expected" ]; then
    echo "[env.sh] ignoring inherited $name=$value; using $expected" >&2
  fi
}
_dataspeech_note_ignored_cache_var HF_HOME "$CACHE_ROOT/hf"
_dataspeech_note_ignored_cache_var TORCH_HOME "$CACHE_ROOT/torch"
_dataspeech_note_ignored_cache_var PENN_CACHE "$CACHE_ROOT/penn"
_dataspeech_note_ignored_cache_var PIP_CACHE_DIR "$CACHE_ROOT/pip"
export HF_HOME="$CACHE_ROOT/hf"
export HF_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$CACHE_ROOT/torch"
export PENN_CACHE="$CACHE_ROOT/penn"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
unset -f _dataspeech_note_ignored_cache_var
# Keep conda's package + env metadata caches in the repo too (don't pollute $HOME).
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$REPO_ROOT/.conda/pkgs}"
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-$REPO_ROOT/.conda/envs}"

# Where the user's two source datasets live (CSV + wavs/ subfolder each).
export DATA_ROOT="${DATA_ROOT:-/leonardo_work/EUHPC_D29_081/gsyllas0/data/tts}"
export FEMALE_DIR="${FEMALE_DIR:-$DATA_ROOT/female_normalized}"
export MALE_DIR="${MALE_DIR:-$DATA_ROOT/male_normalized}"
export MULTI_SPEAKER_DIR="${MULTI_SPEAKER_DIR:-$DATA_ROOT/multi_speaker_combined}"
export GREEK_DATA_ROOT="${GREEK_DATA_ROOT:-$DATA_ROOT/greekData}"
export COMMONVOICE_GREEK_DIR="${COMMONVOICE_GREEK_DIR:-$GREEK_DATA_ROOT/commonVoice_greek_clean_with_speaker_ids}"
export CS10_GREEK_DIR="${CS10_GREEK_DIR:-$GREEK_DATA_ROOT/cs10_greek_dataset}"
export GREEK_MALE_35H_DIR="${GREEK_MALE_35H_DIR:-$GREEK_DATA_ROOT/greek_male_3.5h}"
export GREEK_FEMALE_TTS_DIR="${GREEK_FEMALE_TTS_DIR:-$GREEK_DATA_ROOT/greek_female_tts}"
export GREEK_MALE_TTS_DIR="${GREEK_MALE_TTS_DIR:-$GREEK_DATA_ROOT/greek_male_tts}"
export MULTI_V2_COMPONENTS="${MULTI_V2_COMPONENTS:-commonVoice_greek_clean_with_speaker_ids cs10_greek_dataset greek_male_3.5h greek_female_tts greek_male_tts}"
export COMMONVOICE_GREEK_GENDER_JSON="${COMMONVOICE_GREEK_GENDER_JSON:-$REPO_ROOT/leonardo/config/commonvoice_greek_speaker_genders.json}"
export METADATA_CSV_NAME="${METADATA_CSV_NAME:-metadata.csv}"
export WAVS_SUBDIR="${WAVS_SUBDIR:-wavs}"

# Per-dataset output root (one dir per dataset; all stages write here).
export OUT_ROOT="${OUT_ROOT:-$DATA_ROOT/dataspeech_out}"
export NAMED_OUT_ROOT="${NAMED_OUT_ROOT:-$DATA_ROOT/dataspeech_out_named}"
export NAMED_FEMALE_SPEAKER_NAME="${NAMED_FEMALE_SPEAKER_NAME:-Eleni}"
export NAMED_MALE_SPEAKER_NAME="${NAMED_MALE_SPEAKER_NAME:-Nikos}"
export NAMED_MULTI_SPEAKER_NAMES_JSON="${NAMED_MULTI_SPEAKER_NAMES_JSON:-$REPO_ROOT/leonardo/config/multi_speaker_names.json}"
export NAMED_MULTI_V2_SPEAKER_NAMES_JSON="${NAMED_MULTI_V2_SPEAKER_NAMES_JSON:-$REPO_ROOT/leonardo/config/multi_v2_speaker_names.json}"
export NAMED_SPEAKER_NAME_SEED="${NAMED_SPEAKER_NAME_SEED:-1337}"
export MULTI_MIN_SPEAKER_HOURS="${MULTI_MIN_SPEAKER_HOURS:-1.0}"

# ---- pipeline parameters --------------------------------------------------
# LLM for run_prompt_creation.py (A100 64GB -> fp16 fits Llama-3.1-8B-Instruct).
export LLM_MODEL_ID="${LLM_MODEL_ID:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
export LLM_TORCH_COMPILE="${LLM_TORCH_COMPILE:-1}"
export LLM_TRUST_REMOTE_CODE="${LLM_TRUST_REMOTE_CODE:-0}"
export LLM_USE_HF_TOKEN="${LLM_USE_HF_TOKEN:-0}"
# Brouhaha checkpoint (SNR + reverb).
export BROUHAHA_REPO="${BROUHAHA_REPO:-ylacombe/brouhaha-best}"
# Bin edges / text bins for metadata_to_text.py.
export BIN_EDGES_PATH="${BIN_EDGES_PATH:-$REPO_ROOT/examples/tags_to_annotations/v02_bin_edges.json}"
export TEXT_BINS_PATH="${TEXT_BINS_PATH:-$REPO_ROOT/examples/tags_to_annotations/v02_text_bins.json}"

# Worker counts (Leonardo boost node = 32 physical cores / 1x A100 64GB).
export CPU_NUM_WORKERS="${CPU_NUM_WORKERS:-8}"
export PREPROC_WORKERS="${PREPROC_WORKERS:-8}"
export LLM_EVAL_BATCH_SIZE="${LLM_EVAL_BATCH_SIZE:-64}"

# ---- helpers --------------------------------------------------------------
# Resolve a per-dataset directory from the DATASET name.
dataset_dir_for() {
  case "${1:-}" in
    female) echo "$FEMALE_DIR" ;;
    male)   echo "$MALE_DIR" ;;
    multi)  echo "$MULTI_SPEAKER_DIR" ;;
    commonVoice_greek_clean_with_speaker_ids) echo "$COMMONVOICE_GREEK_DIR" ;;
    cs10_greek_dataset) echo "$CS10_GREEK_DIR" ;;
    greek_male_3.5h) echo "$GREEK_MALE_35H_DIR" ;;
    greek_female_tts) echo "$GREEK_FEMALE_TTS_DIR" ;;
    greek_male_tts) echo "$GREEK_MALE_TTS_DIR" ;;
    multi_v2) echo "$GREEK_DATA_ROOT" ;;
    *) echo "[env.sh] unknown DATASET='${1:-}'" >&2; return 1 ;;
  esac
}

named_speaker_names_json_for() {
  case "${1:-}" in
    multi) echo "$NAMED_MULTI_SPEAKER_NAMES_JSON" ;;
    multi_v2) echo "$NAMED_MULTI_V2_SPEAKER_NAMES_JSON" ;;
    *) echo "[env.sh] no named speaker JSON configured for DATASET='${1:-}'" >&2; return 1 ;;
  esac
}

# Per-dataset output paths. Pass dataset name as $1.
out_dir_for()         { echo "$OUT_ROOT/$1"; }
hf_dataset_dir_for()  { echo "$OUT_ROOT/$1/01_hf_dataset"; }
tags_dir_for()        { echo "$OUT_ROOT/$1/02_tags"; }
text_tags_dir_for()   { echo "$OUT_ROOT/$1/03_text_tags"; }
prompts_det_dir_for() { echo "$OUT_ROOT/$1/04a_prompts_deterministic"; }
prompts_llm_dir_for() { echo "$OUT_ROOT/$1/04b_prompts_llm"; }

# Activate the in-repo conda env on Leonardo. Looks for conda in common spots.
activate_conda_env() {
  local conda_sh=""
  for c in \
    "$REPO_ROOT/.conda/miniforge/etc/profile.d/conda.sh" \
    "$HOME/miniforge3/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "/leonardo/prod/opt/tools/miniconda3/2024.06/none/etc/profile.d/conda.sh"; do
    if [ -f "$c" ]; then conda_sh="$c"; break; fi
  done
  if [ -z "$conda_sh" ]; then
    echo "[env.sh] could not find conda.sh; install miniforge first (see leonardo/login/00_setup_conda_env.sh)" >&2
    return 1
  fi
  # shellcheck disable=SC1090
  source "$conda_sh"
  conda activate "$CONDA_ENV_PREFIX"
}

# Force offline mode for compute nodes (no internet).
set_offline_mode() {
  export HF_DATASETS_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
}

set +u
