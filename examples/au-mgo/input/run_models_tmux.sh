#!/usr/bin/env bash
set -euo pipefail

# Launch Au-MgO dpa/les/sog training in independent tmux windows.
# Supports model x nlayers grid like the NaCl launcher.

SESSION_NAME="au_mgo_models"
TMUX_CMD="tmux -2"
RUN_ID=$(date +%Y%m%d_%H%M%S)
MARKER_DIR="/tmp/${SESSION_NAME}_${RUN_ID}_done"
TASK_MARKERS=()
TASK_COUNT=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/zyjin/anaconda3/envs/dp_devel/bin/python}"
DP_BIN="${DP_BIN:-/home/zyjin/anaconda3/envs/dp_devel/bin/dp_dev}"
DP_CMD="${DP_CMD:-${DP_BIN} --pt train}"
CONDA_SH="${CONDA_SH:-/home/zyjin/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-dp_devel}"
PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-/data/zyjin/dp_pt/dp_devel/deepmd-kit-devel}"

MODELS=("dpa" "les" "sog")
NLAYERS=("1" "3" "6")
NUMB_STEPS="${NUMB_STEPS:-500000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-2}"
GPU_DPA="${GPU_DPA:-5}"
GPU_LES="${GPU_LES:-6}"
GPU_SOG="${GPU_SOG:-7}"
GPU_ALL="${GPU_ALL:-}"
RUN_MODE="${RUN_MODE:-parallel}"
FINUFFT_CACHE="${FINUFFT_CACHE:-on}"
FINUFFT_CACHE_SIZE="${FINUFFT_CACHE_SIZE:-16}"

usage() {
  cat <<'EOF'
Usage: run_models_tmux.sh [options]

Options:
  --session-name NAME   tmux session name (default: au_mgo_models)
  --models "..."       Models list, e.g. "dpa les sog" or "sog"
  --nlayers-list "..." Nlayers list, e.g. "1 3 6"
  --numb-steps N        Override training.numb_steps in generated inputs
  --batch-size N        Override training/validation batch_size in generated inputs
  --save-freq N         Override training.save_freq (0 means only final-step checkpoint)
  --max-ckpt-keep N     Override training.max_ckpt_keep (default: 2)
  --gpu-dpa N           GPU id for dpa (default: 5)
  --gpu-les N           GPU id for les (default: 6)
  --gpu-sog N           GPU id for sog (default: 7)
  --gpu-all N           Use one GPU id for all models (fair compare)
  --run-mode MODE       parallel|serial (default: parallel)
  --serial              Same as --run-mode serial
  --finufft-cache MODE  cufinufft plan cache mode: on|off (default: on)
  --finufft-cache-size N cufinufft simple API cache size (default: 16)
  --help                Show help

Environment overrides:
  PYTHON_BIN, DP_BIN, DP_CMD, CONDA_SH, CONDA_ENV, PYTHONPATH_ROOT,
  FINUFFT_CACHE, FINUFFT_CACHE_SIZE, BATCH_SIZE, SAVE_FREQ, MAX_CKPT_KEEP,
  GPU_ALL, RUN_MODE

Notes:
  - The script runs in au-mgo/input so relative data paths in JSON work.
  - Required files:
    input_torch_au_mgo_split.json
    input_torch_au_mgo_les_split.json
    input_torch_au_mgo_sog_split.json
  - Generated files are written to au-mgo/nlayers_<N>/input and au-mgo/nlayers_<N>/output.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-name)
      SESSION_NAME="$2"
      shift 2
      ;;
    --models)
      # shellcheck disable=SC2206
      MODELS=($2)
      shift 2
      ;;
    --nlayers-list)
      # shellcheck disable=SC2206
      NLAYERS=($2)
      shift 2
      ;;
    --numb-steps)
      NUMB_STEPS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --save-freq)
      SAVE_FREQ="$2"
      shift 2
      ;;
    --max-ckpt-keep)
      MAX_CKPT_KEEP="$2"
      shift 2
      ;;
    --gpu-dpa)
      GPU_DPA="$2"
      shift 2
      ;;
    --gpu-les)
      GPU_LES="$2"
      shift 2
      ;;
    --gpu-sog)
      GPU_SOG="$2"
      shift 2
      ;;
    --gpu-all)
      GPU_ALL="$2"
      shift 2
      ;;
    --run-mode)
      RUN_MODE="$2"
      shift 2
      ;;
    --serial)
      RUN_MODE="serial"
      shift 1
      ;;
    --finufft-cache)
      FINUFFT_CACHE="$2"
      shift 2
      ;;
    --finufft-cache-size)
      FINUFFT_CACHE_SIZE="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

case "${FINUFFT_CACHE}" in
  on)
    CUFINUFFT_CACHE_FLAG=1
    ;;
  off)
    CUFINUFFT_CACHE_FLAG=0
    ;;
  *)
    echo "Invalid --finufft-cache: ${FINUFFT_CACHE}. Use on|off." >&2
    exit 1
    ;;
esac

if ! [[ "${FINUFFT_CACHE_SIZE}" =~ ^[0-9]+$ ]]; then
  echo "Invalid --finufft-cache-size: ${FINUFFT_CACHE_SIZE}. Use a non-negative integer." >&2
  exit 1
fi

if [[ -n "${BATCH_SIZE}" ]] && ! [[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid --batch-size: ${BATCH_SIZE}. Use a positive integer." >&2
  exit 1
fi

if ! [[ "${SAVE_FREQ}" =~ ^[0-9]+$ ]]; then
  echo "Invalid --save-freq: ${SAVE_FREQ}. Use a non-negative integer." >&2
  exit 1
fi

if ! [[ "${MAX_CKPT_KEEP}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid --max-ckpt-keep: ${MAX_CKPT_KEEP}. Use a positive integer." >&2
  exit 1
fi

if [[ -n "${GPU_ALL}" ]] && ! [[ "${GPU_ALL}" =~ ^[0-9]+$ ]]; then
  echo "Invalid --gpu-all: ${GPU_ALL}. Use a non-negative integer." >&2
  exit 1
fi

case "${RUN_MODE}" in
  parallel|serial)
    ;;
  *)
    echo "Invalid --run-mode: ${RUN_MODE}. Use parallel|serial." >&2
    exit 1
    ;;
esac

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed." >&2
  exit 1
fi

if [[ ! -f "${DP_BIN}" ]]; then
  echo "dp executable not found: ${DP_BIN}" >&2
  exit 1
fi

mkdir -p "${MARKER_DIR}"

declare -A INPUT_BY_MODEL=(
  [dpa]="input_torch_au_mgo_split.json"
  [les]="input_torch_au_mgo_les_split.json"
  [sog]="input_torch_au_mgo_sog_split.json"
)

declare -A GPU_BY_MODEL=(
  [dpa]="${GPU_DPA}"
  [les]="${GPU_LES}"
  [sog]="${GPU_SOG}"
)

if [[ -n "${GPU_ALL}" ]]; then
  GPU_BY_MODEL[dpa]="${GPU_ALL}"
  GPU_BY_MODEL[les]="${GPU_ALL}"
  GPU_BY_MODEL[sog]="${GPU_ALL}"
fi

$TMUX_CMD has-session -t "$SESSION_NAME" 2>/dev/null || $TMUX_CMD new-session -d -s "$SESSION_NAME"

LAST_MARKER_FILE=""

for model in "${MODELS[@]}"; do
  if [[ -z "${INPUT_BY_MODEL[$model]:-}" ]]; then
    echo "Skip unknown model: ${model}" >&2
    continue
  fi

  template_json="${SCRIPT_DIR}/${INPUT_BY_MODEL[$model]}"
  gpu_id="${GPU_BY_MODEL[$model]}"

  if [[ ! -f "${template_json}" ]]; then
    echo "Missing input file: ${template_json}" >&2
    continue
  fi

  for nl in "${NLAYERS[@]}"; do
    window_name="${model}-${nl}"
    nl_root_dir="${PROJECT_ROOT}/nlayers_${nl}"
    nl_input_dir="${nl_root_dir}/input"
    nl_output_dir="${nl_root_dir}/output"
    generated_json="${nl_input_dir}/input_torch_au_mgo_${model}_split_nlayers${nl}.json"

    if $TMUX_CMD list-windows -t "$SESSION_NAME" | grep -q "${window_name}"; then
      echo "Window ${window_name} already exists, skipping."
      continue
    fi

    if [[ "${RUN_MODE}" == "serial" && -n "${LAST_MARKER_FILE}" ]]; then
      echo "Serial mode: waiting previous task to finish (${LAST_MARKER_FILE})"
      while [[ ! -f "${LAST_MARKER_FILE}" ]]; do
        sleep 5
      done
    fi

    mkdir -p "${nl_output_dir}" "${nl_input_dir}"

    "$PYTHON_BIN" - "$template_json" "$generated_json" "$nl" "$NUMB_STEPS" "$model" "${BATCH_SIZE:-__KEEP__}" "${SAVE_FREQ:-0}" "${MAX_CKPT_KEEP}" <<'PY'
import json
import os
import sys
from pathlib import Path

template_json = sys.argv[1]
generated_json = sys.argv[2]
nlayers = int(sys.argv[3])
numb_steps = int(sys.argv[4])
model = sys.argv[5]
batch_size_arg = sys.argv[6]
save_freq_arg = sys.argv[7]
max_ckpt_keep_arg = sys.argv[8]

template_dir = Path(template_json).resolve().parent
generated_dir = Path(generated_json).resolve().parent

with open(template_json, "r", encoding="utf-8") as f:
    cfg = json.load(f)

cfg["model"]["descriptor"]["repflow"]["nlayers"] = nlayers
cfg["training"]["numb_steps"] = numb_steps
cfg["training"]["stat_file"] = f"../output/{model}_stat_nl{nlayers}.hdf5"
cfg["training"]["disp_file"] = f"../output/lcurve_{model}_nl{nlayers}.out"
cfg["training"]["save_ckpt"] = f"../output/{model}_model.ckpt"

save_freq = int(save_freq_arg)
if save_freq <= 0:
  save_freq = numb_steps
cfg["training"]["save_freq"] = save_freq
cfg["training"]["max_ckpt_keep"] = int(max_ckpt_keep_arg)

for split_name in ("training_data", "validation_data"):
    systems = cfg.get("training", {}).get(split_name, {}).get("systems", [])
    rewritten = []
    for s in systems:
        abs_path = (template_dir / s).resolve()
        rewritten.append(os.path.relpath(abs_path, generated_dir))
    if split_name in cfg.get("training", {}):
        cfg["training"][split_name]["systems"] = rewritten

if batch_size_arg != "__KEEP__":
    batch_size = int(batch_size_arg)
    if "training_data" in cfg.get("training", {}):
        cfg["training"]["training_data"]["batch_size"] = batch_size
    if "validation_data" in cfg.get("training", {}):
        cfg["training"]["validation_data"]["batch_size"] = batch_size

with open(generated_json, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
PY

    $TMUX_CMD new-window -t "$SESSION_NAME" -n "$window_name"
    marker_file="${MARKER_DIR}/${window_name}.done"
    TASK_MARKERS+=("$marker_file")
    TASK_COUNT=$((TASK_COUNT + 1))

    generated_json_name="$(basename "${generated_json}")"
    cmd="set -o pipefail && cd ${nl_input_dir} && source ${CONDA_SH} && conda activate ${CONDA_ENV} && export CUDA_VISIBLE_DEVICES=${gpu_id} && export PYTHONPATH=${PYTHONPATH_ROOT} && export CUFINUFFT_SIMPLE_PLAN_CACHE=${CUFINUFFT_CACHE_FLAG} && export CUFINUFFT_SIMPLE_PLAN_CACHE_SIZE=${FINUFFT_CACHE_SIZE} && ${DP_CMD} ${generated_json_name} 2>&1 | tee ${nl_output_dir}/train_${model}_nl${nl}.log; status=\$?; touch ${marker_file}; exit \$status"
    $TMUX_CMD send-keys -t "${SESSION_NAME}:${window_name}" "$cmd" C-m

    LAST_MARKER_FILE="${marker_file}"

    echo "Started model=${model} nlayers=${nl} on GPU=${gpu_id} using ${generated_json} (mode=${RUN_MODE}, finufft_cache=${FINUFFT_CACHE}, size=${FINUFFT_CACHE_SIZE}, save_freq=${SAVE_FREQ}, max_ckpt_keep=${MAX_CKPT_KEEP})"
  done
done

if [[ "$TASK_COUNT" -gt 0 ]]; then
  marker_list="${TASK_MARKERS[*]}"
  monitor_cmd="while true; do done_count=0; for marker in ${marker_list}; do [ -f \"\$marker\" ] && done_count=\$((done_count+1)); done; if [ \"\$done_count\" -ge ${TASK_COUNT} ]; then echo \"All submitted tasks finished. Killing session ${SESSION_NAME}.\"; tmux -2 kill-session -t ${SESSION_NAME}; break; fi; sleep 10; done"
  $TMUX_CMD run-shell -t "$SESSION_NAME" -b "bash -lc '$monitor_cmd'"
  echo "Auto-kill monitor started for ${TASK_COUNT} task(s)."
else
  echo "No new tasks were submitted; auto-kill monitor was not started."
fi

echo "Submitted all jobs (mode=${RUN_MODE})."
echo "Attach with: tmux attach -t ${SESSION_NAME}"
