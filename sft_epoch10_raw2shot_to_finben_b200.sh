#!/usr/bin/env bash
set -euo pipefail

# =========================
# Global settings
# =========================
REPO_ROOT="$(readlink -f "$(dirname "$0")")"

DATASET_PATH="${DATASET_PATH:-/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/benckmark/PV_benckmark/split_out/non_test}"
PY_SCRIPT="${PY_SCRIPT:-${REPO_ROOT}/sft_peft_ddp.py}"
MERGE_SCRIPT="${MERGE_SCRIPT:-${REPO_ROOT}/merge_lora.py}"
FINBEN_TASKS_PATH="${FINBEN_TASKS_PATH:-/home/lm2445/project_pi_sjf37/lm2445/finben/FinBen/tasks/pv_miner}"

MAX_LEN="${MAX_LEN:-8192}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-2e-4}"

TP="${TP:-2}"
NUM_GPUS="${NUM_GPUS:-${TP}}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-${TP}}"

GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-${MAX_LEN}}"
ENFORCE_EAGER="${ENFORCE_EAGER:-True}"

RAW_NUM_FEWSHOT="${RAW_NUM_FEWSHOT:-2}"
SFT_NUM_FEWSHOT="${SFT_NUM_FEWSHOT:-0}"
TASK="${TASK:-PvExtraction_full}"

PIPELINE_ROOT="${PIPELINE_ROOT:-$(readlink -f "${REPO_ROOT}/runs_pv_epoch${EPOCHS}_b200")}"

# =========================
# Models
# =========================
MODELS=(
  "meta-llama/Llama-3.3-70B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
)

guess_grad_accum () {
  case "$1" in
    meta-llama/Llama-3.3-70B-Instruct) echo 4 ;;
    meta-llama/Llama-3.1-8B-Instruct)  echo 8 ;;
    meta-llama/Llama-3.2-3B-Instruct)  echo 8 ;;
    Qwen/Qwen2.5-1.5B-Instruct)        echo 8 ;;
    *)                                 echo 8 ;;
  esac
}

model_slug () {
  local model_basename="$1"
  model_basename="${model_basename,,}"
  model_basename="${model_basename/llama-/llama}"
  model_basename="${model_basename//-/_}"
  printf '%s' "${model_basename}"
}

require_path () {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

# =========================
# Environment hygiene
# =========================
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "${PIPELINE_ROOT}"
PIPELINE_ROOT="$(readlink -f "${PIPELINE_ROOT}")"

require_path "${PY_SCRIPT}" "SFT script"
require_path "${MERGE_SCRIPT}" "merge script"
require_path "${DATASET_PATH}" "training dataset"
require_path "${FINBEN_TASKS_PATH}" "FinBen task path"

echo "============================================================"
echo "REPO_ROOT       : ${REPO_ROOT}"
echo "DATASET_PATH    : ${DATASET_PATH}"
echo "PIPELINE_ROOT   : ${PIPELINE_ROOT}"
echo "FINBEN_TASKS    : ${FINBEN_TASKS_PATH}"
echo "EPOCHS          : ${EPOCHS}"
echo "TP/GPUS         : TP=${TP} NUM_GPUS=${NUM_GPUS}"
echo "RAW FEWSHOT     : ${RAW_NUM_FEWSHOT}"
echo "SFT FEWSHOT     : ${SFT_NUM_FEWSHOT}"
echo "============================================================"

# =========================
# Main loop
# =========================
for MODEL in "${MODELS[@]}"; do
  if [[ "${MODEL}" == *"AWQ"* ]]; then
    echo "Skip ${MODEL} (AWQ not suitable for QLoRA training)"
    continue
  fi

  MODEL_TAG="$(basename "${MODEL}")"
  MODEL_SLUG="$(model_slug "${MODEL_TAG}")"
  RUN_TAG="sft_${EPOCHS}ep"

  MODEL_DIR="${PIPELINE_ROOT}/${MODEL_SLUG}/${RUN_TAG}"
  ADAPTER_DIR="${MODEL_DIR}/sft_adapter"
  MERGED_DIR="${MODEL_DIR}/merged"
  LOG_DIR="${MODEL_DIR}/logs"
  RAW_EVAL_DIR="${MODEL_DIR}/raw_2shot_lm_eval_results"
  SFT_EVAL_DIR="${MODEL_DIR}/sft_lm_eval_results"

  mkdir -p "${ADAPTER_DIR}" "${MERGED_DIR}" "${LOG_DIR}" "${RAW_EVAL_DIR}" "${SFT_EVAL_DIR}"

  MODEL_DIR="$(readlink -f "${MODEL_DIR}")"
  ADAPTER_DIR="$(readlink -f "${ADAPTER_DIR}")"
  MERGED_DIR="$(readlink -f "${MERGED_DIR}")"
  LOG_DIR="$(readlink -f "${LOG_DIR}")"
  RAW_EVAL_DIR="$(readlink -f "${RAW_EVAL_DIR}")"
  SFT_EVAL_DIR="$(readlink -f "${SFT_EVAL_DIR}")"

  GA="$(guess_grad_accum "${MODEL}")"

  echo "============================================================"
  echo "MODEL        : ${MODEL}"
  echo "MODEL_DIR    : ${MODEL_DIR}"
  echo "ADAPTER_DIR  : ${ADAPTER_DIR}"
  echo "MERGED_DIR   : ${MERGED_DIR}"
  echo "RAW_EVAL_DIR : ${RAW_EVAL_DIR}"
  echo "SFT_EVAL_DIR : ${SFT_EVAL_DIR}"
  echo "TP=${TP} | GA=${GA} | epochs=${EPOCHS} | lr=${LR}"
  echo "============================================================"

  # --------------------------------------------------
  # 1) Raw model 2-shot eval
  # --------------------------------------------------
  rm -rf "${RAW_EVAL_DIR}/${TASK}"
  lm_eval --model vllm \
    --model_args "pretrained=${MODEL},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},gpu_memory_utilization=${GPU_MEM_UTIL},max_model_len=${MAX_MODEL_LEN},enforce_eager=${ENFORCE_EAGER}" \
    --tasks "${TASK}" \
    --num_fewshot "${RAW_NUM_FEWSHOT}" \
    --batch_size auto \
    --output_path "${RAW_EVAL_DIR}/${TASK}" \
    --log_samples \
    --apply_chat_template \
    --include_path "${FINBEN_TASKS_PATH}" \
    2>&1 | tee "${LOG_DIR}/eval_raw_${RAW_NUM_FEWSHOT}shot_${TASK}.log"

  # --------------------------------------------------
  # 2) SFT (QLoRA, DDP)
  # --------------------------------------------------
  torchrun --nproc_per_node="${TP}" "${PY_SCRIPT}" \
    --dataset_path "${DATASET_PATH}" \
    --model_name "${MODEL}" \
    --output_dir "${ADAPTER_DIR}" \
    --use_qlora --bf16 \
    --max_length "${MAX_LEN}" \
    --batch_size "${BATCH_SIZE}" \
    --grad_accum "${GA}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    2>&1 | tee "${LOG_DIR}/sft.log"

  # --------------------------------------------------
  # 3) Merge LoRA -> full model
  # --------------------------------------------------
  ADAPTER_PATH="${ADAPTER_DIR}/lora_adapter"
  if [[ ! -f "${ADAPTER_PATH}/adapter_config.json" ]]; then
    ADAPTER_PATH="${ADAPTER_DIR}"
  fi
  require_path "${ADAPTER_PATH}" "adapter output"

  python "${MERGE_SCRIPT}" \
    --base "${MODEL}" \
    --adapter "${ADAPTER_PATH}" \
    --out "${MERGED_DIR}" \
    --dtype bf16 \
    2>&1 | tee "${LOG_DIR}/merge.log"

  # --------------------------------------------------
  # 4) SFT model eval
  # --------------------------------------------------
  rm -rf "${SFT_EVAL_DIR}/${TASK}"
  lm_eval --model vllm \
    --model_args "pretrained=${MERGED_DIR},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},gpu_memory_utilization=${GPU_MEM_UTIL},max_model_len=${MAX_MODEL_LEN},enforce_eager=${ENFORCE_EAGER}" \
    --tasks "${TASK}" \
    --num_fewshot "${SFT_NUM_FEWSHOT}" \
    --batch_size auto \
    --output_path "${SFT_EVAL_DIR}/${TASK}" \
    --log_samples \
    --apply_chat_template \
    --include_path "${FINBEN_TASKS_PATH}" \
    2>&1 | tee "${LOG_DIR}/eval_sft_${SFT_NUM_FEWSHOT}shot_${TASK}.log"

  echo "DONE: ${MODEL_TAG}"
done

echo
echo "All epoch-${EPOCHS} SFT runs finished. Outputs under:"
echo "  ${PIPELINE_ROOT}"
