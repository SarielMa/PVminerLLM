#!/usr/bin/env bash
set -euo pipefail

# =========================================================================
# Testing-only script (no training, no merging).
#
# For each of the 4 models trained under runs_pv_epoch10_b200:
#   - eval the SFT (merged) model   : 0-shot
#   - eval the corresponding RAW base model : 0-shot, 1-shot, 2-shot
#
# Every eval is repeated 3 times with temperature=0.7 (sampling on),
# using a different seed per repetition so the runs differ.
#
# Each (model x variant x shot x run) writes to its own absolute,
# split output directory.
# =========================================================================

# =========================
# Global settings
# =========================
REPO_ROOT="$(readlink -f "$(dirname "$0")")"

FINBEN_TASKS_PATH="${FINBEN_TASKS_PATH:-/home/lm2445/project_pi_sjf37/lm2445/finben/FinBen/tasks/pv_miner}"

# Base context length (0-shot). Few-shot evals scale this up per shot count,
# since each exemplar adds roughly a full document to the prompt.
BASE_LEN="${BASE_LEN:-8192}"

TP="${TP:-2}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-${TP}}"

GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
ENFORCE_EAGER="${ENFORCE_EAGER:-True}"

# max_model_len per few-shot count:
#   0-shot -> BASE_LEN, 1-shot -> BASE_LEN*2, 2-shot -> BASE_LEN*4
shot_maxlen () {
  local shot="$1"
  case "${shot}" in
    0) echo "${BASE_LEN}" ;;
    1) echo "$(( BASE_LEN * 2 ))" ;;
    2) echo "$(( BASE_LEN * 4 ))" ;;
    *) echo "$(( BASE_LEN * (1 << shot) ))" ;;
  esac
}

EPOCHS="${EPOCHS:-10}"
RUN_TAG="sft_${EPOCHS}ep"
TASK="${TASK:-PvExtraction_full}"

# Testing config
TEMPERATURE="${TEMPERATURE:-0.7}"
NUM_RUNS="${NUM_RUNS:-3}"
RAW_SHOTS=(0 1 2)
SFT_SHOT="${SFT_SHOT:-0}"

PIPELINE_ROOT="${PIPELINE_ROOT:-$(readlink -f "${REPO_ROOT}/runs_pv_epoch${EPOCHS}_b200")}"

# =========================
# Models (raw base models -> slug must match the trained run folders)
# =========================
MODELS=(
  "meta-llama/Llama-3.3-70B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
)

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

require_path "${PIPELINE_ROOT}" "pipeline root"
require_path "${FINBEN_TASKS_PATH}" "FinBen task path"

echo "============================================================"
echo "REPO_ROOT     : ${REPO_ROOT}"
echo "PIPELINE_ROOT : ${PIPELINE_ROOT}"
echo "FINBEN_TASKS  : ${FINBEN_TASKS_PATH}"
echo "TASK          : ${TASK}"
echo "BASE_LEN      : ${BASE_LEN} (0/1/2-shot -> ${BASE_LEN}/$((BASE_LEN*2))/$((BASE_LEN*4)))"
echo "TEMPERATURE   : ${TEMPERATURE}"
echo "NUM_RUNS      : ${NUM_RUNS}"
echo "RAW SHOTS     : ${RAW_SHOTS[*]}"
echo "SFT SHOT      : ${SFT_SHOT}"
echo "TP/GPUS       : TP=${TP}"
echo "============================================================"

# Sampling kwargs for stochastic decoding at the requested temperature.
GEN_KWARGS="temperature=${TEMPERATURE},do_sample=True"

# =========================
# Helper: run one lm_eval
#   $1 pretrained path/name
#   $2 num_fewshot
#   $3 absolute output dir
#   $4 absolute log file
#   $5 seed
# =========================
run_eval () {
  local pretrained="$1"
  local fewshot="$2"
  local out_dir="$3"
  local log_file="$4"
  local seed="$5"

  mkdir -p "${out_dir}" "$(dirname "${log_file}")"
  out_dir="$(readlink -f "${out_dir}")"

  local max_model_len
  max_model_len="$(shot_maxlen "${fewshot}")"

  rm -rf "${out_dir}/${TASK}"

  echo "    max_model_len=${max_model_len} (fewshot=${fewshot})"

  lm_eval --model vllm \
    --model_args "pretrained=${pretrained},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},gpu_memory_utilization=${GPU_MEM_UTIL},max_model_len=${max_model_len},enforce_eager=${ENFORCE_EAGER}" \
    --tasks "${TASK}" \
    --num_fewshot "${fewshot}" \
    --batch_size auto \
    --gen_kwargs "${GEN_KWARGS}" \
    --seed "${seed}" \
    --output_path "${out_dir}/${TASK}" \
    --log_samples \
    --apply_chat_template \
    --include_path "${FINBEN_TASKS_PATH}" \
    2>&1 | tee "${log_file}"
}

# =========================
# Main loop
# =========================
for MODEL in "${MODELS[@]}"; do
  MODEL_TAG="$(basename "${MODEL}")"
  MODEL_SLUG="$(model_slug "${MODEL_TAG}")"

  MODEL_DIR="$(readlink -f "${PIPELINE_ROOT}/${MODEL_SLUG}/${RUN_TAG}")"
  MERGED_DIR="${MODEL_DIR}/merged"
  TEST_ROOT="${MODEL_DIR}/test_t${TEMPERATURE}"
  LOG_DIR="${TEST_ROOT}/logs"

  require_path "${MERGED_DIR}" "merged SFT model (${MODEL_SLUG})"

  echo "============================================================"
  echo "MODEL      : ${MODEL}"
  echo "MERGED_DIR : ${MERGED_DIR}"
  echo "TEST_ROOT  : ${TEST_ROOT}"
  echo "============================================================"

  for ((R=1; R<=NUM_RUNS; R++)); do
    SEED=$((1000 + R))

    # ----- SFT (merged) model -----
    SFT_OUT="${TEST_ROOT}/sft/${SFT_SHOT}shot/run${R}"
    SFT_LOG="${LOG_DIR}/eval_sft_${SFT_SHOT}shot_run${R}.log"
    echo ">>> [${MODEL_SLUG}] SFT  ${SFT_SHOT}-shot  run ${R}  seed ${SEED}"
    run_eval "${MERGED_DIR}" "${SFT_SHOT}" "${SFT_OUT}" "${SFT_LOG}" "${SEED}"

    # ----- RAW base model, 0/1/2-shot -----
    for SHOT in "${RAW_SHOTS[@]}"; do
      RAW_OUT="${TEST_ROOT}/raw/${SHOT}shot/run${R}"
      RAW_LOG="${LOG_DIR}/eval_raw_${SHOT}shot_run${R}.log"
      echo ">>> [${MODEL_SLUG}] RAW  ${SHOT}-shot  run ${R}  seed ${SEED}"
      run_eval "${MODEL}" "${SHOT}" "${RAW_OUT}" "${RAW_LOG}" "${SEED}"
    done
  done

  echo "DONE: ${MODEL_TAG}"
done

echo
echo "All testing runs finished. Outputs under:"
for MODEL in "${MODELS[@]}"; do
  MODEL_SLUG="$(model_slug "$(basename "${MODEL}")")"
  echo "  ${PIPELINE_ROOT}/${MODEL_SLUG}/${RUN_TAG}/test_t${TEMPERATURE}"
done
