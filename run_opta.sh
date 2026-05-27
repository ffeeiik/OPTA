#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p evaluation_results

if [ -n "${CONDA_PREFIX:-}" ]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

EXISTING_NO_PROXY="${NO_PROXY:-${no_proxy:-}}"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1${EXISTING_NO_PROXY:+,${EXISTING_NO_PROXY}}"
export no_proxy="${NO_PROXY}"

OPTA_TRAJECTORIES="${OPTA_TRAJECTORIES:-10}"
OPTA_WORKERS="${OPTA_WORKERS:-8}"

MODEL_PATH="${MODEL_PATH:-/your/path/LLM_models/Qwen3-30B-A3B-Instruct-2507}"
MODEL_NAME="${MODEL_NAME:-Qwen3-30B-A3B-Instruct-2507}"
EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-/your/path/LLM_models/Qwen3-Embedding-8B}"
VLLM_PORT="${VLLM_PORT:-1840}"
SEARCH_PORT="${SEARCH_PORT:-18040}"

DATA_PATH="${DATA_PATH:-/your/path/data/bc+_166.parquet}"
NUM_SAMPLES="${NUM_SAMPLES:-all}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-64000}"

export AGENT_MODEL_NAME="${MODEL_NAME}"
export MODEL_NAME="${MODEL_NAME}"
export LOCAL_MODEL_PATH="${MODEL_PATH}"
export LOCAL_SEARCH_URL="http://localhost:${SEARCH_PORT}"
export OPTA_LLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"
export CORPUS_DATA_PATH="${CORPUS_DATA_PATH:-/your/path/data/corpus}"
export CORPUS_EMBEDDINGS_PATH="${CORPUS_EMBEDDINGS_PATH:-${CORPUS_DATA_PATH%/}/corpus_embeddings.pkl}"

export OPTA_MAX_TURN="${OPTA_MAX_TURN:-300}"
export MAX_MODEL_LEN

export OPTA_TRAJECTORIES
export OPTA_WORKERS
export OPTA_MIN_SUMMARY_ROUNDS="${OPTA_MIN_SUMMARY_ROUNDS:-1}"
export OPTA_SAMPLE_TEMPERATURE="${OPTA_SAMPLE_TEMPERATURE:-0.2}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TASK_ID="opta_${MODEL_NAME}_traj${OPTA_TRAJECTORIES}_workers${OPTA_WORKERS}_maxturn${OPTA_MAX_TURN}_${TIMESTAMP}"
OUTPUT_DIR="${OUTPUT_DIR:-./evaluation_results/${TASK_ID}}"
mkdir -p "${OUTPUT_DIR}"

SEARCH_PID=""
VLLM_PID=""
EVAL_PID=""

cleanup() {
  if [ -n "${EVAL_PID}" ]; then kill "${EVAL_PID}" 2>/dev/null || true; fi
  if [ -n "${VLLM_PID}" ]; then kill "${VLLM_PID}" 2>/dev/null || true; fi
  if [ -n "${SEARCH_PID}" ]; then kill "${SEARCH_PID}" 2>/dev/null || true; fi
}
trap cleanup EXIT

echo "Starting search server on cuda at port ${SEARCH_PORT}"
lsof -ti :"${SEARCH_PORT}" | xargs -r kill -9 || true
python -m opta.search_server \
  --port "${SEARCH_PORT}" \
  --host 0.0.0.0 \
  --model "${EMBEDDING_MODEL_PATH}" \
  > /dev/null 2>&1 &
SEARCH_PID=$!
sleep 20

echo "Starting vLLM server on cuda at port ${VLLM_PORT}"
lsof -ti :"${VLLM_PORT}" | xargs -r kill -9 || true
python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${MODEL_NAME}" \
  --port "${VLLM_PORT}" \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --max-model-len "${MAX_MODEL_LEN}" \
  --host 0.0.0.0 \
  > /dev/null 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM server..."
sleep 60
ready=false
for _ in {1..3000}; do
  if ss -tuln | grep -q ":${VLLM_PORT} "; then
    ready=true
    echo "vLLM server is ready on port ${VLLM_PORT}"
    break
  fi
  sleep 5
done
if [ "${ready}" != true ]; then
  echo "vLLM server did not become ready"
  exit 1
fi

CMD=(
  python -u -m opta.evaluator
  --data_path "${DATA_PATH}"
  --output_dir "${OUTPUT_DIR}"
)

if [ -n "${NUM_SAMPLES}" ] && [ "${NUM_SAMPLES}" != "all" ] && [ "${NUM_SAMPLES}" != "none" ]; then
  CMD+=(--num_samples "${NUM_SAMPLES}")
fi

echo "Running evaluator: workers=${OPTA_WORKERS}, trajectories=${OPTA_TRAJECTORIES}"
("${CMD[@]}") &
EVAL_PID=$!

if ! wait "${EVAL_PID}"; then
  echo "Evaluator failed."
  exit 1
fi

echo "Done. Outputs saved to ${OUTPUT_DIR}"
