#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

BASE_MODEL_ROOT="${BASE_MODEL_ROOT:-/work/training/FlexMoRE/models}"
RANKED_MODEL_ROOT="${RANKED_MODEL_ROOT:-/work/training/FlexMoRE/models}"
MERGE_CONDA_ENV="${MERGE_CONDA_ENV:-flexolmo}"
MERGE_DEVICE="${MERGE_DEVICE:-cpu}"
MERGE_DTYPE="${MERGE_DTYPE:-bfloat16}"
GPUS="${GPUS:-1}"
EVAL_CONDA_ENV="${EVAL_CONDA_ENV:-flexolmo}"
LOG_FILE="${LOG_FILE:-${ROOT_DIR}/logs/flexmore_v02_selected_eval.log}"

require_path() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 1
  fi
}

mkdir -p "$(dirname "${LOG_FILE}")"

require_path "${BASE_MODEL_ROOT}"
require_path "${RANKED_MODEL_ROOT}"

echo "Running selected-rank FlexMoRE/FlexOlmo pipeline"
echo "BASE_MODEL_ROOT=${BASE_MODEL_ROOT}"
echo "RANKED_MODEL_ROOT=${RANKED_MODEL_ROOT}"
echo "MERGE_CONDA_ENV=${MERGE_CONDA_ENV}"
echo "MERGE_DEVICE=${MERGE_DEVICE}"
echo "MERGE_DTYPE=${MERGE_DTYPE}"
echo "GPUS=${GPUS}"
echo "EVAL_CONDA_ENV=${EVAL_CONDA_ENV}"
echo "LOG_FILE=${LOG_FILE}"

BASE_MODEL_ROOT="${BASE_MODEL_ROOT}" \
RANKED_MODEL_ROOT="${RANKED_MODEL_ROOT}" \
MERGE_CONDA_ENV="${MERGE_CONDA_ENV}" \
MERGE_DEVICE="${MERGE_DEVICE}" \
MERGE_DTYPE="${MERGE_DTYPE}" \
GPUS="${GPUS}" \
EVAL_CONDA_ENV="${EVAL_CONDA_ENV}" \
LOG_FILE="${LOG_FILE}" \
bash src/scripts/flexmore/build_and_eval_flexmore_v02_selected.sh
