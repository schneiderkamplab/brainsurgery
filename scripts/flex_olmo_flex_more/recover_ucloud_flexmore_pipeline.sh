#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

BASE_MODEL_ROOT="${BASE_MODEL_ROOT:-/work/training/FlexMoRE/models}"
RANKED_MODEL_ROOT="${RANKED_MODEL_ROOT:-/work/training/FlexMoRE/models}"
MERGE_CONDA_ENV="${MERGE_CONDA_ENV:-flexolmo}"
EVAL_CONDA_ENV="${EVAL_CONDA_ENV:-olmes_production}"
REVISION="${REVISION:-}"
SHOW_REVISIONS_ONLY="${SHOW_REVISIONS_ONLY:-0}"
SKIP_EVAL_CHECK="${SKIP_EVAL_CHECK:-0}"
RUN_WRAPPER="${RUN_WRAPPER:-0}"

print_usage() {
  cat <<'EOF'
Usage:
  bash src/scripts/flexmore/recover_ucloud_flexmore_pipeline.sh

Environment variables:
  REVISION=<n>            Roll back MERGE_CONDA_ENV to this conda revision before checks.
  SHOW_REVISIONS_ONLY=1   Only print conda revisions for MERGE_CONDA_ENV, then exit.
  SKIP_EVAL_CHECK=1       Skip checking oe_eval in EVAL_CONDA_ENV.
  RUN_WRAPPER=1           After checks, run build_and_eval_flexmore_v02_selected.sh.
  BASE_MODEL_ROOT=...     Defaults to /work/training/FlexMoRE/models
  RANKED_MODEL_ROOT=...   Defaults to /work/training/FlexMoRE/models
  MERGE_CONDA_ENV=...     Defaults to flexolmo
  EVAL_CONDA_ENV=...      Defaults to olmes_production

Examples:
  SHOW_REVISIONS_ONLY=1 bash src/scripts/flexmore/recover_ucloud_flexmore_pipeline.sh
  REVISION=12 bash src/scripts/flexmore/recover_ucloud_flexmore_pipeline.sh
  REVISION=12 RUN_WRAPPER=1 bash src/scripts/flexmore/recover_ucloud_flexmore_pipeline.sh
EOF
}

require_command() {
  local cmd="$1"
  command -v "${cmd}" >/dev/null 2>&1 || {
    echo "Missing required command: ${cmd}" >&2
    exit 1
  }
}

require_path() {
  local path="$1"
  [[ -e "${path}" ]] || {
    echo "Missing required path: ${path}" >&2
    exit 1
  }
}

show_revisions() {
  echo "Conda revisions for ${MERGE_CONDA_ENV}:"
  conda list -n "${MERGE_CONDA_ENV}" --revisions
}

rollback_if_requested() {
  if [[ -z "${REVISION}" ]]; then
    return
  fi
  echo "Rolling ${MERGE_CONDA_ENV} back to conda revision ${REVISION}"
  conda install -n "${MERGE_CONDA_ENV}" --revision "${REVISION}" -y
}

check_merge_env() {
  echo "Checking merge env: ${MERGE_CONDA_ENV}"
  conda run -n "${MERGE_CONDA_ENV}" python -c "from transformers import FlexOlmoConfig, FlexOlmoForCausalLM; print('transformers ok')"
  conda run -n "${MERGE_CONDA_ENV}" python src/scripts/flexmore/merge_experts_to_flexolmo.py --help >/dev/null
  echo "Merge env looks usable."
}

check_eval_env() {
  if [[ "${SKIP_EVAL_CHECK}" == "1" ]]; then
    echo "Skipping eval env check."
    return
  fi
  echo "Checking eval env: ${EVAL_CONDA_ENV}"
  conda run -n "${EVAL_CONDA_ENV}" python -c "import oe_eval; print('oe_eval ok')"
  echo "Eval env looks usable."
}

run_wrapper() {
  echo "Running build_and_eval_flexmore_v02_selected.sh"
  BASE_MODEL_ROOT="${BASE_MODEL_ROOT}" \
  RANKED_MODEL_ROOT="${RANKED_MODEL_ROOT}" \
  MERGE_CONDA_ENV="${MERGE_CONDA_ENV}" \
  EVAL_CONDA_ENV="${EVAL_CONDA_ENV}" \
  bash src/scripts/flexmore/build_and_eval_flexmore_v02_selected.sh
}

main() {
  require_command conda
  require_path "${BASE_MODEL_ROOT}"
  require_path "${RANKED_MODEL_ROOT}"

  if [[ "${1:-}" == "--help" ]]; then
    print_usage
    exit 0
  fi

  show_revisions

  if [[ "${SHOW_REVISIONS_ONLY}" == "1" ]]; then
    exit 0
  fi

  rollback_if_requested
  check_merge_env
  check_eval_env

  if [[ "${RUN_WRAPPER}" == "1" ]]; then
    run_wrapper
  else
    cat <<EOF

Checks completed.

Next command:
  BASE_MODEL_ROOT=${BASE_MODEL_ROOT} \\
  RANKED_MODEL_ROOT=${RANKED_MODEL_ROOT} \\
  MERGE_CONDA_ENV=${MERGE_CONDA_ENV} \\
  EVAL_CONDA_ENV=${EVAL_CONDA_ENV} \\
  bash src/scripts/flexmore/build_and_eval_flexmore_v02_selected.sh | tee logs/flexmore_v02_selected_eval.log
EOF
  fi
}

main "$@"
