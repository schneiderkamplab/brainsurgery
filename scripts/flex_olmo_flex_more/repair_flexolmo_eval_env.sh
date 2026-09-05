#!/usr/bin/env bash
set -euo pipefail

# Repairs the current `flexolmo` conda env back to the dependency stack this
# repo expects for FlexMoRE merge + OLMES eval.
#
# What it does:
# 1. Saves a snapshot of the current env state.
# 2. Reinstalls the expected torch / torchvision / torchaudio trio.
# 3. Reinstalls ai2-olmo and ai2-olmo-core to the matching versions.
# 4. Reinstalls the transformers fork referenced by the repo eval setup.
# 5. Installs the OLMES eval dependency that provides `oe_eval`.
# 6. Verifies the critical imports used by merge/eval.
# 7. Optionally restores the original merge script from git.
#
# Intended use on UCloud:
#   conda activate flexolmo
#   bash src/scripts/flexmore/repair_flexolmo_eval_env.sh
#
# Optional:
#   RESTORE_MERGE_SCRIPT=1 bash src/scripts/flexmore/repair_flexolmo_eval_env.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
RESTORE_MERGE_SCRIPT="${RESTORE_MERGE_SCRIPT:-1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers@git+https://github.com/swj0419/transformers}"
OLMES_SPEC="${OLMES_SPEC:-ai2-olmes@git+https://github.com/allenai/olmes@4f04122642bcee6d74393ec2ecfb0e572a64da53}"

mkdir -p "${LOG_DIR}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "No active conda env detected. Activate 'flexolmo' first." >&2
  exit 1
fi

echo "Using conda env: ${CONDA_PREFIX}"

snapshot_env() {
  echo "Saving current env snapshot to ${LOG_DIR}"
  python -c "import sys; print(sys.executable)" > "${LOG_DIR}/flexolmo_python_path.txt"
  pip freeze > "${LOG_DIR}/flexolmo_before_repair.txt"
}

restore_merge_script() {
  if [[ "${RESTORE_MERGE_SCRIPT}" != "1" ]]; then
    return
  fi
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Restoring original merge script from git"
    git restore src/scripts/flexmore/merge_experts_to_flexolmo.py || true
  fi
}

reinstall_torch_stack() {
  echo "Reinstalling torch / torchvision / torchaudio"
  pip uninstall -y torch torchvision torchaudio || true
  pip install \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    --index-url "${TORCH_INDEX_URL}"
}

reinstall_olmo_stack() {
  echo "Reinstalling ai2-olmo / ai2-olmo-core"
  pip uninstall -y ai2-olmo ai2-olmo-core || true
  pip install ai2-olmo==0.6.0 ai2-olmo-core==0.1.0
}

reinstall_transformers() {
  echo "Reinstalling transformers fork: ${TRANSFORMERS_SPEC}"
  pip uninstall -y transformers || true
  pip install "${TRANSFORMERS_SPEC}"
}

install_olmes() {
  echo "Installing OLMES eval dependency: ${OLMES_SPEC}"
  pip uninstall -y ai2-olmes || true
  pip install "${OLMES_SPEC}"
}

verify_imports() {
  echo "Verifying critical imports"
  python - <<'PY'
import torch
import torchvision
import torchaudio
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("torchvision:", torchvision.__version__)
print("torchaudio:", torchaudio.__version__)
from transformers import FlexOlmoConfig, FlexOlmoForCausalLM
print("transformers ok")
import olmo_core
print("olmo_core ok")
import oe_eval
print("oe_eval ok")
PY
}

main() {
  snapshot_env
  restore_merge_script
  reinstall_torch_stack
  reinstall_olmo_stack
  reinstall_transformers
  install_olmes
  verify_imports

  cat <<'EOF'

Environment repair completed.

Recommended next commands:
  rm -rf src/scripts/analysis/results/flexmore_v02_selected_models
  BASE_MODEL_ROOT=/work/training/FlexMoRE/models \
  RANKED_MODEL_ROOT=/work/training/FlexMoRE/models \
  MERGE_DEVICE=cpu \
  MERGE_DTYPE=bfloat16 \
  bash src/scripts/flexmore/build_and_eval_flexmore_v02_selected.sh | tee logs/flexmore_v02_selected_eval.log
EOF
}

main "$@"
