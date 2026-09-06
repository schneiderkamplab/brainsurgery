#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This reported behavioral run requires Linux." >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "The Git checkout must be clean." >&2
  exit 1
fi
if [[ ! -f models/gpt2/model.safetensors ]]; then
  echo "Missing pinned source checkpoint: models/gpt2/model.safetensors" >&2
  exit 1
fi
if [[ -e models/behavioral_gpt2_lossless ]]; then
  echo "Refusing to overwrite models/behavioral_gpt2_lossless" >&2
  exit 1
fi

RUN_ID="${RUN_ID:-eacl2027_behavioral_cuda_$(git rev-parse --short HEAD)}"
RUN_ROOT="log/revision_tests/${RUN_ID}/behavioral"
GIT_COMMIT="$(git rev-parse HEAD)"
SOURCE_REVISION="607a30d783dfa663caf39e06633721c8d4cfcd7e"

.venv/bin/python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
.venv/bin/python revision_tests/behavioral/validate_manifest.py
.venv/bin/python -m pytest -q revision_tests/behavioral/test_analysis.py

.venv/bin/brainsurgery revision_tests/behavioral/lossless_gpt2_plan.yaml \
  --provider inmemory \
  --num-workers 1 \
  --summary-mode resolve \
  --summarize-path "${RUN_ROOT}/executed_plan.yaml"

# Transformers loads weights from the transformed directory but requires the
# pinned reference architecture and generation configuration as sidecars.
cp models/gpt2/config.json models/behavioral_gpt2_lossless/config.json
cp models/gpt2/generation_config.json \
  models/behavioral_gpt2_lossless/generation_config.json

.venv/bin/python revision_tests/behavioral/validate_lossless.py \
  --source models/gpt2/model.safetensors \
  --transformed models/behavioral_gpt2_lossless \
  --output "${RUN_ROOT}/tensor_validation.json"

.venv/bin/python revision_tests/behavioral/run_model.py \
  --role reference \
  --model models/gpt2 \
  --tokenizer models/gpt2 \
  --revision "${SOURCE_REVISION}" \
  --tokenizer-revision "${SOURCE_REVISION}" \
  --config-revision "${SOURCE_REVISION}" \
  --device cuda:0 \
  --dtype float32 \
  --local-files-only \
  --output "${RUN_ROOT}/reference"

.venv/bin/python revision_tests/behavioral/run_model.py \
  --role transformed \
  --model models/behavioral_gpt2_lossless \
  --tokenizer models/gpt2 \
  --config models/gpt2 \
  --revision "${GIT_COMMIT}_lossless_gpt2_v1" \
  --tokenizer-revision "${SOURCE_REVISION}" \
  --config-revision "${SOURCE_REVISION}" \
  --device cuda:0 \
  --dtype float32 \
  --local-files-only \
  --output "${RUN_ROOT}/transformed"

.venv/bin/python revision_tests/behavioral/analyze.py \
  --reference "${RUN_ROOT}/reference" \
  --transformed "${RUN_ROOT}/transformed" \
  --output "${RUN_ROOT}/comparison.json"

echo "Completed CUDA behavioral evaluation: ${RUN_ROOT}/comparison.json"
