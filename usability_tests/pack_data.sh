#!/usr/bin/env bash
# Bundle the generated study data (synthetic fine-tunes, LoRA adapters, hidden
# references) for transfer to another machine, so it need not be regenerated
# there. Base checkpoints are not included (download them from HuggingFace at
# the pinned revisions in targets.py).
#
#   usability_tests/pack_data.sh [OUT.tar]        default: log/usability_tests-data.tar
#
# On the receiving machine, with the base models already under models/:
#   tar -xf usability_tests-data.tar -C models/     # creates models/usability_tests/<target>/{inputs,references}
#   .venv/bin/python usability_tests/setup.py       # links base files, skips existing generated data
#   .venv/bin/python usability_tests/make_manifest.py --verify
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-log/usability_tests-data.tar}"
DATA_ROOT="$(readlink -f models/usability_tests)"
echo "packing $DATA_ROOT -> $OUT (generated inputs and references, ~35 GB)"
tar -cf "$OUT" -C "$(dirname "$DATA_ROOT")" \
  --exclude='usability_tests/*/inputs/base' \
  --exclude='usability_tests/*/inputs/ft1/*.json' --exclude='usability_tests/*/inputs/ft1/*.txt' \
  --exclude='usability_tests/*/inputs/ft2/*.json' --exclude='usability_tests/*/inputs/ft2/*.txt' \
  usability_tests
ls -la "$OUT"
sha256sum "$OUT"
