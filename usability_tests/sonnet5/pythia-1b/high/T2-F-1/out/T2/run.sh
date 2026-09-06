#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
.venv/bin/python out/T2/solution.py inputs/base/model.safetensors out/T2/model.safetensors
