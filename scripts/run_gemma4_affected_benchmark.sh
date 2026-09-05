#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${1:-log-gemma4-inline-remove-builtins-p6}"

conda run --no-capture-output -n brainsurgery python -m scripts.run_gemma4_affected_benchmark_py "$LOG_DIR"
