#!/usr/bin/env bash
# Run this task's solution from the sandbox root:
#   ./out/T3/run.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
.venv/bin/python out/T3/solution.py
