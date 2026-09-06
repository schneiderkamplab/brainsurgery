#!/usr/bin/env bash
# Run the T4 task-vector merge with the sandbox's private venv.
set -euo pipefail
cd "$(dirname "$0")/../.."
.venv/bin/python out/T4/solution.py
