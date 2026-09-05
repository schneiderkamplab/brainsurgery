#!/usr/bin/env bash
# Run from the sandbox root (T1-F-1/), using the sandbox's own .venv:
#   ./.venv/bin/python out/T1/solution.py
set -euo pipefail
cd "$(dirname "$0")/../.."
./.venv/bin/python out/T1/solution.py
