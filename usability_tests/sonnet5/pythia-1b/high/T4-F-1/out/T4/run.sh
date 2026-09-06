#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
.venv/bin/python3.13 out/T4/solution.py
