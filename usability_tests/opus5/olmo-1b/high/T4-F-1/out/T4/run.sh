#!/bin/sh
# Runs the T4 merge. Exits non-zero if any required check fails.
set -eu
cd "$(dirname "$0")/../.."
exec .venv/bin/python out/T4/solution.py
