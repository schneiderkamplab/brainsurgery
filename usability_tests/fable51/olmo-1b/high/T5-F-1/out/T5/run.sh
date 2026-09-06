#!/usr/bin/env sh
# Runs the T5 merge-and-shard solution with the sandbox interpreter.
set -eu
cd "$(dirname "$0")/../.."
exec .venv/bin/python out/T5/solution.py
