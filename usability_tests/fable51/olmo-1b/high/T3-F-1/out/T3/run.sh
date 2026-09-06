#!/bin/sh
# Re-run the T3 export from the sandbox root: sh out/T3/run.sh
set -e
cd "$(dirname "$0")/../.."
exec .venv/bin/python out/T3/solution.py
