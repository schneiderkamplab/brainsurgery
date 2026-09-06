#!/bin/sh
# Run T1 from the sandbox root: sh out/T1/run.sh
set -e
exec "$(dirname "$0")/../../.venv/bin/python" "$(dirname "$0")/solution.py"
