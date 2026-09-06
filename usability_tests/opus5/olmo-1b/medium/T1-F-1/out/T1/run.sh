#!/bin/sh
# T1 entry point. Run from the sandbox root: sh out/T1/run.sh
set -e
exec "$(dirname "$0")/../../.venv/bin/python" "$(dirname "$0")/solution.py"
