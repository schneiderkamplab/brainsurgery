#!/usr/bin/env bash
# Run the full Claude matrix: every (agent, effort) with run_matrix.py, sequentially,
# cheapest agent first, resumable (graded cells are skipped; infrastructure
# failures are rerun). Usage from the repository root:
#
#   nohup usability_tests/run_full_claude.sh [REPEAT] [PARALLEL] > log/usability-full-driver.txt 2>&1 &
#
# Environment: TIMEOUT (default 1800 s per cell), MAX_TURNS (default 40).
set -uo pipefail
REPEAT="${1:-1}"
PARALLEL="${2:-4}"
TIMEOUT="${TIMEOUT:-1800}"
MAX_TURNS="${MAX_TURNS:-40}"
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for pair in "sonnet5 claude-sonnet-5" "opus5 claude-opus-5" "fable51 claude-fable-5-1"; do
  set -- $pair
  agent=$1; model=$2
  for effort in low medium high; do
    echo "=== $(date -Is) $agent $effort repeat $REPEAT"
    $PY usability_tests/run_matrix.py --agent "$agent" --model "$model" --effort "$effort" \
      --repeat "$REPEAT" --parallel "$PARALLEL" --venv --max-turns "$MAX_TURNS" --timeout "$TIMEOUT" \
      --log-dir "log/usability-$agent-$effort"
  done
done
echo "=== $(date -Is) full matrix done"
