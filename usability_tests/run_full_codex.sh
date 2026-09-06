#!/usr/bin/env bash
# Run one full Codex repeat across light, medium and high effort. Run repeat 1
# and complete its audit/bookkeeping before starting repeat 2.
#
# Usage from the repository root:
#   PRICE_IN=10 PRICE_OUT=50 PRICE_CACHE_READ=1 PRICE_CACHE_WRITE=12.5 \
#     nohup usability_tests/run_full_codex.sh 1 1 \
#     > log/usability-codex-full-r1.txt 2>&1 &
#
# Environment: AGENT and MODEL are required. Do not use AGENT=astra: that
# namespace contains the excluded macOS pilot cells committed with the kit.
# TIMEOUT (default 1800 seconds). The second positional argument is parallelism
# and defaults to 1 because checkpoint cells have substantial RAM peaks.
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "run_full_codex.sh requires Linux because the frozen F environment contains Linux-only packages" >&2
  exit 2
fi

REPEAT="${1:-1}"
PARALLEL="${2:-1}"
: "${AGENT:?set AGENT to the frozen official cohort id (not astra)}"
: "${MODEL:?set MODEL to the exact participant model id}"
if [[ "$AGENT" == "astra" ]]; then
  echo "AGENT=astra is reserved for excluded macOS pilot cells; choose a new official cohort id" >&2
  exit 2
fi
TIMEOUT="${TIMEOUT:-1800}"
: "${PRICE_IN:?set PRICE_IN from the current vendor rate card (USD per million input tokens)}"
: "${PRICE_OUT:?set PRICE_OUT from the current vendor rate card (USD per million output tokens)}"
: "${PRICE_CACHE_READ:?set PRICE_CACHE_READ from the current vendor rate card}"
: "${PRICE_CACHE_WRITE:?set PRICE_CACHE_WRITE from the current vendor rate card}"

cd "$(dirname "$0")/.."
PY=.venv/bin/python

command -v codex >/dev/null
$PY usability_tests/make_manifest.py --verify

if (( REPEAT > 1 )); then
  missing=0
  for prior_effort in light medium high; do
    for target in gpt-2 olmo-1b pythia-1b; do
      for test in T1 T2 T3 T4 T5; do
        for condition in P F B; do
          prior_dir="usability_tests/$AGENT/$target/$prior_effort/$test-$condition-$((REPEAT - 1))"
          for record in harness.json grade.json review.json; do
            if [[ ! -f "$prior_dir/$record" ]]; then
              echo "missing prior-repeat record: $prior_dir/$record" >&2
              missing=1
            fi
          done
        done
      done
    done
  done
  if (( missing )); then
    echo "refusing repeat $REPEAT until all repeat $((REPEAT - 1)) cells are complete" >&2
    exit 2
  fi
fi

for effort in light medium high; do
  echo "=== $(date -Is) $AGENT $effort repeat $REPEAT"
  $PY usability_tests/run_matrix_codex.py \
    --agent "$AGENT" --model "$MODEL" --effort "$effort" \
    --repeat "$REPEAT" --parallel "$PARALLEL" --venv --timeout "$TIMEOUT" \
    --price-in "$PRICE_IN" --price-out "$PRICE_OUT" \
    --price-cache-read "$PRICE_CACHE_READ" --price-cache-write "$PRICE_CACHE_WRITE" \
    --log-dir "log/usability-$AGENT-$effort-r$REPEAT"
done

echo "=== $(date -Is) repeat $REPEAT complete; audit and fill bookkeeping before another repeat"
$PY usability_tests/audit_codex.py --agent "$AGENT" || true
