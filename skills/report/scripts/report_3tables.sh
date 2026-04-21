#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <log_dir> [extra args...]" >&2
  exit 2
fi

LOG_DIR="$1"
shift || true

conda run -n brainsurgery python /work/training/brainsurgery/scripts/benchmark_report_3tables.py "$LOG_DIR" "$@"
