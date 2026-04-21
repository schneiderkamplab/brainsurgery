#!/usr/bin/env bash
set -euo pipefail

PARALLEL="${PARALLEL:-24}"
MODELS_ROOT="${MODELS_ROOT:-models}"
SEARCH_ROOT="${SEARCH_ROOT:-brainsurgery/synapse/models}"

if ! [[ "$PARALLEL" =~ ^[0-9]+$ ]] || [ "$PARALLEL" -lt 1 ]; then
  echo "PARALLEL must be a positive integer, got: $PARALLEL" >&2
  exit 2
fi

if [ ! -d "$SEARCH_ROOT" ]; then
  echo "SEARCH_ROOT not found: $SEARCH_ROOT" >&2
  exit 2
fi

mapfile -d '' GENERIC_AXONS < <(find "$SEARCH_ROOT" -type f -name 'generic-*.axon' -print0 | sort -z)
COUNT="${#GENERIC_AXONS[@]}"

if [ "$COUNT" -eq 0 ]; then
  echo "No generic axon files found under $SEARCH_ROOT"
  exit 0
fi

echo "Rematerializing $COUNT generic axon files with PARALLEL=$PARALLEL and MODELS_ROOT=$MODELS_ROOT"

export MODELS_ROOT

find "$SEARCH_ROOT" -type f -name 'generic-*.axon' -print0 \
  | sort -z \
  | xargs -0 -P "$PARALLEL" -I{} bash -lc '
      set -euo pipefail
      file="$1"
      if out="$(brainsurgery synapse axon-materialize "$file" --models-root "$MODELS_ROOT" 2>&1)"; then
        printf "OK\t%s\n" "$file"
      else
        printf "FAIL\t%s\n%s\n" "$file" "$out" >&2
        exit 17
      fi
    ' _ {}

echo "Done."
