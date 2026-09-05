#!/usr/bin/env bash

set -euo pipefail

brainsurgery synapse axon-benchmark-render \
  log-min20b-pp4/stream.csv \
  --table-format html > log-min20b-pp4/partial.html
