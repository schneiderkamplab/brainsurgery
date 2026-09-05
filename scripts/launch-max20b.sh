#!/usr/bin/env bash

set -euo pipefail

OMP_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
brainsurgery synapse axon-benchmark \
  brainsurgery/synapse/models \
  --device cuda \
  --processes 6 \
  --max-billion-parameters 20 \
  --log-dir log-max20b \
  --stream-csv log-max20b/stream.csv
