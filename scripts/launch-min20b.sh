#!/usr/bin/env bash

set -euo pipefail

OMP_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=0,1,2,3 \
brainsurgery synapse axon-benchmark \
  brainsurgery/synapse/models \
  --device cuda \
  --processes 1 \
  --axon-backend pipeline \
  --pipeline-parallel-size 4 \
  --min-billion-parameters 20 \
  --max-billion-parameters 150 \
  --log-dir log-min20b-pp4 \
  --stream-csv log-min20b-pp4/stream.csv
