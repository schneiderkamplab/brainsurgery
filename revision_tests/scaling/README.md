# Scaling and systems evaluation

This directory contains the frozen, operation-matched checkpoint-scaling
experiment for the EACL 2027 demo revision. It is separate from
`usability_tests/` and never starts a coding-agent participant session.

## What is compared

The same CPU checkpoint rewrite is executed by:

1. an independent, direct PyTorch/safetensors in-memory baseline;
2. BrainSurgery with the `inmemory` provider;
3. BrainSurgery with the file-backed `arena` provider.

All methods scale every floating-point tensor whose complete key ends in
`.weight` by exactly `0.5`, preserve every other tensor exactly, and write a
safetensors checkpoint with a 512 MiB shard budget. The frozen model/revision
matrix is in `cases.yaml`; the complete claim boundary and measurement rules
are in `protocol.md`.

The ten-checkpoint design separates two questions:

- a primary within-family curve over Pythia 70M, 410M, 2.8B, and 12B;
- architecture/storage generalization pairs over GPT-2 124M/XL 1.5B,
  OLMo 1B/7B, and Qwen2.5 0.5B/7B.

The heterogeneous families are not pooled into one fitted scaling line.
Checkpoint bytes are the primary workload-size variable because the frozen
families use different dtypes; parameter count, dtype bytes, and shard counts
are reported separately.

The operation is deliberately CPU/I/O-bound. CUDA is disabled for every
method, even when the Linux host has a GPU, so these measurements must not be
described as GPU performance. The Linux/CUDA host is still suitable for this
run, but the GPU is recorded only as part of the machine fingerprint.

## Mac preflight (not paper evidence)

This runs a tiny deterministic fixture through all three methods and the
independent oracle. It validates the harness but suppresses timing and memory
from the generated paper table.

```bash
.venv/bin/python -m pytest revision_tests/scaling/test_scaling.py
.venv/bin/python revision_tests/scaling/validate_protocol.py
.venv/bin/python revision_tests/scaling/run.py \
  --run-id mac_scaling_preflight \
  --smoke \
  --repetitions 1
```

Raw artifacts are written beneath
`log/revision_tests/mac_scaling_preflight/scaling/`. Do not cite the Mac
preflight as a systems result.

## Linux reported run

Use a clean checkout of one frozen commit and no competing workload. Download
the exact revisions first:

```bash
.venv/bin/python revision_tests/scaling/download_models.py
.venv/bin/python revision_tests/scaling/validate_protocol.py --check-models
```

Then run one warm-up and at least five measured repetitions per method/model
(30 warm-ups plus 150 measured executions across the frozen matrix):

```bash
.venv/bin/python revision_tests/scaling/run.py \
  --run-id eacl2027_scaling_linux \
  --repetitions 5 \
  --num-workers 1 \
  --workload-note "exclusive host; no concurrent user jobs"
```

Do not pull, change dependencies, move the model cache to another filesystem,
or change worker counts during the run. The runner refuses dirty, non-Linux,
partial-matrix, under-replicated, or correctness-incomplete runs as reportable.
It never overwrites a run directory and retains failed outputs. Correct outputs
are deleted only after validation to control disk use.

## Outputs

Each run records commands, plans, environment and Git provenance, input and
output manifests, process-tree RSS and I/O counters, arena temporary-disk
peaks, validation decisions, and compact Markdown/LaTeX/text reports. Raw
artifacts belong under `log/revision_tests/<run_id>/`; only audited compact
summaries should be committed under `results/`.
