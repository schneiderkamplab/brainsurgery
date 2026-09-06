# Correctness and preservation evaluation

This directory contains a CPU-only, independently checked evaluation of
BrainSurgery's checkpoint transformations. It does not exercise or import Axon
or Synapse.

## Research question

For transformations presented as lossless, does BrainSurgery produce the
intended tensor state while leaving every tensor outside the declared write-set
exactly unchanged?

The precise protocol and limitations are in [`protocol.md`](protocol.md). The
frozen case matrix is in [`cases.yaml`](cases.yaml). `oracle.py` constructs the
fixture and expected outputs using PyTorch only; it deliberately does not import
BrainSurgery. `run.py` invokes the installed `brainsurgery` CLI in fresh
subprocesses and compares its serialized outputs with those oracles.

## Run

From the repository root:

```bash
.venv/bin/python revision_tests/correctness/run.py
```

The runner creates a unique directory below
`log/revision_tests/<run_id>/correctness/` containing:

- the exact command and environment fingerprint;
- input and generated-plan manifests with SHA-256 checksums;
- the actual plan used for every case;
- captured CLI stdout and stderr;
- tensor-by-tensor comparisons;
- verifier negative-control results;
- `summary.json` and `paper_table.md`.

It exits nonzero if any primary endpoint or verifier control fails. Existing run
directories are never overwritten.

To publish a compact, reviewable copy without raw artifacts:

```bash
.venv/bin/python revision_tests/correctness/run.py \
  --publish-dir revision_tests/correctness/results/<result_id>
```

The published directory contains only the environment, manifests, generated
plans, summary, and paper table. It excludes checkpoint tensors and verbose raw
logs.

## Interpretation

The primary claim is about the tensor state dictionary: names, shapes, dtypes,
and exact tensor bytes. Safetensors custom header metadata and arbitrary
sidecar files are audited separately because BrainSurgery's current checkpoint
interface operates on tensor state dictionaries. Any observed metadata loss
must be disclosed and must not be folded into a broader “no information loss”
claim.
