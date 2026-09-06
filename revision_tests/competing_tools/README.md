# Competing-tool equivalence study

This directory defines the controlled comparison requested in the EACL 2027
reviews. It compares only end-to-end checkpoint operations with genuinely
overlapping semantics and validates every output against the same independent
oracle.

| Case | Operation | BrainSurgery | MergeKit | torch-state-bridge |
|---|---|:---:|:---:|:---:|
| R01 | Regex layer-key rename | yes | no | yes |
| M01 | Two-checkpoint weighted linear merge | yes | yes | no |
| M02 | Base-relative two-task-vector arithmetic | yes | yes | no |

“No” means the case is outside that tool's documented operation model; it is
not recorded as a failed run. Orbax is positioned separately in `protocol.md`
because it is a JAX checkpointing/restore framework rather than a like-for-like
PyTorch/Hugging Face checkpoint-editing command for these cases.

## What is ready on macOS

The fixture generator, neutral case definitions, tool-facing specification
renderer, independent oracle, process monitor, result analyzer, and negative
controls can all be validated locally:

```bash
.venv/bin/python revision_tests/competing_tools/validate_protocol.py
.venv/bin/python -m pytest -q revision_tests/competing_tools/test_competing_tools.py
```

A full local preflight requires isolated installations of both competitors.
The preflight is deliberately stamped `reported_eligible=false`; its timing is
not paper evidence.

On 2026-09-06, the actual-package macOS preflight passed all six measured
tool/case pairs on both the tiny fixture and a fixture derived from the pinned
GPT-2 checkpoint below. R01 preserved all 161 tensors byte-exactly. M01 and M02
validated all 148 common arithmetic parameters; the largest error was
`2.9802322387695312e-08`. This establishes harness integration and output
equivalence only. It is not a timing result, a scaling result, or evidence that
the synthetically offset checkpoints have downstream quality.

## Frozen Linux environment

Use separate tool environments and the same Torch, safetensors, and NumPy
versions so dependency resolution cannot advantage or break a competitor. The
controller may use the BrainSurgery environment:

```bash
uv venv --python 3.13 .comparison_envs/brainsurgery
uv pip install --python .comparison_envs/brainsurgery/bin/python \
  --editable . torch==2.14.0 safetensors==0.5.3 numpy==2.4.6

uv venv --python 3.11 .comparison_envs/mergekit
uv pip install --python .comparison_envs/mergekit/bin/python \
  mergekit==0.1.4 transformers==4.57.1 \
  torch==2.14.0 safetensors==0.5.3 numpy==2.4.6

uv venv --python 3.11 .comparison_envs/torch_state_bridge
uv pip install --python .comparison_envs/torch_state_bridge/bin/python \
  torch-state-bridge==0.1.0 torch==2.14.0 \
  safetensors==0.5.3 numpy==2.4.6
```

Before the reported run, save complete `uv pip freeze` output for all three
environments and verify installed versions against `tools.yaml`. Do not install
competitors into the BrainSurgery environment.

## Mac preflight

```bash
.comparison_envs/brainsurgery/bin/python revision_tests/competing_tools/run.py \
  --run-id <unique_run_id> --smoke --repetitions 1 \
  --brainsurgery-cli .comparison_envs/brainsurgery/bin/brainsurgery \
  --mergekit-cli .comparison_envs/mergekit/bin/mergekit-yaml \
  --torch-state-bridge-python .comparison_envs/torch_state_bridge/bin/python
```

## Reported Linux run

Run from a clean checkout on the same local filesystem and with no competing
jobs. The protocol uses an unmeasured warm-up followed by five measured runs
per tool/case pair and a deterministic interleaved schedule:

```bash
OMP_NUM_THREADS=1 .comparison_envs/brainsurgery/bin/python \
  revision_tests/competing_tools/run.py \
  --run-id <unique_run_id> --repetitions 5 --num-threads 1 \
  --source-model models/gpt2 \
  --source-id openai-community/gpt2 \
  --source-revision 607a30d783dfa663caf39e06633721c8d4cfcd7e \
  --workload-note "no other material workload observed" \
  --brainsurgery-cli .comparison_envs/brainsurgery/bin/brainsurgery \
  --mergekit-cli .comparison_envs/mergekit/bin/mergekit-yaml \
  --torch-state-bridge-python .comparison_envs/torch_state_bridge/bin/python
```

Raw artifacts are written to
`log/revision_tests/<run_id>/competing_tools/`. The runner refuses to overwrite
an existing run, preserves failed outputs and diagnostics, and deletes only
successfully validated repetition outputs unless `--keep-outputs` is supplied.
The runner records the upstream identifier and revision label, hashes the local
checkpoint files and every derived tensor, rechecks fixture and protocol hashes
before and after measurement, and rejects reportable runs with a dirty checkout,
shared environments, wrong dependency versions, fewer than five repetitions,
or missing operator workload disclosure.

## Interpretation boundary

The primary endpoint is output correctness, not speed. Runtime and peak RSS are
reported only for outputs that satisfy the neutral contract. Each number is an
end-to-end invocation including checkpoint load and save. Do not interpret
these three cases as overall tool rankings: MergeKit supports a much broader
model-merging workflow, while torch-state-bridge is a small key-rewriting
library whose persistence wrapper is part of this study's measured path.

The usability study's condition F remains separate: it measures an agent
choosing among allowed packages, while this directory controls the operation,
fixture, oracle, and execution environment.

## Source and submission boundary

Tool scope and version choices were checked against the official
[MergeKit repository](https://github.com/arcee-ai/mergekit),
[torch-state-bridge package page](https://pypi.org/project/torch-state-bridge/),
and [Orbax documentation](https://orbax.readthedocs.io/en/latest/). The
[EACL 2027 demo guidelines](https://2027.eacl.org/calls/demos/#submission-guidelines)
motivate the reproducible evaluation artifact, but this table alone does not
satisfy the track's broader paper, system, and demonstration requirements.

Full raw records contain local paths and a hostname and must remain private
during anonymous review. After a Linux run passes every reporting gate, create
a text-only candidate supplement with:

```bash
.comparison_envs/brainsurgery/bin/python \
  revision_tests/competing_tools/export_anonymous.py \
  --input log/revision_tests/<run_id>/competing_tools \
  --output <new_anonymous_export_directory>
```

This excludes model/output binaries, redacts recorded local roots and the
hostname, and emits a checksum manifest. It is a first-pass safeguard, not a
substitute for a human anonymity review of the entire submission package and
anonymous code repository.
