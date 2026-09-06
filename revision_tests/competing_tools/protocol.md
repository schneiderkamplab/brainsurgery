# Competing-tool comparison protocol

Protocol identifier: `eacl2027_competing_tools_v1`

Status: frozen before the first reported systems run

## Question

For a small set of genuinely overlapping checkpoint operations, do the tools
produce the same independently specified output, and what end-to-end time and
peak resident memory do correct runs require on identical hardware and storage?

The comparison responds to reviewer requests without claiming that one tool is
universally better. Feature breadth, interface style, and performance are
separate dimensions.

## Tool and case inclusion

MergeKit 0.1.4 is compared on a two-checkpoint linear merge and a base-relative
two-task-vector merge. Its documented layer slicing, Frankenmerging, LoRA
extraction, MoE construction, tokenizer reconciliation, evolutionary search,
and multi-stage workflows are acknowledged but not collapsed into these narrow
arithmetic cases.

torch-state-bridge 0.1.0 is compared on regex/capture key rewriting. It is an
in-process state-dictionary library rather than a checkpoint CLI, so its frozen
adapter performs safetensors load, transformation, and save. The wrapper is
timed and disclosed. The PyPI description claims an MIT license, but the wheel's
Core Metadata contains neither a license nor a project URL; this provenance
limitation must remain visible.

An unconstrained MergeKit 0.1.4 installation resolved to Transformers 5.12.1
on 2026-09-06 and failed before execution in its Pydantic architecture model.
The frozen environment therefore pins Transformers 4.57.1, from the supported
4.x API line, without patching MergeKit. This compatibility finding is
reported as protocol preparation rather than counted as a benchmark failure.

Orbax Checkpoint 0.12.3 is not an executable baseline here. Orbax targets JAX
checkpoint persistence, restore-time PyTree transformation, distributed
resharding, and related storage concerns. Forcing the PyTorch/Hugging Face
safetensors cases through a JAX-format conversion would measure migration
overhead rather than equivalent checkpoint surgery. Orbax is therefore covered
as adjacent related work, not marked unsupported or failed.

## Inputs and independent oracle

`prepare.py` deterministically creates one tiny GPT-2-shaped float32 fixture
family and a separate mixed-dtype rename fixture. The tensor names and shapes
are sufficient for MergeKit's pinned GPT-2 architecture description, while the
small payload makes preflight and negative controls practical. Every fixture
file, config, tensor, and rendered tool specification is hashed in the run. The
reportable run records the upstream model identifier, requested revision, and
hashes of the actual local checkpoint files; the hashes, rather than the path
or revision label alone, establish the bytes that were tested.

For the reportable-size GPT-2 fixture, R01 includes every source tensor. M01 and
M02 use the exact parameter-name intersection exposed by MergeKit's pinned
GPT-2 architecture definition. Persistent causal-mask buffers that MergeKit
does not enumerate are excluded from both tools' arithmetic inputs and listed
by name and hash in the fixture manifest. The optional tied `lm_head.weight` is
not synthesized when absent. Source aliases such as `h.*` and `wte.*` are
canonicalized to the Hugging Face `transformer.*` form before either tool runs,
and the complete source-to-contract mapping is recorded. This prevents either
tool from being penalized for serialization aliases or for preserving tensors
the comparison tool never reads.

`oracle.py` imports neither BrainSurgery nor either competitor. It computes the
three output contracts directly with PyTorch and validates names, tensor count,
shape, dtype, exact bytes for R01, and `atol=rtol=1e-6` plus maximum and mean
absolute error for M01/M02. Arithmetic cases use float32 throughout. A timing
sample is ineligible unless its output passes this oracle.

The final paper comparison must repeat the same cases on a named real,
revision-pinned checkpoint family. Tiny-fixture results validate semantics and
the harness but are not sufficient evidence for a scaling claim.

## Execution and measurements

- three isolated tool environments, with the controller using the
  BrainSurgery environment;
- Torch 2.14.0, safetensors 0.5.3, and NumPy 2.4.6 shared across tools;
- the exact repository commit and package freezes recorded;
- one machine, local filesystem, CPU execution, and `OMP_NUM_THREADS=1`;
- one unmeasured warm-up for every tool/case pair;
- five measured repetitions in a deterministic interleaved order;
- a new output directory per attempt, never overwritten;
- wall-clock duration from process start through completed checkpoint save;
- peak aggregate RSS of the process tree sampled every 10 ms;
- process-tree read/write byte counters when the OS exposes them;
- output bytes, tensor count, manifest, exit status, and validation result;
- median, minimum, maximum, arithmetic mean, sample standard deviation, and
  individual measurements reported together.

The cache policy is explicitly warm-cache. The warm-up is unmeasured and not
used for correctness counts. The runner does not drop operating-system caches.
CPU affinity, filesystem, storage device, concurrent workload, and thread count
must be recorded. The operator supplies an explicit concurrent-workload note.
GPU is not used for these I/O/CPU cases.

Successful outputs may be removed after their tensor manifests and validation
records are persisted; failed or invalid outputs are retained. Raw artifacts
remain under `log/revision_tests/<run_id>/competing_tools/`.

## Decision and reporting

Correctness is primary. R01 passes only with byte-exact tensors. M01 and M02
pass only when all tensors satisfy the frozen tolerance. Report both counts and
rates, never percentages alone. Timing summaries exclude warm-ups, crashes,
timeouts, and invalid outputs, with excluded counts shown.

Report paired tool/case measurements and ratios with the raw distributions. Do
not pool cases, extrapolate tiny-fixture timings to large checkpoints, interpret
configuration line count as usability, or infer a general tool ranking.

## Negative controls and stop conditions

Unit tests must demonstrate detection of a changed value, a renamed/missing
tensor, a dtype change, an invalid arithmetic result, and incompatible case or
protocol metadata. A reported run stops if:

- any pinned version differs;
- the Git worktree is dirty;
- a fixture/specification checksum changes;
- environments share a site-packages directory;
- fewer than five correct measured repetitions exist for any expected pair;
- another material workload is discovered during the run; or
- the filesystem or thread policy differs between tools.

Dependency/install failure is reported as such and is not silently repaired by
changing the pinned protocol mid-run.
