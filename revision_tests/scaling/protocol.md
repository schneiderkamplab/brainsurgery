# Frozen checkpoint-scaling protocol

Protocol identifier: `eacl2027_scaling_v1`

Status: frozen before the first reported model-scale run

## Question and scope

This experiment asks how wall time, peak resident memory, process I/O, output
sharding, and temporary-disk use change with checkpoint size for equivalent
checkpoint rewrites. It is a systems experiment, not a usability, inference,
training, downstream-quality, Axon, or Synapse experiment.

## Frozen operation and methods

For each checkpoint, scale every floating-point tensor whose full key matches
`.*\.weight` by `0.5`; preserve every nonmatching tensor byte-for-byte; retain
all names, shapes, and dtypes; and write safetensors with a 512 MiB shard
budget. At least one weight must match. A tensor larger than the budget may
occupy a shard alone. Configuration/tokenizer sidecars are outside the output
contract and are neither copied nor counted.

The methods are an independent direct PyTorch/safetensors in-memory script,
BrainSurgery `inmemory`, and BrainSurgery `arena`. All use CPU tensors, one I/O
worker, fixed thread environment variables, and `CUDA_VISIBLE_DEVICES=""`.
The arena segment size is 4 GiB so one large tensor fits in a segment.

## Model matrix and analysis groups

`cases.yaml` freezes ten base-model checkpoints at exact Hugging Face commit
hashes:

- primary within-family scaling: Pythia 70M, 410M, 2.8B, and 12B;
- GPT-2 architecture pair: 124M and XL 1.5B;
- OLMo architecture/storage pair: 1B and 7B;
- Qwen2.5 architecture/storage pair: 0.5B and 7B.

The Pythia points form the primary fitted scaling curve. GPT-2, OLMo, and
Qwen2.5 are paired generalization checks and must be displayed as separate
families, not pooled into a single parameter-count regression. Llama is not
included: for this architecture-neutral checkpoint operation it adds less
structural coverage than the selected legacy/full-attention/GQA families while
adding gated-access and license friction.

Nominal family size is recorded separately from the exact stored tensor-element
count, logical tensor bytes, checkpoint file bytes, dtype-specific bytes, and
shard counts computed from each input. Checkpoint bytes are the primary
workload-size axis because the frozen checkpoints use float16, float32, and
bfloat16. The source layout and matched-weight dtype must equal their
declarations in `cases.yaml`.

## Schedule and cache policy

Run on one otherwise idle Linux machine and one filesystem. Each model/method
receives one unmeasured warm-up. Five or more measured repetitions follow in a
deterministic rotating method order. The OS page cache is not dropped, so the
estimand is explicitly warm-cache end-to-end execution. Input, output shard
budget, worker count, storage, validation, and environment are identical
across methods. Models execute in the frozen matrix order. Family membership
and analysis role are recorded for every point.

## Measurements

The parent harness samples the complete process tree and records wall time,
peak summed RSS, process read/write byte counters, and whether sampling
degraded. It also samples logical and allocated bytes under the method's arena
temporary directory. Logical checkpoint input/output bytes, shard count,
parameter count, tensor count, and effective logical throughput
`(input bytes + output bytes) / wall time` are recorded separately from OS I/O
counters. GPU peak memory is not applicable because the operation is forced to
CPU; the host GPU inventory is provenance only.

## Independent correctness gate

The oracle uses safetensors and PyTorch directly and does not import
BrainSurgery. It requires identical tensor-name sets, shapes, and dtypes; the
exact independently computed `input * 0.5` value for matched tensors; exact
bytes for nonmatched tensors; valid index coverage; the shard-size rule; and
unchanged source files. It emits per-tensor and per-file hashes. A process or
validation failure makes that attempt timing-ineligible. Failed outputs are
retained; valid outputs may be removed only after validation.

## Reporting gate and claim boundary

A reportable candidate requires Linux, a clean frozen Git checkout, all ten
pinned models, the declared layouts/dtypes and revision metadata, all three methods,
at least five correct measured repetitions per pair, no degraded resource
sampling, unchanged inputs, and one explicit operator workload note. The
runner suppresses performance fields in paper fragments when any gate fails or
when `--smoke` is used.

Report medians plus the full repetition values (or a dispersion statistic),
not a single best run. Use actual checkpoint bytes for the primary systems
axis and show parameter count as complementary information. Do not mix Mac and
Linux values, different machines, or heterogeneous families in an
undifferentiated scaling fit. The results support only the frozen CPU
checkpoint rewrite on the enumerated models/revisions and hardware. They do
not establish general speed superiority, GPU performance, training/inference
scaling, or behavior preservation after the intentionally non-identity edit.
