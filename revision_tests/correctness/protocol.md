# Correctness and preservation protocol

Protocol identifier: `eacl2027_correctness_v1`

Status: frozen before the first reported run

## Scope

This evaluation covers BrainSurgery checkpoint transformations only. Axon and
Synapse compilation, execution, and model parity are excluded.

The unit of analysis is one deterministic transformation case applied to a
small safetensors state dictionary. The fixture contains float32, float64,
int64, and boolean tensors with explicit arithmetic sequences or literal
values. No random generator or BrainSurgery implementation is used to construct
the fixture or expected output.

## Claims and endpoints

For each lossless case, the primary endpoints are:

1. the output tensor-name set equals the independently expected name set;
2. every output tensor has the expected shape and dtype;
3. every output tensor is byte-identical to the independent oracle;
4. every tensor outside the declared write-set is byte-identical to its input;
5. the source checkpoint file remains byte-identical after execution.

The bfloat16 conversion case is intentionally lossy. Its changed tensor must be
byte-identical to PyTorch's independently computed `float32 -> bfloat16`
conversion, while all tensors outside its write-set remain exact.

The sharded case additionally requires:

1. more than one shard;
2. an index whose weight map covers every tensor exactly once;
3. an index `total_size` equal to the sum of tensor payload bytes;
4. independent loading of the shards reproduces the oracle exactly;
5. loading the sharded output through a second BrainSurgery identity plan and
   saving it as one file also reproduces the oracle exactly.

All primary endpoints must pass. There is no aggregate tolerance or majority
criterion.

## Frozen case matrix

The cases and plans are serialized in `cases.yaml`:

| ID | Operation | Classification | Declared write-set |
|---|---|---|---|
| C01 | Identity serialization through an explicit multiply-by-one no-op | lossless | `math.a` |
| C02 | Rename followed by inverse rename | lossless/metamorphic | `layer.0.weight` |
| C03 | Copy | lossless | `copy.clone` |
| C04 | Move/rename | lossless | source and destination names |
| C05 | Delete | lossless structural edit | deleted name |
| C06 | Split followed by concatenate | lossless/metamorphic | `matrix.weight` |
| C07 | Add, subtract, multiply, and scale | exact arithmetic oracle | four destinations |
| C08 | Same-dtype cast | lossless | `cast.same` |
| C09 | Float32-to-bfloat16 cast | intentionally lossy | `cast.lossy` |
| C10 | Sharded save and reload through an explicit multiply-by-one no-op | lossless/serialization | `math.a` |

“Lossless structural edit” means that the requested deletion or name change is
itself intentional; it does not mean that the deleted tensor remains in the
output.

The CLI infers the output model from mutating transform destinations and
therefore rejects a zero-transform output plan. C01 and both stages of C10 use
an explicit multiply-by-one `scale_` on `math.a` to select the output model; the
oracle requires the resulting tensor bytes to remain identical.

## Independence controls

- `oracle.py` may import PyTorch but may not import `brainsurgery`.
- The tested implementation is invoked only through its public CLI in a fresh
  subprocess.
- Plans and expected states are encoded separately: plans in `cases.yaml`,
  expected transformations in `oracle.py`.
- The verifier runs three negative controls by corrupting an otherwise correct
  state: one tensor value, one dtype, and the tensor-name set. The run is invalid
  unless all corruptions are detected.

This does not constitute a formal proof for arbitrary checkpoints or plans. It
provides independently reproducible evidence for the enumerated primitives and
their compositions.

## Secondary observations

Safetensors custom header metadata is recorded for the fixture and outputs but
is not a primary endpoint. BrainSurgery currently exposes a tensor-state-dict
interface, not a general byte-preserving checkpoint-container copier. Any
metadata difference is reported explicitly.

File-level byte equality is also not required: safetensors serialization may
change header ordering or metadata while retaining the exact logical tensor
state.

## Reproducibility record

Every run records:

- protocol identifier and case-file checksum;
- repository commit and relevant-path cleanliness;
- exact command;
- platform, CPU, memory, Python, PyTorch, safetensors, and BrainSurgery versions;
- fixture file and logical tensor checksums;
- actual generated plans and checksums;
- CLI exit code, stdout, stderr, and duration;
- complete endpoint and negative-control results.

Raw outputs live under `log/revision_tests/<run_id>/`. Compact results may be
committed below `revision_tests/correctness/results/`.

## Reporting

Report counts, not only percentages: cases passing, tensors matching the
oracle, untouched tensors verified exact, source files unchanged, shard-index
checks, and negative controls detected. Report metadata preservation separately.
Do not generalize beyond the operations, dtypes, provider, and serialization
formats tested here.
