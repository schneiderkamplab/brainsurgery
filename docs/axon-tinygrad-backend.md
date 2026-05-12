# Axon Tinygrad Backend

`codegen2-tinygrad` is a separate Graph IR backend. It is selectable through the
same Axon backend option as the torch-backed paths and emits runnable Python
source from Graph IR.

The existing torch-backed backend is named `codegen2-torch`; the interpreter-style
torch path is named `runtime2-torch`; the partitioned torch path is named
`pipeline2-torch`.

Backend-independent Graph IR runtime semantics live in
`brainsurgery/synapse/axon/codegen2_common/`. That layer owns primitive-name
normalization, path composition, config lookup, null handling, cache-length lookup,
and Python container/runtime helper primitives. Tensor-library-specific code stays
in `codegen2_torch` and `codegen2_tinygrad`.

## Non-Obvious Primitive Mappings

These backend-independent operations are shared through `codegen2_common`, not
implemented separately in each tensor backend:

| Op group | Shared behavior |
|---|---|
| `params_*` | State-dict parameter access, root existence, and path normalization. |
| `config_*` | Config lookup, path-template resolution output handling, defaults, and scalar coercions. |
| `shape` | Python list of runtime tensor dimensions. |
| `list_init`, `list_append`, `list_index` | Python container semantics used by flattened Axon. |
| `require` | Null guard semantics. |

The backend maps common transformer tensor primitives to tinygrad and keeps
benchmark integration at the PyTorch public boundary only for inputs/outputs.
Generated models can load normal safetensors directly through
`tinygrad.nn.state.safe_load`, cast floating tensors to the requested dtype, and
move them to the selected tinygrad device. If torch tensors are supplied by a
caller, CUDA tensors are wrapped with `Tensor.from_blob(..., device="CUDA")`
while retaining torch backing tensors for lifetime safety; this is no longer the
benchmark loading path for normal safetensors.

Generated tinygrad models are plain Python callables, not `torch.nn.Module`
subclasses. They expose `forward`, `__call__`, `to`, and `eval` only to satisfy
the benchmark/runtime surface.

Implemented primitives include `embedding`, `linear`, `expert_linear`,
`layernorm`, `rmsnorm`, activation functions, tensor shape/view ops,
indexing/select ops, tensor creation ops, scalar/math ops, and the shared
config/parameter/list primitives.

Tinygrad-specific fixes currently covered by targeted benchmark reruns:

| Area | Implementation |
|---|---|
| `topk(sorted=false)` | Emitted through sorted tinygrad top-k, preserving the selected value/index pairs Axon uses. |
| `index_add` | Emitted through `scatter_reduce(..., reduce="sum")`, with index rank expansion when needed. |
| `expert_linear` | Gathers the per-token expert weight bank with `weight[expert_idx]` and performs the selected batched matmul over the router top-k axis. |
| `unsqueeze(dim=-1)` | Normalized with PyTorch/Axon semantics, appending a trailing singleton axis. |
| tensor placement | Normal safetensors load through tinygrad directly; fallback torch tensors are wrapped directly on the tinygrad CUDA device instead of copied through CPU NumPy. |

Current limitations:

- Execution is correct for the validated rows but not optimized. Public outputs
  are still converted back to torch through NumPy because the benchmark harness
  consumes torch tensors.
- The primitive `_where_indices` is intentionally unsupported in
  `codegen2-tinygrad`. Tinygrad has no direct variable-length
  `nonzero`/`where` index primitive, and the backend must not silently
  materialize it through NumPy. Normal Axon modules such as `MoE.select` and
  `Tensor.where_indices` are not backend special cases; if their lowered graph
  reaches `_where_indices`, tinygrad emission fails with an unsupported
  primitive error.
- Pipeline/parameter distribution is not implemented for tinygrad.
- Coverage has been smoke-tested on `gpt2.axon`.
