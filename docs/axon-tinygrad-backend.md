# Axon Tinygrad Backend

`codegen2-tinygrad` is a separate Graph IR backend. It is selectable through the
same Axon backend option as the torch-backed paths, but it currently stops before
emitting code and reports unsupported/non-obvious primitive operations.

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

These tensor-facing operations still need an explicit tinygrad implementation policy
before they should be emitted:

| Op | Reason |
|---|---|
| `embedding` | Can be expressed as gather/indexing, but requires validated tinygrad integer indexing semantics and weight placement. |
| `linear` | Matmul is obvious; expert slicing, transpose convention, bias/path leaves, and dtype policy need backend-specific handling. |
| `layernorm` | Can be composed, but weight/bias path handling and epsilon/dtype behavior need parity checks. |
| `rmsnorm` | Composable, but dtype/cast behavior needs parity checks. |
| `activations_gelu*` | Composable, but exact vs tanh-approx parity must be chosen and validated. |
| `activations_gegelu` | Compound gated GELU with clipping rules; needs explicit implementation. |
| `activations_xielu` | Custom activation; no obvious tinygrad primitive. |
| `reshape`, `arange` | Tensor operation is simple, but symbolic shape/device/dtype evaluation must be shared with the backend runtime. |
| `slice`, `chunk`, `split`, `concat` | Need Axon list/destructuring semantics and dynamic axis/size policy. |
| `repeat`, `expand`, `permute`, `transpose`, `unsqueeze` | Likely expressible, but tinygrad broadcast/view semantics need parity checks. |
| `softmax`, `sum`, `cumsum` | Likely available, but dtype override/fp32 behavior needs parity checks. |
| `where`, `gather`, `scatter`, `index_add`, `topk`, `where_indices` | Indexed/selective tensor semantics are non-trivial and need direct validation. |
| `tensor_like`, `cast`, `cast_like`, `dtype_value` | Need dtype/device mapping between Axon names and tinygrad dtypes. |
| `empty`, `empty_like`, `fill`, `zeros`, `zeros_like`, `full` | Tensor creation must define initialization, dtype, and placement behavior. |

Pure scalar/elementwise arithmetic and simple transcendental operations are the first
safe implementation slice: `add`, `mul`, `div`, `pow`, `floor`, `sqrt`, `sin`, `cos`,
`exp`, `log`, `matmul`, `le`, `eq`, `and`, `tanh`, `silu`, `sigmoid`, `relu`,
`relu2`, and `clamp`.
