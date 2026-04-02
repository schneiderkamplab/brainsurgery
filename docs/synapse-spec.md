# SYNAPSE/1 Specification

## 1. Purpose
SYNAPSE/1 is a declarative, high-level modeling language for defining how tensors and modules are assembled into a PyTorch model.

It is intended to be:
- Readable enough for architecture design reviews.
- Deterministic enough to compile into concrete `torch.nn.Module` implementations.
- Compatible with `brainsurgery` workflows by allowing optional lowering to transform plans for weight import/export and post-build surgery.

## 2. Naming
- Language name: `SYNAPSE`
- Version: `1`
- Document identifier: `SYNAPSE/1`

## 3. Scope
SYNAPSE/1 defines:
- Model graph declaration (modules, dataflow edges, named tensors).
- Shape variables and constraints.
- Reusable blocks and parameter sharing.
- Build targets (inference/train mode metadata, output heads).

SYNAPSE/1 does not define:
- Full training loops.
- Optimizer internals.
- Runtime kernel scheduling.

## 4. Top-Level YAML Shape
A SYNAPSE/1 document is a YAML mapping:

```yaml
synapse: 1
model:
  name: TinyGPT
  dtype: float16
  layout: batch_first
  symbols:
    B: null
    T: null
    D: 768
    H: 12
  inputs:
    tokens: { shape: [B, T], dtype: int64 }
  params:
    vocab_size: 50257
    n_layers: 12
  graph:
    - embed:
        op: embedding
        in: tokens
        out: x
        vocab: ${params.vocab_size}
        dim: D
    - repeat:
        var: l
        range: ${params.n_layers}
        body:
          - block:
              use: transformer_block
              in: x
              out: x
              bindings: { layer_index: ${l} }
  blocks:
    transformer_block:
      inputs:
        x: { shape: [B, T, D] }
      graph:
        - ln1: { op: layernorm, in: x, out: h1, dim: D }
        - attn: { op: mha, in: h1, out: a1, dim: D, heads: H, causal: true }
        - add1: { op: add, in: [x, a1], out: h2 }
        - ln2: { op: layernorm, in: h2, out: h3, dim: D }
        - ff: { op: mlp, in: h3, out: f1, hidden: 4*D }
        - add2: { op: add, in: [h2, f1], out: x }
  outputs:
    logits:
      from: x
      head: { op: linear, dim: ${params.vocab_size} }
```

## 5. Core Concepts

### 5.1 Symbols
`symbols` declares named dimensions/constants.
- Numeric literal means fixed value.
- `null` means symbolic/unknown at compile-time, validated by constraints.

### 5.2 Tensors
Named tensor slots are introduced by:
- `inputs`
- Any node `out`
- Block outputs

Tensor metadata fields:
- `shape`
- `dtype`
- `device` (optional)

### 5.3 Nodes
Each graph item is a single-key mapping:
- Key is a stable node id (`embed`, `ln1`, etc.).
- Value is a node spec with required `op` and standardized `in`/`out` fields.

### 5.4 Blocks
`blocks` defines reusable subgraphs with explicit interfaces.
`use: <block_name>` instantiates a block.
`bindings` passes compile-time variables into block scope.

### 5.5 Repeat
`repeat` unrolls a subgraph with a loop variable:
- `var`: loop variable name
- `range`: integer or expression
- `body`: graph entries

Unrolling is deterministic and produces stable node names with index suffixes.

## 6. Operation Set (Current Runtime)
The current Synapse runtime/compiler supports these model-domain ops:
- `embedding`
- `linear`
- `layernorm`
- `rmsnorm`
- `activation`
- `softmax`
- `topk`
- `zeros_like`
- `add`
- `mul`
- `arange_positions`
- `split` (`parts` or `sizes`)
- `reshape_heads`
- `apply_rope_pair`
- `kv_seq_len`
- `repeat_kv`
- `merge_heads`
- `attention`
- `causal_mask`
- `moe_select_tokens`
- `moe_scatter_add`
- `index`
- `init_list`
- `append`
- `kv_cache_update`
- `coalesce`

### 6.1 Clarified Semantics (Stability-Critical)
- `kv_cache_update`:
  - operation contract is `(past, k, v) -> (k_all, v_all, present)`
  - no op kwargs; conditional execution is represented by lazy `select` nodes, not op kwargs
- `coalesce`:
  - generic grouped fallback, not KV-specific
  - if output arity is `n`, inputs are partitioned as strided groups:
    - `out_j = first non-None(in_j, in_{j+n}, in_{j+2n}, ...)`
  - input count must be divisible by output count
- `moe_select_tokens`:
  - generic sparse dispatch selector over router top-k assignments
  - returns selected hidden rows and routing metadata `(selected_hidden, token_idx, topk_pos, selected_scores)`
  - empty selections are valid and must produce empty tensors with consistent shapes/dtypes
- `moe_scatter_add`:
  - generic weighted sparse accumulation primitive
  - semantics: `accum[token_idx] += updates * scores`
  - repeated indices accumulate; empty indices are a no-op

Control/build graph constructs:
- `repeat` (control node, not an op module)
- `use` block invocation
- nested `graph` scopes
- lazy `select` value nodes for conditional expressions

Internal IR-only nodes (lowering artifacts, not public DSL surface):
- `_ir_alias`
- `_ir_const`

## 7. Expressions
Supported expression forms:
- Literal: `12`, `0.1`, `true`
- Symbol reference: `D`
- Parameter reference: `${params.vocab_size}`
- Loop variable reference: `${l}`
- Arithmetic over numeric symbols: `4*D`, `D/H`, `D + 64`

Expressions are compile-time only unless explicitly marked runtime.

## 8. Validation Rules
Compiler MUST validate:
1. `synapse` exists and equals `1`.
2. Node ids are unique within a scope.
3. Every `in` reference is defined before use (after repeat/block expansion).
4. Shape compatibility for op contracts.
5. `mha.dim % mha.heads == 0`.
6. `add`/`mul` inputs follow exact-shape or allowed broadcast rules.
7. `outputs.*.from` resolves to an existing tensor.

## 9. Deterministic Lowering to PyTorch
SYNAPSE/1 compiles into:
1. Module registry and constructor args.
2. `nn.Module` tree for blocks and repeated layers.
3. A generated `forward()` dataflow from graph order.
4. Optional named tensor map for debugging/inspection.

Lowering constraints:
- Graph order is execution order.
- Repeat unroll is static.
- All generated parameter names are stable and path-addressable.

### 9.1 Parameter Naming Convention
Compiler should emit parameter paths such as:
- `embed.weight`
- `layers.0.attn.q_proj.weight`
- `layers.11.mlp.fc2.bias`

This keeps compatibility with checkpoint tooling and `brainsurgery` tensor references.

## 10. Optional Lowering to BrainSurgery Transforms
For checkpoint migration or surgery workflows, a compiler may emit companion `brainsurgery` plans:
- `load` source checkpoint
- mapping transforms (`copy`, `assign`, `reshape`, `permute`, `concat`, `split`)
- `save` output checkpoint

This is optional and separate from runtime module generation.

## 11. Error Model
Errors must include:
- Code (e.g., `SYN-E-SHAPE-001`)
- Message
- YAML path (e.g., `model.graph[3].attn.heads`)
- Context snippet
- Suggested fix

## 12. Minimal Grammar (Structural)
This grammar describes structural constraints, not full scalar expression syntax.

```ebnf
document      = map ;
map           = "{" , { key_value } , "}" ;
key_value     = key , ":" , value ;
value         = scalar | list | map ;

synapse_doc   = "{" ,
                "synapse" ":" int(1) ","
                "model" ":" model_map ,
                "}" ;

model_map     = "{" ,
                "name" ":" string ","
                [ "dtype" ":" string "," ]
                [ "layout" ":" string "," ]
                [ "symbols" ":" map "," ]
                [ "params" ":" map "," ]
                [ "inputs" ":" map "," ]
                "graph" ":" graph_list ","
                [ "blocks" ":" blocks_map "," ]
                [ "outputs" ":" outputs_map ]
                "}" ;

graph_list    = "[" , { graph_item } , "]" ;
graph_item    = node | repeat | block_use ;
node          = "{" , node_id ":" node_spec , "}" ;
repeat        = "{" , "repeat" ":" repeat_spec , "}" ;
block_use     = "{" , block_id ":" use_spec , "}" ;
```

## 13. Example: Tiny MLP

```yaml
synapse: 1
model:
  name: TinyMLP
  symbols: { B: null, D: 256 }
  inputs:
    x: { shape: [B, D], dtype: float32 }
  graph:
    - fc1: { op: linear, in: x, out: h1, dim: 1024 }
    - act: { op: gelu, in: h1, out: h2 }
    - fc2: { op: linear, in: h2, out: y, dim: D }
  outputs:
    yhat: { from: y }
```

## 14. Compatibility and Versioning
- `SYNAPSE/1` is the initial stable schema.
- Future revisions must be opt-in via the `synapse` version field.
- Tooling may provide upgrade helpers (`SYNAPSE/1 -> SYNAPSE/2`) with explicit migration notes.

## 15. Code Generation
For decoder-only GPT-2 specs that provide the required `model.symbols` and `model.params` fields, BrainSurgery can emit standalone PyTorch model code:

```bash
brainsurgery synapse emit path/to/spec.yaml path/to/generated_model.py --class-name MyModel
```

The generated file includes:
- A `torch.nn.Module` model class with GPT-2 compatible tensor names.
- `from_state_dict(...)` constructor helper.
- `forward(input_ids)` returning logits with tied LM head weights.

PyTorch op mapping config template:
- [/Users/petersk/Nobackup/brainsurgery/brainsurgery/synapse/torch_op_map.yaml](/Users/petersk/Nobackup/brainsurgery/brainsurgery/synapse/torch_op_map.yaml)
