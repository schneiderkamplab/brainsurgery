# TinyGrad Backend for Synapse Codegen

**Date:** 2026-03-29
**Status:** Approved

## Motivation

- Hardware diversity: TinyGrad supports AMD, Qualcomm, Apple Silicon beyond NVIDIA/CUDA
- Performance: Lazy execution and minimal abstraction may offer benefits
- Architecture validation: Prove Synapse codegen supports multiple backends

## Approach: Pluggable Backend Modules

Backend-specific code lives in `synapse/backends/{pytorch,tinygrad}/`. The core graph representation and lowering remain backend-agnostic.

### Directory Structure

```
synapse/
  ops/                          # Backend-agnostic: metadata, lowering signatures, validation
    linear.py                   # OP_NAME, LOWERING_ARITY, uses_node_path, validate, infer_metadata
    attention.py
    add.py
    embedding.py
    ...
  backends/
    __init__.py                 # Backend registry: get_emitter("tinygrad")
    base.py                     # _BaseEmitter: graph walking, env, expressions
    pytorch/
      __init__.py
      emitter.py                # _PyTorchEmitter (extracted from current codegen.py)
      op_map.yaml               # Moved from torch_op_map.yaml
      ops/                      # PyTorch compile() + interpret() implementations
        __init__.py
        linear.py
        attention.py
        add.py
        embedding.py
        ...
    tinygrad/
      __init__.py
      emitter.py                # _TinyGradEmitter
      op_map.yaml               # TinyGrad API mappings
      ops/                      # TinyGrad compile() + interpret() implementations
        __init__.py
        linear.py
        attention.py
        embedding.py
        add.py
        layernorm.py
        rmsnorm.py
        ...
```

### Entry Point API

```python
# Existing (unchanged, defaults to pytorch)
code = emit_model_code_from_synapse_spec(spec)

# New: explicit backend
code = emit_model_code_from_synapse_spec(spec, backend="tinygrad")
```

### Backend Registry

```python
# synapse/backends/__init__.py
_backends: dict[str, type] = {"pytorch": PyTorchEmitter, "tinygrad": TinyGradEmitter}

def get_emitter(backend: str) -> type:
    return _backends[backend]
```

### Emitter Architecture

**_BaseEmitter** (shared):
- Graph walking: `_compile_graph()`, `_compile_block_call()`
- Environment management: `_fresh()`, `_assign_out_var()`, `_read_env_var()`
- Expression handling: `_expr_code()`, `_substitute_expr_names()`, `_try_eval_numeric()`
- Name mangling: `_py_name()`

**_PyTorchEmitter** (backend-specific):
- Imports: `torch`, `torch.nn`, `torch.nn.functional`
- Class template: `class X(nn.Module)` with `super().__init__()`
- `generate()`: uses `torch.inference_mode()`, `torch.argmax(dim=-1)`, `model.eval()/train()`
- State: `_state` dict + `_param(path)`, MXFP4 materialization
- Forward: `def forward(self, input_ids=None, **inputs)`

**_TinyGradEmitter** (backend-specific):
- Imports: `from tinygrad import Tensor, dtypes, TinyJit`
- Class template: plain class (no nn.Module base), `__call__` delegates to `forward`
- `generate()`: lazy by default (no inference_mode), `x.argmax(axis=-1)`, no eval/train
- State: same `_state` dict + `_param(path)` pattern (backend-agnostic)
- Forward: `def forward(self, input_ids=None, **inputs)` + `def __call__` wrapper
- Key API differences: `axis` vs `dim`, method-based API, `.cast()` vs `.to()`, `.realize()` for eager execution

### Op Compilation

Backend-agnostic metadata stays in `synapse/ops/`. Each backend provides its own `compile()` in `backends/{name}/ops/`.

**PyTorch op example (linear):**
```python
# Emits: F.linear(x.float(), w.float(), b.float()).to(dtype=x.dtype)
```

**TinyGrad op example (linear):**
```python
# Emits: x.linear(w.transpose(), b)
# Note: TinyGrad Linear transposes weight explicitly
```

**Key op mappings:**

| Op | PyTorch | TinyGrad |
|---|---|---|
| linear | `F.linear(x, w, b)` | `x.linear(w.transpose(), b)` |
| attention | `F.scaled_dot_product_attention(...)` | `q.scaled_dot_product_attention(k, v, ...)` |
| embedding | `F.embedding(idx, w)` | `w[idx]` |
| layernorm | `F.layer_norm(x, [d])` | `x.layernorm(axis=-1, eps=e)` |
| rmsnorm | custom computation | `x * (x.square().mean(-1) + eps).rsqrt()` |
| softmax | `F.softmax(x, dim=-1)` | `x.softmax(axis=-1)` |
| gelu | `F.gelu(x)` | `x.gelu()` |
| concat | `torch.cat([a, b], dim=0)` | `a.cat(b, dim=0)` |
| where | `torch.where(cond, a, b)` | `cond.where(a, b)` |
| add | `x + y` | `x + y` (same) |

Ops where PyTorch and TinyGrad emit identical code (add, multiply, reshape) can use the op map as a fallback without dedicated backend implementations.

### TinyGrad Op Map (tinygrad_op_map.yaml)

```yaml
version: 1
name: synapse_tinygrad_default
defaults:
  tensor_namespace: Tensor
  nn_namespace: tinygrad.nn
ops:
  linear:
    kind: tensor_method
    target: linear
  embedding:
    kind: nn_module
    target: tinygrad.nn.Embedding
  attention:
    kind: tensor_method
    target: scaled_dot_product_attention
  layernorm:
    kind: tensor_method
    target: layernorm
  softmax:
    kind: tensor_method
    target: softmax
```

### Scope

**In scope:**
- Backend registry and base emitter refactoring
- PyTorch backend extraction (move current code into backends/pytorch/)
- TinyGrad emitter with full class template
- TinyGrad op map
- All TinyGrad op compile() implementations
- TinyGrad generate() method

**Out of scope (future work):**
- TinyGrad runtime (SynapseProgramModel stays PyTorch-only for now)
- MXFP4 TinyGrad support
- Axon/lowering changes (none needed)
- TinyJit integration in generated code

### Files to Modify

- `synapse/codegen.py` — simplified to delegate to backend registry
- `synapse/ops/*.py` — remove `compile()` and `interpret()`, keep metadata only
- `synapse/__init__.py` — update exports

### Files to Create

- `synapse/backends/__init__.py`
- `synapse/backends/base.py`
- `synapse/backends/pytorch/` (full backend extraction)
- `synapse/backends/tinygrad/` (full TinyGrad backend)
