# TinyGrad Backend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add TinyGrad as a pluggable backend to the Synapse codegen system, enabling generation of TinyGrad Python model code from Synapse specs.

**Architecture:** Extract PyTorch-specific code from `codegen.py` and `ops/` into `backends/pytorch/`. Create a base emitter with shared graph-walking logic. Add `backends/tinygrad/` with a TinyGrad-specific emitter and op implementations. The entry point `emit_model_code_from_synapse_spec()` gains a `backend` parameter defaulting to `"pytorch"`.

**Tech Stack:** Python 3.11+, TinyGrad, OmegaConf

---

## Op Categorization

### Tier 1: No PyTorch dependency (shared across backends)
These ops emit plain Python or have no tensor API calls. They can live in the base emitter or be imported by both backends:
- `ir_alias` — simple assignment
- `ir_const` — literal assignment
- `list_append` — Python list ops
- `list_index` — Python indexing
- `list_init` — Python list creation
- `mul` — `a * b` (operator, no framework API)
- `cache_seq_len` — `.shape` access only

### Tier 2: Simple mapping (straightforward TinyGrad equivalent)
- `add` — `a + b` (same, just strip fp32 accum path)
- `concat` — `torch.cat` → `a.cat(b, dim=0)` (use `.cat()` method)
- `softmax` — `F.softmax(x, dim=d)` → `x.softmax(axis=d)`
- `clamp` — `torch.clamp(x, ...)` → `x.clip(min, max)`
- `zeros_like` — `torch.zeros_like(x)` → `Tensor.zeros_like(x)` or `x.zeros_like()`
- `repeat` — `.expand().reshape()` (same in TinyGrad)
- `split` — `torch.split/torch.chunk` → `.split()/.chunk()`
- `topk` — `torch.topk` → `x.topk()`

### Tier 3: Complex mapping (careful TinyGrad port)
- `linear` — `F.linear(x, w, b)` → `x.linear(w.transpose(), b)`
- `attention` — `F.scaled_dot_product_attention` → `q.scaled_dot_product_attention(k, v, ...)`
- `embedding` — `F.embedding(idx, w)` → `w[idx]`
- `layernorm` — `F.layer_norm(x, [d], w, b)` → `x.layernorm(axis=-1) * w + b`
- `rmsnorm` — manual computation (same math, different method names)
- `activation` — `F.gelu(x)` → `x.gelu()`, `F.silu(x)` → `x.silu()`, etc.
- `reshape_heads` — `.view().transpose()` → `.reshape().transpose()`
- `merge_heads` — `.transpose().contiguous().view()` → `.transpose().reshape()`
- `split_qkv_heads` — `torch.chunk` → `.chunk()`, `.permute()` → `.permute()`
- `rope_pair` — `torch.cos/sin/arange/cat` → TinyGrad equivalents
- `causal_mask` — `torch.arange/ones/where/triu` → TinyGrad equivalents
- `position_ids` — `torch.arange/cumsum/masked_fill` → TinyGrad equivalents
- `cache_update` — `torch.cat` → `.cat()`
- `linear_position_bias` — `torch.tensor/arange/pow/cat` → TinyGrad equivalents

### Tier 4: PyTorch-specific (stub for now, raise NotImplementedError)
- `mamba_scan` — sequential scan loop with complex PyTorch ops
- `causal_conv1d` — uses `F.conv1d`
- `moe_grouped_ffn` — uses `torch.histc`, `grouped_mm`, `index_add_`
- `moe_scatter_add` — uses `index_add_`
- `moe_select` — uses `nonzero` with specific semantics

---

## Task 1: Create backend registry and base emitter

**Files:**
- Create: `brainsurgery/synapse/backends/__init__.py`
- Create: `brainsurgery/synapse/backends/base.py`

**Step 1: Create backends package**

Create `brainsurgery/synapse/backends/__init__.py`:

```python
from __future__ import annotations

from typing import Any

_emitter_registry: dict[str, type] = {}


def register_backend(name: str, emitter_cls: type) -> None:
    _emitter_registry[name] = emitter_cls


def get_emitter_class(backend: str) -> type:
    cls = _emitter_registry.get(backend)
    if cls is None:
        raise ValueError(
            f"Unknown backend {backend!r}. Available: {sorted(_emitter_registry.keys())}"
        )
    return cls


def available_backends() -> list[str]:
    return sorted(_emitter_registry.keys())


# Auto-register backends when their packages are imported.
def _auto_register() -> None:
    from pkgutil import iter_modules

    package_path = globals().get("__path__", [])
    for module_info in iter_modules(package_path):
        if module_info.name.startswith("_"):
            continue
        try:
            __import__(f"{__name__}.{module_info.name}")
        except Exception:
            pass


_auto_register()

__all__ = ["register_backend", "get_emitter_class", "available_backends"]
```

**Step 2: Create base emitter**

Create `brainsurgery/synapse/backends/base.py` containing the shared logic extracted from `codegen.py:_Emitter`. This includes:

- `__init__(self, class_name, spec, symbols)` — store class_name, spec, model, blocks, symbols, _counter, _active_env
- `_fresh(self, base)` — unique variable naming
- `_py_name(self, value)` — sanitize to Python identifier
- `_assign_out_var(self, env, out_name)` — create or reuse output variable
- `_read_env_var(self, env, name)` — read from env with error
- `_infer_param_expr(self, node_spec, node_path_var, param_name)` — resolve param paths
- `_expr_code(self, expr, env)` — expression codegen
- `_substitute_expr_names(self, text, env)` — name substitution
- `_try_eval_numeric(self, text)` — safe numeric eval
- `_compile_graph(...)` — graph walking (core loop)
- `_compile_block_call(...)` — block invocation compilation
- `_node_output_names(self, node_spec)` — extract output names
- Abstract methods (to be overridden): `render()`, `_render_block_method()`, `_render_forward()`, `_render_generate()`, `_compile_op(...)`, `_op_uses_node_path(...)`

The base class leaves these as abstract (raise NotImplementedError):
```python
def render(self) -> str:
    raise NotImplementedError

def _render_block_method(self, block_name, block_spec) -> list[str]:
    raise NotImplementedError

def _render_forward(self) -> list[str]:
    raise NotImplementedError

def _render_generate(self) -> list[str]:
    raise NotImplementedError

def _compile_op(self, *, op, node_spec, env, node_path_var, scope_var, indent) -> list[str]:
    raise NotImplementedError

def _op_uses_node_path(self, op, node_spec) -> bool:
    raise NotImplementedError
```

**Step 3: Commit**

```bash
git add brainsurgery/synapse/backends/__init__.py brainsurgery/synapse/backends/base.py
git commit -m "feat: add backend registry and base emitter skeleton"
```

---

## Task 2: Extract PyTorch backend from current codegen.py

**Files:**
- Create: `brainsurgery/synapse/backends/pytorch/__init__.py`
- Create: `brainsurgery/synapse/backends/pytorch/emitter.py`
- Create: `brainsurgery/synapse/backends/pytorch/ops/__init__.py`
- Move: `brainsurgery/synapse/torch_op_map.yaml` → `brainsurgery/synapse/backends/pytorch/op_map.yaml`
- Modify: `brainsurgery/synapse/ops/*.py` — move `compile()` and `interpret()` to `backends/pytorch/ops/`
- Modify: `brainsurgery/synapse/codegen.py` — delegate to backend registry
- Modify: `brainsurgery/synapse/__init__.py` — keep exports working

**Step 1: Create PyTorch backend package**

Create `brainsurgery/synapse/backends/pytorch/__init__.py`:
```python
from .emitter import PyTorchEmitter

__all__ = ["PyTorchEmitter"]
```

**Step 2: Move torch_op_map.yaml**

```bash
git mv brainsurgery/synapse/torch_op_map.yaml brainsurgery/synapse/backends/pytorch/op_map.yaml
```

**Step 3: Create PyTorch emitter**

Create `brainsurgery/synapse/backends/pytorch/emitter.py` containing `_PyTorchEmitter` class that:
- Inherits from `BaseEmitter` (from `synapse.backends.base`)
- Moves the current `_Emitter.render()`, `_render_block_method()`, `_render_forward()`, `_render_generate()` methods
- Registers itself: `register_backend("pytorch", PyTorchEmitter)` at module level
- Implements `_compile_op()` and `_op_uses_node_path()` using the PyTorch op modules

The `load_synapse_torch_op_map()` function moves here too, updating the path to `backends/pytorch/op_map.yaml`.

**Step 4: Create PyTorch ops package**

Create `brainsurgery/synapse/backends/pytorch/ops/__init__.py` with auto-discovery (same pattern as current `synapse/ops/__init__.py` but requiring only `OP_NAME`, `compile`, `interpret`, `uses_node_path`).

For each of the 31 current op files, split them:
- Keep `synapse/ops/<name>.py` with: `OP_NAME`, metadata constants (`LOWERING_*`), `lowering_*` functions, `uses_node_path`
- Create `synapse/backends/pytorch/ops/<name>.py` with: `compile()` and `interpret()` functions (imported/moved from original)

For Tier 1 ops (no PyTorch dependency), the `compile()` can stay in the base op module and be re-exported by the PyTorch backend ops, OR the base emitter can handle them directly. Simpler approach: have the PyTorch ops re-export from the original location.

**Step 5: Update codegen.py**

Simplify `brainsurgery/synapse/codegen.py` to:
```python
from __future__ import annotations

from typing import Any

from .backends import get_emitter_class
from .ops import OP_MODULES, get_op_module


def load_synapse_torch_op_map() -> dict[str, Any]:
    from .backends.pytorch.emitter import load_synapse_torch_op_map as _load
    return _load()


def emit_model_code_from_synapse_spec(
    spec: dict[str, Any],
    *,
    class_name: str = "GeneratedSynapseModel",
    backend: str = "pytorch",
    op_map: dict[str, Any] | None = None,
) -> str:
    if not class_name.isidentifier():
        raise ValueError(f"Invalid class name: {class_name!r}")
    if spec.get("synapse") != 1:
        raise ValueError("Only synapse: 1 specs are supported")

    _validate_spec_ops(spec)

    model = spec.get("model")
    if not isinstance(model, dict):
        raise ValueError("spec.model must be a mapping")

    symbols_raw = model.get("symbols", {})
    symbols = {k: v for k, v in symbols_raw.items() if isinstance(v, (int, float, bool))}

    emitter_cls = get_emitter_class(backend)
    emitter = emitter_cls(
        class_name=class_name,
        spec=spec,
        symbols=symbols,
        op_map=op_map,
    )
    return emitter.render()


def _validate_spec_ops(spec: dict[str, Any]) -> None:
    # Keep current _validate_spec_ops logic unchanged
    # It references OP_MODULES from synapse/ops/ (backend-agnostic)
    ...
```

**Step 6: Update synapse/__init__.py**

The exports stay the same — `emit_model_code_from_synapse_spec` and `load_synapse_torch_op_map` are still importable from `synapse`. Add `available_backends` to exports.

**Step 7: Verify nothing breaks**

```bash
cd brainsurgery
python -c "from synapse import emit_model_code_from_synapse_spec; print('import OK')"
python -c "from synapse import load_synapse_torch_op_map; m = load_synapse_torch_op_map(); print(f'Loaded {len(m)} ops')"
python -c "from synapse.backends import available_backends; print(available_backends())"
```

Expected: All imports work, backends shows `["pytorch"]`.

**Step 8: Commit**

```bash
git add -A
git commit -m "refactor: extract PyTorch backend from codegen into backends/pytorch/"
```

---

## Task 3: Implement TinyGrad emitter

**Files:**
- Create: `brainsurgery/synapse/backends/tinygrad/__init__.py`
- Create: `brainsurgery/synapse/backends/tinygrad/emitter.py`

**Step 1: Create TinyGrad backend package**

Create `brainsurgery/synapse/backends/tinygrad/__init__.py`:
```python
from .emitter import TinyGradEmitter

__all__ = ["TinyGradEmitter"]
```

**Step 2: Create TinyGrad emitter**

Create `brainsurgery/synapse/backends/tinygrad/emitter.py` with `_TinyGradEmitter` class that:
- Inherits from `BaseEmitter`
- Registers itself: `register_backend("tinygrad", TinyGradEmitter)`

**render() method** generates:
```python
from __future__ import annotations

from typing import Any

import math
from tinygrad import Tensor, dtypes

class {class_name}:
    def __init__(self, state_dict: dict[str, Any] | None = None) -> None:
        self._state: dict[str, Any] = {}
        self._symbols: dict[str, int | float | bool] = {symbols}
        if state_dict is not None:
            self.load_state_dict_tensors(state_dict)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @classmethod
    def from_state_dict(cls, state_dict):
        return cls(state_dict=state_dict)

    def load_state_dict_tensors(self, state_dict):
        self._state = dict(state_dict)

    def _param(self, path):
        return self._state[path]

    # ... same helpers: _join_scope, _scope_of, _safe_get, _first_tensor
    # ... same helpers: _prepare_env, _for_values, _reset_trace, _trace_op
    # ... but using Tensor instead of torch.Tensor in type annotations
```

Key differences from PyTorch emitter:
- No `nn.Module` base class
- No `super().__init__()`
- No `materialize_mxfp4_aliases` call (MXFP4 is out of scope)
- `__call__` delegates to `forward`
- Type annotations use `Tensor` instead of `torch.Tensor`
- `_trace_op` uses `.realize().numpy()` pattern instead of `.detach().float().cpu()`

**_render_generate() method** generates:
```python
def generate(self, input_ids, *, eos_token_id, max_len, attention_mask=None, attn_mask=None):
    # Same structure but:
    # - No torch.inference_mode()
    # - Tensor.zeros instead of tensor.new_empty/new_zeros
    # - argmax(axis=-1) instead of argmax(dim=-1)
    # - No self.eval()/self.train()
    # - Use Tensor.zeros for finished tracking
```

**Step 3: Verify TinyGrad backend registers**

```bash
python -c "from synapse.backends import available_backends; print(available_backends())"
```

Expected: `["pytorch", "tinygrad"]`

**Step 4: Commit**

```bash
git add brainsurgery/synapse/backends/tinygrad/
git commit -m "feat: add TinyGrad emitter skeleton"
```

---

## Task 4: Create TinyGrad op map

**Files:**
- Create: `brainsurgery/synapse/backends/tinygrad/op_map.yaml`

**Step 1: Create op map**

```yaml
version: 1
name: synapse_tinygrad_default
defaults:
  tensor_class: Tensor
  nn_namespace: tinygrad.nn
ops:
  linear:
    kind: tensor_method
    target: linear
  embedding:
    kind: indexing
    target: weight[input]
  attention:
    kind: tensor_method
    target: scaled_dot_product_attention
  layernorm:
    kind: tensor_method
    target: layernorm
  softmax:
    kind: tensor_method
    target: softmax
  gelu:
    kind: tensor_method
    target: gelu
  silu:
    kind: tensor_method
    target: silu
  relu:
    kind: tensor_method
    target: relu
  sigmoid:
    kind: tensor_method
    target: sigmoid
  tanh:
    kind: tensor_method
    target: tanh
  cat:
    kind: tensor_method
    target: cat
  clamp:
    kind: tensor_method
    target: clip
  zeros_like:
    kind: tensor_method
    target: zeros_like
  topk:
    kind: tensor_method
    target: topk
  where:
    kind: tensor_method
    target: where
```

**Step 2: Commit**

```bash
git add brainsurgery/synapse/backends/tinygrad/op_map.yaml
git commit -m "feat: add TinyGrad op map"
```

---

## Task 5: Implement Tier 1 ops (no PyTorch dependency)

These ops have no framework-specific code. They can be implemented once in the base emitter or as shared TinyGrad ops that simply delegate to the existing logic.

**Files:**
- Create: `brainsurgery/synapse/backends/tinygrad/ops/__init__.py`
- Create: `brainsurgery/synapse/backends/tinygrad/ops/ir_alias.py`
- Create: `brainsurgery/synapse/backends/tinygrad/ops/ir_const.py`
- Create: `brainsurgery/synapse/backends/tinygrad/ops/list_append.py`
- Create: `brainsurgery/synapse/backends/tinygrad/ops/list_index.py`
- Create: `brainsurgery/synapse/backends/tinygrad/ops/list_init.py`
- Create: `brainsurgery/synapse/backends/tinygrad/ops/mul.py`
- Create: `brainsurgery/synapse/backends/tinygrad/ops/cache_seq_len.py`

**Step 1: Create TinyGrad ops package with auto-discovery**

Create `brainsurgery/synapse/backends/tinygrad/ops/__init__.py` — same auto-discovery pattern as `synapse/ops/__init__.py`, requiring `OP_NAME`, `compile`, `interpret`, `uses_node_path`.

**Step 2: Implement each Tier 1 op**

For each op, the TinyGrad `compile()` function is identical to the PyTorch version since they use no framework APIs:

- `ir_alias`: `out_var = source_expr` (same)
- `ir_const`: `out_var = value_code` (same)
- `list_append`: Python list operations (same)
- `list_index`: Python indexing (same)
- `list_init`: `out_var = []` (same)
- `mul`: `a * b` (same)
- `cache_seq_len`: `.shape` access (same)

Copy the `compile()` functions from the PyTorch ops. For `interpret()`, create stub implementations that raise `NotImplementedError("TinyGrad interpret not yet implemented")` since TinyGrad runtime is out of scope.

**Step 3: Verify**

```bash
python -c "
from synapse.backends.tinygrad.ops import OP_MODULES
print(f'Loaded {len(OP_MODULES)} TinyGrad ops')
for name in sorted(OP_MODULES.keys()):
    print(f'  {name}')
"
```

Expected: 7 ops loaded.

**Step 4: Commit**

```bash
git add brainsurgery/synapse/backends/tinygrad/ops/
git commit -m "feat: add Tier 1 TinyGrad ops (no framework dependency)"
```

---

## Task 6: Implement Tier 2 ops (simple mapping)

**Files:**
- Create: `brainsurgery/synapse/backends/tinygrad/ops/add.py`
- Create: `brainsurgery/synapse/backends/tinygrad/ops/concat.py`
- Create: `brainsurgery/synapse/backends/tinygrad/ops/softmax.py`
- Create: `brainsurgery/synapse/backends/tinygrad/ops/clamp.py`
- Create: `brainsurgery/synapse/backends/tinygrad/ops/zeros_like.py`
- Create: `brainsurgery/synapse/backends/tinygrad/ops/repeat.py`
- Create: `brainsurgery/synapse/backends/tinygrad/ops/split.py`
- Create: `brainsurgery/synapse/backends/tinygrad/ops/topk.py`

**Step 1: Implement each Tier 2 op**

Key mappings:

**add.py**: Strip `_hf_align_add_fp32_accum` path. Just `out_var = a + b`.

**concat.py**: `torch.cat([x, y], dim=d)` → `x.cat(y, dim=d)`:
```python
return [f"{indent}{out} = {x}.cat({y}, dim=int({dim_expr}))"]
```

**softmax.py**: `F.softmax(x, dim=d)` → `x.softmax(axis=d)`:
```python
lines.append(f"{indent}{out_var} = {src}.softmax(axis=int({dim}))")
```
For dtype kwarg: `x.softmax(axis=d, dtype=dtypes.float32)`.

**clamp.py**: `torch.clamp(x, min, max)` → `x.clip(min, max)`:
```python
lines.append(f"{indent}{out_var} = {src}.clip(min=({min_code}), max=({max_code}))")
```

**zeros_like.py**: `torch.zeros_like(x)` → `x.zeros_like()` or `Tensor.zeros(*x.shape)`:
```python
lines.append(f"{indent}{out_var} = {src}.zeros_like()")
```

**repeat.py**: Same `.expand().reshape()` pattern works in TinyGrad (operator overloading).

**split.py**: `torch.split(x, sizes, dim)` → `x.split(sizes, dim)`; `torch.chunk(x, n, dim)` → `x.chunk(n, dim)`. Use `axis` instead of `dim`:
```python
lines.append(f"{indent}{tmp} = {src}.split([{sizes_code}], axis=-1)")
```

**topk.py**: `torch.topk(x, k, dim=d)` → `x.topk(k, dim)`:
```python
lines.append(f"{indent}{values_var}, {indices_var} = {src}.topk(int({k}), dim=int({dim}))")
```

**Step 2: Verify**

```bash
python -c "
from synapse.backends.tinygrad.ops import OP_MODULES
print(f'Total TinyGrad ops: {len(OP_MODULES)}')
for name in ['add','concat','softmax','clamp','zeros_like','repeat','split','topk']:
    assert name in OP_MODULES, f'Missing {name}'
print('All Tier 2 ops loaded')
"
```

**Step 3: Commit**

```bash
git add brainsurgery/synapse/backends/tinygrad/ops/
git commit -m "feat: add Tier 2 TinyGrad ops (simple API mapping)"
```

---

## Task 7: Implement Tier 3 ops - Part 1 (core ops)

**Files:**
- Create: `brainsurgery/synapse/backends/tinygrad/ops/linear.py`
- Create: `brainsurgery/synapse/backends/tinygrad/ops/embedding.py`
- Create: `brainsurgery/synapse/backends/tinygrad/ops/activation.py`
- Create: `brainsurgery/synapse/backends/tinygrad/ops/layernorm.py`
- Create: `brainsurgery/synapse/backends/tinygrad/ops/rmsnorm.py`

**Step 1: Implement linear.py**

TinyGrad linear uses `x.linear(w.transpose(), b)` instead of `F.linear(x, w, b)`. Strip `_hf_align_linear_fp32_accum` path. Keep expert selection logic (it's just indexing, works the same).

```python
# Key generated code pattern:
# out = x.linear(w.transpose(), b)
# For transpose node_spec: out = x.dot(w) or x.matmul(w)
# For expert: w = w[int(expert_idx)]
```

Handle the transpose flag: in PyTorch, `F.linear` always does `x @ W.T`, but in TinyGrad `x.linear(w_t, b)` expects already-transposed weight. So:
- Non-transpose (default): `x.linear(w.transpose(), b)`
- Transpose: `x.linear(w, b)` or `x.matmul(w)`

Also strip dtype alignment code (TinyGrad handles dtypes differently).

**Step 2: Implement embedding.py**

TinyGrad: `weight[input_ids]` (simple indexing).

```python
lines.append(f"{indent}{out_var} = emitter._param({weight_expr})[{src}]")
```

**Step 3: Implement activation.py**

Map activations to TinyGrad tensor methods:
- `gelu` → `x.gelu()`
- `relu` → `x.relu()`
- `silu` → `x.silu()`
- `sigmoid` → `x.sigmoid()`
- `gelu_new/gelu_pytorch_tanh` → manual tanh approximation (same math)
- `swiglu` → `x.silu() * x`

Strip `_hf_align` fp32 accumulation paths.

**Step 4: Implement layernorm.py**

TinyGrad has `x.layernorm(axis=-1, eps=e)` which computes the normalized output. But it doesn't apply weight/bias. So:
```python
# Generated code:
normed = x.layernorm(axis=-1, eps=float(eps))
out = normed * weight + bias
```

Strip `_hf_align_norm_fp32` path.

**Step 5: Implement rmsnorm.py**

RMS norm in TinyGrad:
```python
# Manual computation:
norm = x * (x.square().mean(axis=-1, keepdim=True) + eps).rsqrt()
out = norm * weight
# For unit_offset: out = norm * (1.0 + weight)
```

**Step 6: Commit**

```bash
git add brainsurgery/synapse/backends/tinygrad/ops/
git commit -m "feat: add core TinyGrad ops (linear, embedding, activation, layernorm, rmsnorm)"
```

---

## Task 8: Implement Tier 3 ops - Part 2 (attention and heads)

**Files:**
- Create: `brainsurgery/synapse/backends/tinygrad/ops/attention.py`
- Create: `brainsurgery/synapse/backends/tinygrad/ops/reshape_heads.py`
- Create: `brainsurgery/synapse/backends/tinygrad/ops/merge_heads.py`
- Create: `brainsurgery/synapse/backends/tinygrad/ops/split_qkv_heads.py`

**Step 1: Implement attention.py**

TinyGrad has `q.scaled_dot_product_attention(k, v, attn_mask=None, is_causal=False, scale=None)`.

Map the three attention paths:
- SDPA path: `q.scaled_dot_product_attention(k, v, attn_mask=mask, is_causal=is_causal, scale=scale)`
- Eager path: manual matmul + softmax
  ```python
  scores = q.matmul(k.transpose(-2, -1)) * scale
  # apply mask
  probs = scores.softmax(axis=-1)
  out = probs.matmul(v)
  ```
- Sink path: similar to PyTorch but with TinyGrad methods

Strip `_hf_align_mask_contract` and `_hf_align_attention_eager` paths (use SDPA only in TinyGrad).

**Step 2: Implement reshape_heads.py**

Same logic but use `.reshape()` instead of `.view()` (TinyGrad has no `.view()`):
```python
out = src.reshape(bsz, seq_len, heads, head_dim).transpose(1, 2)
```

**Step 3: Implement merge_heads.py**

```python
out = src.transpose(1, 2).reshape(bsz, seq_len, heads * head_dim)
```
Note: TinyGrad doesn't need `.contiguous()` since it's lazy.

**Step 4: Implement split_qkv_heads.py**

Replace `torch.chunk` with `.chunk()`:
```python
q_lin, k_lin, v_lin = src.chunk(3, dim=-1)
q = q_lin.reshape(bsz, seq, heads, hd).permute(0, 2, 1, 3)
```

**Step 5: Commit**

```bash
git add brainsurgery/synapse/backends/tinygrad/ops/
git commit -m "feat: add TinyGrad attention and head ops"
```

---

## Task 9: Implement Tier 3 ops - Part 3 (positional and cache)

**Files:**
- Create: `brainsurgery/synapse/backends/tinygrad/ops/rope_pair.py`
- Create: `brainsurgery/synapse/backends/tinygrad/ops/causal_mask.py`
- Create: `brainsurgery/synapse/backends/tinygrad/ops/position_ids.py`
- Create: `brainsurgery/synapse/backends/tinygrad/ops/cache_update.py`
- Create: `brainsurgery/synapse/backends/tinygrad/ops/linear_position_bias.py`

**Step 1: Implement rope_pair.py**

Replace PyTorch ops with TinyGrad equivalents:
- `torch.arange(0, n, ...)` → `Tensor.arange(n, ...)`
- `torch.cos(x)` → `x.cos()`
- `torch.sin(x)` → `x.sin()`
- `torch.cat([a, b], dim=-1)` → `a.cat(b, dim=-1)`
- `torch.where(cond, a, b)` → `cond.where(a, b)`
- `x.to(device=..., dtype=...)` → `x.cast(dtype)` (TinyGrad handles device separately)

**Step 2: Implement causal_mask.py**

Replace:
- `torch.arange(n)` → `Tensor.arange(n)`
- `torch.ones((r, c), ...)` → `Tensor.ones(r, c, ...)`
- `torch.where(cond, a, b)` → `cond.where(a, b)`
- `torch.finfo(dtype).min` → use a literal large negative value or `dtypes.min(...)`
- `.tril()` → `.tril()`

Note: Causal mask caching can be simplified for TinyGrad (or removed initially).

**Step 3: Implement position_ids.py**

Replace:
- `torch.arange(start, end, ...)` → `Tensor.arange(end - start) + start` (TinyGrad arange starts at 0)
- `.cumsum(dim=-1)` → `.cumsum(axis=-1)`
- `.masked_fill(mask, val)` → `mask.where(val, x)` (inverted logic)

**Step 4: Implement cache_update.py**

Replace `torch.cat` → `.cat()`:
```python
k_ctx = past[0].cat(k_new, dim=-2)
v_ctx = past[1].cat(v_new, dim=-2)
```

**Step 5: Implement linear_position_bias.py**

Replace:
- `torch.tensor(val)` → `Tensor(val)` or `Tensor.full(..., val)`
- `torch.arange(...)` → `Tensor.arange(...)`
- `torch.pow(base, exp)` → `base.pow(exp)` or `base ** exp`
- `torch.cat([a, b])` → `a.cat(b)`
- `.view()` → `.reshape()`

**Step 6: Commit**

```bash
git add brainsurgery/synapse/backends/tinygrad/ops/
git commit -m "feat: add TinyGrad positional and cache ops"
```

---

## Task 10: Implement Tier 4 ops (stubs)

**Files:**
- Create: `brainsurgery/synapse/backends/tinygrad/ops/mamba_scan.py`
- Create: `brainsurgery/synapse/backends/tinygrad/ops/causal_conv1d.py`
- Create: `brainsurgery/synapse/backends/tinygrad/ops/moe_grouped_ffn.py`
- Create: `brainsurgery/synapse/backends/tinygrad/ops/moe_scatter_add.py`
- Create: `brainsurgery/synapse/backends/tinygrad/ops/moe_select.py`

**Step 1: Create stub implementations**

Each stub has:
```python
OP_NAME = "mamba_scan"  # etc.

def compile(emitter, node_spec, env, *, node_path_var, scope_var, indent):
    raise NotImplementedError(f"TinyGrad backend does not yet support op '{OP_NAME}'")

def interpret(model, node_spec, env, *, node_path, scope, symbols):
    raise NotImplementedError(f"TinyGrad backend does not yet support op '{OP_NAME}'")

def uses_node_path(emitter, node_spec):
    return True
```

**Step 2: Commit**

```bash
git add brainsurgery/synapse/backends/tinygrad/ops/
git commit -m "feat: add Tier 4 TinyGrad op stubs (not yet implemented)"
```

---

## Task 11: Wire TinyGrad emitter to TinyGrad ops

**Files:**
- Modify: `brainsurgery/synapse/backends/tinygrad/emitter.py`

**Step 1: Implement _compile_op and _op_uses_node_path**

In `TinyGradEmitter`, add:

```python
from .ops import OP_MODULES as TINYGRAD_OP_MODULES, get_op_module as get_tinygrad_op

def _compile_op(self, *, op, node_spec, env, node_path_var, scope_var, indent):
    op_module = get_tinygrad_op(op)
    if op_module is None:
        raise NotImplementedError(f"Unsupported op in TinyGrad codegen: {op!r}")
    prev_env = self._active_env
    self._active_env = env
    try:
        return op_module.compile(
            self, node_spec, env,
            node_path_var=node_path_var,
            scope_var=scope_var,
            indent=indent,
        )
    finally:
        self._active_env = prev_env

def _op_uses_node_path(self, op, node_spec):
    op_module = get_tinygrad_op(op)
    if op_module is None:
        raise NotImplementedError(f"Unsupported op: {op!r}")
    return bool(op_module.uses_node_path(self, node_spec))
```

**Step 2: Verify end-to-end codegen**

Create a minimal test spec and verify TinyGrad code generation:

```python
spec = {
    "synapse": 1,
    "model": {
        "symbols": {"hidden_size": 64, "vocab_size": 1000},
        "inputs": {"input_ids": {"optional": False}},
        "outputs": {"logits": "hidden"},
        "graph": [
            {"embed": {"_op": "embedding", "_args": "input_ids", "_bind": "hidden"}},
        ],
    },
}

from synapse import emit_model_code_from_synapse_spec
code = emit_model_code_from_synapse_spec(spec, backend="tinygrad")
print(code)
assert "from tinygrad import Tensor" in code
assert "class GeneratedSynapseModel:" in code
```

**Step 3: Commit**

```bash
git add brainsurgery/synapse/backends/tinygrad/emitter.py
git commit -m "feat: wire TinyGrad emitter to TinyGrad ops"
```

---

## Task 12: End-to-end validation

**Files:**
- Modify: `brainsurgery/synapse/axon_test.py` (add backend parameter)
- Modify: `brainsurgery/synapse/op_parity.py` (add backend parameter)

**Step 1: Update axon_test.py to support backend selection**

Add an optional `backend` parameter that gets passed through to `emit_model_code_from_synapse_spec()`.

**Step 2: Update op_parity.py similarly**

**Step 3: Run parity tests for PyTorch backend**

Verify the PyTorch backend still works after the refactoring:

```bash
python -c "
from synapse import emit_model_code_from_synapse_spec
# Use an existing spec file to test
# code = emit_model_code_from_synapse_spec(spec)
# print('PyTorch codegen works')
"
```

**Step 4: Test TinyGrad codegen with a real model spec**

Pick a simple model (e.g., a small GPT-2 config) and generate TinyGrad code:
```bash
python -c "
from synapse import emit_model_code_from_synapse_spec
# Load a spec, generate TinyGrad code
# code = emit_model_code_from_synapse_spec(spec, backend='tinygrad')
# Verify the code is syntactically valid Python
compile(code, '<generated>', 'exec')
print('TinyGrad codegen produces valid Python')
"
```

**Step 5: Final commit**

```bash
git add -A
git commit -m "feat: complete TinyGrad backend with end-to-end validation"
```

---

## Summary of All Tasks

| Task | Description | Files |
|------|-------------|-------|
| 1 | Backend registry + base emitter | 2 new |
| 2 | Extract PyTorch backend | ~35 files (move/split) |
| 3 | TinyGrad emitter | 2 new |
| 4 | TinyGrad op map | 1 new |
| 5 | Tier 1 ops (no dependency) | 8 new |
| 6 | Tier 2 ops (simple mapping) | 8 new |
| 7 | Tier 3 ops - core | 5 new |
| 8 | Tier 3 ops - attention/heads | 4 new |
| 9 | Tier 3 ops - positional/cache | 5 new |
| 10 | Tier 4 ops (stubs) | 5 new |
| 11 | Wire emitter to ops | 1 modify |
| 12 | End-to-end validation | 2 modify |

Total: ~40 new files, ~3 modified files.
