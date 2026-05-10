# Axon DSL (Draft v1)

Axon is an indentation-sensitive, expression-oriented DSL for model graphs.

Core style:

- no mandatory curly braces,
- `do`-style headers with indentation blocks,
- single-assignment bindings with `<-`,
- module-path call sugar via `op@path(...)`,
- namespaced ops via `ns::op(...)`,
- namespaced module calls via `Namespace.module(...)`,
- expression-level conditional via `if cond then a else b`,
- ternary shorthand `cond ? a : b` (equivalent to `if/then/else`),
- explicit return via `return ...`,
- mixed composition styles: nested call, forward pipe `|>`, and bind `>>=`.

## 1) Syntax Principles

1. Whitespace is structural (Python-like blocks).
2. Every bound name is assigned once in a scope.
3. Module hierarchy drives parameter-path inference.
4. Tensor programs should read like linear algebra pipelines.

## 2) Binding and Calls

- Binding: `x1 <- expr`
- Multi-bind: `q, k, v <- expr1, expr2, expr3`
- Module-scoped op: `linear@attn.c_attn(x)`
- Namespaced op: `cache::update(past, k, v)`
- Namespaced module call: `Act.swiglu(x)` (requires `import Act`)
- Conditional: `if use_cache then a else b`
- Ternary shorthand: `use_cache ? a : b`
- Header:
  - `name :: T1 -> ?T2 -> (T3, ?T4)`
  - `name a b = do`
- Composition (equivalent intent):
  - `y <- g(f(x))`
  - `y <- x |> f |> g`
  - `y <- f(x) >>= \\z -> g(z)`

## 3) Scopes and Parameters

Paths define parameter ownership:

- `linear@mlp.c_fc(x)` implies parameters at `...mlp.c_fc.weight/bias`.
- Inside repeats/modules, lexical scope prefixes are appended automatically.

## 3.1 Imports and Namespaces

- Top-level imports: `import Act`
- File imports are resolved relative to the current `.axon` file:
  - `import Lib` -> `./Lib.axon`
  - `import foo.bar` -> `./foo/bar.axon`
- Imported namespace calls use dotted module names:
  - `Act.swiglu(x)`
  - `Act.gelu_tanh(x)`
- A namespaced module call must have a matching `import <Namespace>` in source.
- Import resolution order:
  - local file first (relative to importing file),
  - then builtins directory `brainsurgery/synapse/builtins/`.
- `Act` is shipped as built-in file `brainsurgery/synapse/builtins/Act.axon`.

Multi-path module parameters are supported via repeated `@` in module names/calls:

- definition: `expert_ffn@gate@up@down :: @Path -> @Path -> @Path -> Tensor -> Tensor`
- call: `expert_ffn@mlp.gate_proj@mlp.up_proj@mlp.down_proj(x)`

## 4) Symbols and Types

Top-level constants define symbols used by module signatures and args:
`D = 768`, `L = 12`, `EPS = 1e-05`.

## 5) Core Forms

## 5.1 Tensor forms

- `embed@path(ids)`
- `linear@path(x)`
- `layernorm@path(x, eps=...)`
- `rmsnorm@path(x, eps=..., cast_float=..., unit_offset=...)`
- `act::gelu_new(x)` (and other activations)
- `attention(q, k, v, causal=true, mask=?, scale=?)`
- `reshape_heads(x, heads=H, head_dim=Hd)`
- `merge_heads(x)`
- `topk(x, k=K)`

## 5.2 Control/graph forms

- `x <- expr`
- `return expr`
- `for@path i <- [0..N) do` blocks
- conditional `if c then t else f`
- ternary shorthand `c ? t : f`

## 5.3 KV-cache helpers

- `cache::update(past, k, v)`
- `cache::seq_len(past)`
- `cache::coalesce(k_all, v_all, k, v)`
- Builtin module wrappers (recommended in authored Axon):
  - `Cache.update(past, k, v)`
  - `Cache.seq_len(past)`
  - `Cache.coalesce(k_all, v_all, k, v)`
  - `Cache.init_list()`
  - `Cache.index(cache, i)`
  - `Cache.append(cache, present_i)`

Semantics notes:
- `cache::update` has no op-level kwargs.
- `cache::coalesce` is grouped first-non-`None` fallback over strided candidate groups (generic, not KV-specific).

KV remains optional at spec level; codegen/runtime may use hints to optimize decoding.

### 5.3.1 Cache Patterns (Recommended)

Use `import Cache` and prefer `Cache.*` wrappers for readable, reusable cache flow:

```axon
import Cache

decoder :: Tensor[B,S,D] -> ?Cache -> ?Bool -> (Tensor[B,S,D], ?Cache)
decoder x past_kv use_cache = do
  q, k, v <- ...
  k_all, v_all, present <- if use_cache then Cache.update past_kv k v else k, v, null
  k_ctx, v_ctx <- Cache.coalesce k_all v_all k v
  y <- attention q k_ctx v_ctx ...
  return y, present

model :: TokenIds[B,S] -> ?Cache -> ?Bool -> (Tensor[B,S,V], ?Cache)
model input_ids past_key_values use_cache = do
  x <- ...
  present_key_values <- Cache.init_list()
  for@layers i <- [0..L) do
    past_i <- Cache.index past_key_values i
    x, present_i <- decoder x past_i use_cache
    present_key_values <- use_cache ? Cache.append present_key_values present_i : present_key_values
  return x, present_key_values
```

Notes:
- Conditional append is expressed directly as
  `use_cache ? Cache.append(cache, present_i) : cache`.
- `Cache.init_list()` must be called with `()`; without it, Axon treats it as a value expression.

## 5.4 MoE helpers (composable)

- `linear@...` + activation + `mul` + `linear@...` for expert FFN (compositional, no dedicated expert op)
- `moe_select_tokens(hidden, topk_scores, topk_idx, expert=e)`
- `moe_scatter_add(accum, token_idx, updates, weights)`

Semantics notes:
- `moe_select_tokens` is a generic sparse dispatch primitive; empty selections are valid.
- `moe_scatter_add` is a generic weighted sparse accumulation primitive; empty indices are a no-op.

## 6) Parallelism Hints

Hints are annotations, not semantics changes:

```axon
@tp(axis="hidden", parts=2)
@pp(stage=1)
@cp(axis="seq", parts=2)
h_attn :: Tensor[B,S,D] -> Tensor[B,S,D]
h_attn x = do
```

These are preserved for lowering/planning (TP/PP/CP), ignored by pure interpretation.

## 7) Example (Refined GPT-2 Block)

```axon
gpt2_block :: Tensor[B,S,D] -> ?Cache -> ?Bool -> (Tensor[B,S,D], ?Cache)
gpt2_block x past_kv use_cache = do
  x1 <- layernorm@ln_1(x, eps=1e-5)
  qkv <- linear@attn.c_attn(x1)
  q_lin, k_lin, v_lin <- split(qkv)
  q, k, v <- reshape_heads(q_lin, heads=H), reshape_heads(k_lin, heads=H), reshape_heads(v_lin, heads=H)
  k_all, v_all, present_kv <- if use_cache then cache::update(past_kv, k, v) else (k, v, null)
  k_ctx, v_ctx <- cache::coalesce(k_all, v_all, k, v)
  a_heads <- attention(q, k_ctx, v_ctx, causal=true)
  x2 <- x + linear@attn.c_proj(merge_heads(a_heads))
  x3 <- layernorm@ln_2(x2, eps=1e-5)
  m1 <- linear@mlp.c_fc(x3)
  m2 <- act::gelu_new(m1)
  y <- x2 + linear@mlp.c_proj(m2)
  return y, present_kv
```

## 8) Lowering to Graph IR

Lowering is mechanical:

- `<-` bindings map to graph node outputs,
- primitive calls map to typed Graph IR nodes,
- path operands remain structured `Path` values, including absolute templates,
- flattened loop helpers remain ordinary graph-call nodes,
- ternary shorthand maps to typed data-flow, not control-flow blocks,
- annotations map to planner metadata.

Graph IR is the canonical machine-readable lowering target. Axon remains the readable
authoring/rendering format and the source of stage roundtrip tests.
