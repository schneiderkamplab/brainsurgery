# Flattening Suggestion (Current)

## Goal

Flatten Axon model code into a core form that is easy to typecheck and lower:

- explicit state flow
- explicit helper calls
- no ambient scope/path guessing

## Core Loop Form

Use explicit loop state with a yield expression:

```axon
x, new_kv <- for@h i <- [0..L) carry (x, new_kv) yield (gpt2_loop_step@h i x attn_mask past_kv new_kv)
```

Rules:

- `carry (...)`, `yield (...)`, and loop LHS must have matching arity/types.
- loop step logic should live in helper modules/functions for fully flat structure.

## Path Resolution Strategy

Prefer lexical path resolution and absolutize as early as possible.

- static scopes: resolve directly to absolute `@@...` paths at flattening time.
- iterative scopes (e.g. loop index `i`): resolve to **parametric absolute** path templates, e.g. `@@h.{i}.attn.c_attn.weight`.

This avoids runtime ambient-scope heuristics while still supporting loops without unrolling.

## Practical Lowering Model

1. Parse/import-load modules.
2. Desugar/flatten to core form (helpers + explicit carry/yield loops).
3. Resolve paths:
   - static -> `@@...`
   - loop-dependent -> `@@...{i}...` template tokens in IR
4. Lower/codegen substitutes loop indices where needed.

## Flattened gpt2 Example

```axon
{-# CHECKPOINTS "openai-community/gpt2" #-}
{-# TASK "causal_lm" #-}
import Activations (gelu_new)
import Cache
import Positions (position_ids)
import Attention (reshape_heads, merge_heads)
import Attention (attention_masked)
import Masking (causal_mask)
T = 1024
D = 768
L = 12
H = 12
gpt2_block_attn :: Tensor[B,S,D] -> ?Tensor[B,S] -> ?CacheLayer -> Int -> (Tensor[B,S,D], ?CacheLayer)
gpt2_block_attn x1 attn_mask past_kv i = do
  attn_proj <- NN.linear x1 dim=(3 * D) bias=true transpose=true weight_path=@@h.{i}.attn.c_attn.weight bias_path=@@h.{i}.attn.c_attn.bias
  q_lin, k_lin, v_lin <- Tensor.chunk attn_proj parts=3
  q <- reshape_heads q_lin heads=H
  k <- reshape_heads k_lin heads=H
  v <- reshape_heads v_lin heads=H
  k, v, new_kv <- Cache.update past_kv k v
  mask <- causal_mask q k window=T padding_mask=attn_mask
  attn_ctx <- attention_masked q k v mask=mask
  attn_merged <- merge_heads attn_ctx
  a <- NN.linear attn_merged dim=D bias=true transpose=true weight_path=@@h.{i}.attn.c_proj.weight bias_path=@@h.{i}.attn.c_proj.bias
  return a, new_kv
gpt2_block_mlp :: Tensor[B,S,D] -> Int -> Tensor[B,S,D]
gpt2_block_mlp x3 i = do
  m0 <- NN.linear x3 dim=(4 * D) bias=true transpose=true weight_path=@@h.{i}.mlp.c_fc.weight bias_path=@@h.{i}.mlp.c_fc.bias
  m1 <- gelu_new m0
  m <- NN.linear m1 dim=D bias=true transpose=true weight_path=@@h.{i}.mlp.c_proj.weight bias_path=@@h.{i}.mlp.c_proj.bias
  return m
gpt2_block :: Tensor[B,S,D] -> ?Tensor[B,S] -> ?CacheLayer -> Int -> (Tensor[B,S,D], ?CacheLayer)
gpt2_block x attn_mask past_kv i = do
  x1 <- NN.layernorm x eps=1e-5 weight_path=@@h.{i}.ln_1.weight bias=true bias_path=@@h.{i}.ln_1.bias
  a, new_kv <- gpt2_block_attn x1 attn_mask past_kv i
  x2 <- x + a
  x3 <- NN.layernorm x2 eps=1e-5 weight_path=@@h.{i}.ln_2.weight bias=true bias_path=@@h.{i}.ln_2.bias
  m <- gpt2_block_mlp x3 i
  y <- x2 + m
  return y, new_kv
gpt2_loop_step :: Int -> Tensor[B,S,D] -> ?Tensor[B,S] -> ?Cache -> ?Cache -> (Tensor[B,S,D], ?Cache)
gpt2_loop_step i x attn_mask past_kv new_kv = do
  past_i <- Cache.index past_kv i
  x, new_i <- gpt2_block x attn_mask past_i i
  new_kv <- Cache.append new_kv new_i
  return x, new_kv
gpt2 :: TokenIds[B,S] -> ?Tensor[B,S] -> ?Cache -> ?Bool -> (Tensor[B,S,V], ?Cache)
gpt2 input_ids attn_mask past_kv use_cache = do
  tok <- NN.embedding input_ids dim=D weight_path=@@wte.weight
  past_len <- Cache.past_length past_kv
  pos_ids <- position_ids input_ids attn_mask=attn_mask past_length=past_len pad_fill=1
  pos <- NN.embedding pos_ids dim=D weight_path=@@wpe.weight
  x <- tok + pos
  new_kv <- use_cache ? Cache.init : null
  x, new_kv <- for@h i <- [0..L) carry (x, new_kv) yield (gpt2_loop_step i x attn_mask past_kv new_kv)
  x_ln <- NN.layernorm x eps=1e-5 weight_path=@@ln_f.weight bias=true bias_path=@@ln_f.bias
  logits <- NN.linear x_ln dim=V weight_path=@@wte.weight bias=false transpose=false
  return logits, new_kv
```
