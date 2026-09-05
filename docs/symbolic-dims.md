# Symbolic Dims In Value Expressions

This file tracks where Axon uses type-signature dimension symbols (`B`, `S`, `T`, `K`, `HD`, etc.) in value-level expressions (not only in type signatures).

Scope: `brainsurgery/synapse/builtins/*.axon` and `brainsurgery/synapse/models/*/*.axon`.

## Pure Symbolic-Dim Occurrences

Pure = expression uses only symbolic dims and literals (no runtime-derived scalar like `rot`, `past_length`, `window`, etc.).

| File | Line | Expression | Notes |
|---|---:|---|---|
| `builtins/Cache.axon` | 28 | `return T` | Cache sequence length read directly from type alias dim. |
| `builtins/Masking.axon` | 9 | `Tensor.arange q start=0 end=Q` | Sequence-length bound from signature dim. |
| `builtins/Masking.axon` | 10 | `Tensor.arange k start=0 end=K` | Key-length bound from signature dim. |
| `builtins/Masking.axon` | 11 | `aligned <- row_ids + (K - Q)` | Symbolic alignment arithmetic. |
| `builtins/Masking.axon` | 14 | `Tensor.reshape ... shape=[1, 1, Q, K]` | Symbolic mask shape. |
| `builtins/Masking.axon` | 15 | `Tensor.expand ... shape=[B, 1, Q, K]` | Symbolic broadcast shape. |
| `builtins/Masking.axon` | 21 | `Tensor.reshape ... shape=[B, 1, 1, K]` | Symbolic pad mask shape. |
| `builtins/Masking.axon` | 35 | `Tensor.arange q start=0 end=Q` | Same as above (bidirectional path). |
| `builtins/Masking.axon` | 36 | `Tensor.arange k start=0 end=K` | Same as above (bidirectional path). |
| `builtins/Masking.axon` | 37 | `aligned <- row_ids + (K - Q)` | Same alignment pattern. |
| `builtins/Masking.axon` | 41 | `Tensor.reshape ... shape=[1, 1, Q, K]` | Symbolic mask shape. |
| `builtins/Masking.axon` | 42 | `Tensor.expand ... shape=[B, 1, Q, K]` | Symbolic broadcast shape. |
| `builtins/Masking.axon` | 48 | `Tensor.slice ... start=(SM - K) end=SM` | Symbolic slice window into attention mask. |
| `builtins/Masking.axon` | 50 | `Tensor.reshape ... shape=[B, 1, 1, K]` | Symbolic pad mask shape after slice. |
| `builtins/Attention.axon` | 29 | `Tensor.reshape ... shape=[B, S, heads, (D / heads)]` | Symbolic + parameter `heads`. |
| `builtins/Attention.axon` | 34 | `Tensor.reshape ... shape=[B, S, (H * HD)]` | Symbolic merged-head shape. |
| `builtins/Attention.axon` | 39 | `?scale=(1.0 / sqrt HD)` | Symbolic dim in default kwarg expression. |
| `builtins/Attention.axon` | 52 | `slice probs ... start=0 end=K` | Symbolic trim after temporary sink concat. |
| `builtins/Attention.axon` | 75 | `reshape sink shape=[1, H, 1, 1]` | Symbolic head count in sink broadcast. |
| `builtins/Positions.axon` | 27 | `Tensor.slice ... start=(SM - S) end=SM` | Symbolic mask-tail slice. |
| `builtins/Positions.axon` | 34 | `Tensor.reshape ... shape=[1, S]` | Symbolic position row shape. |
| `builtins/Positions.axon` | 35 | `Tensor.expand ... shape=[B, S]` | Symbolic batch/sequence expansion. |
| `builtins/Positions.axon` | 53 | `Tensor.slice ... end=(R / 2)` | Symbolic half-rotation split. |
| `builtins/Positions.axon` | 54 | `Tensor.slice ... start=(R / 2) end=R` | Symbolic half-rotation split. |
| `builtins/Positions.axon` | 60-65 | `shape=[..., (R / 2), ...]`, `shape=[..., R]` | Symbolic interleaved rotate/merge pipeline. |
| `builtins/Positions.axon` | 76, 83 | `shape=[..., (2 * RH)]` | Symbolic half-to-full expansion. |
| `builtins/Positions.axon` | 88 | `reshape ... shape=[B, S, R]` | Symbolic rope reference reshape. |
| `builtins/Positions.axon` | 90-91 | `slice ... (R / 2)` | Symbolic sin/cos split. |
| `builtins/Positions.axon` | 100 | `slice ... end=HD` | Symbolic tail boundary. |
| `builtins/Positions.axon` | 122-127 | `rope_pair_base` flow | Symbolic dims carried through rope pair outputs. |
| `builtins/Positions.axon` | 132-142 | `(HD / 2)`, reshape with `HD` | Symbolic proportional-ROPE frequency construction. |
| `builtins/Positions.axon` | 150-152 | `inv_shape <- (2.0 * idx) / HD` | Symbolic dim in inverse-frequency base formula. |
| `builtins/Positions.axon` | 173 | `reshape inv_freq shape=[1, 1, (HD / 2)]` | Symbolic reshape for angle broadcast. |
| `builtins/Positions.axon` | 186 | `slice ... end=HD` | Symbolic tail boundary (inv-freq path). |
| `builtins/Positions.axon` | 356-359 | `end=Q`, `shape=[-1,1]`, `end=K`, `expand ...` | Symbolic relative-position bias layout. |

## Mixed Symbolic/Runtime Occurrences

Mixed = expression includes symbolic dims plus runtime/computed scalars.

| File | Line | Expression | Why mixed |
|---|---:|---|---|
| `builtins/Positions.axon` | 32 | `end=(past_length + S)` | `past_length` is runtime input; `S` is symbolic. |
| `builtins/Positions.axon` | 99 | `slice ... end=rot` | `rot` computed at runtime from `partial_rotary_factor`. |
| `builtins/Positions.axon` | 107-109 | `rot <- floor (HD * partial_rotary_factor)` + clamps | `HD` symbolic, factor runtime/default. |
| `builtins/Positions.axon` | 185 | `slice ... end=rot` | Same mixed boundary in inv-freq branch. |
| `builtins/Positions.axon` | 193-195 | `rot <- floor (HD * partial_rotary_factor)` + clamps | Same mixed derivation in inv-freq branch. |
| `builtins/Positions.axon` | 226 | `rot <- ... ? HD : floor (HD * partial_rotary_factor)` | Mixed conditional symbol/runtime. |
| `builtins/Positions.axon` | 229-234 | shapes and `idx` built from `(rot / 2)` | `rot` runtime-derived, still dim-like. |
| `builtins/Masking.axon` | 13, 40 | `(aligned +/- window)` | `window` runtime optional int; alignment terms symbolic. |

## Notes

- In models (`models/*/*.axon`), there are currently no direct `_shape`/`List.index(shape, ...)` dimension-extraction patterns analogous to old `Cache.past_length`.
- Most symbolic-dim value usage is concentrated in builtins (`Masking.axon`, `Attention.axon`, `Positions.axon`).
- `Tensor.size` in `builtins/Tensor.axon` intentionally remains runtime-generic (`dim` is a runtime argument), so it is not a pure-symbolic replacement candidate.
