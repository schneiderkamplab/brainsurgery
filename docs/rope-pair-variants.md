# Rope Pair Variants (5-way split, low-granularity form)

This keeps the same 5 semantic variants, but the Axon definitions avoid deprecated coarse ops like `rope_pair` and are composed from low-granularity ops.

## Variant Names

1. `rope_pair_base`
2. `rope_pair_freq_split`
3. `rope_pair_scaled_beta`
4. `rope_pair_longrope`
5. `rope_pair_yarn`

## Coverage Summary

Observed `rope_pair` call sites: 194

- `rope_pair_base`: 117
- `rope_pair_freq_split`: 38
- `rope_pair_scaled_beta`: 14
- `rope_pair_longrope`: 22
- `rope_pair_yarn`: 3

## Low-Granularity Definitions

```axon
rope_rotate_half_basic :: Tensor[B,H,S,R] -> ?Bool -> Tensor[B,H,S,R]
rope_rotate_half_basic x ?interleaved=false = if interleaved then
    rope_rotate_half_interleaved_basic x
  else
    rope_rotate_half_noninterleaved_basic x

rope_rotate_half_noninterleaved_basic :: Tensor[B,H,S,R] -> Tensor[B,H,S,R]
rope_rotate_half_noninterleaved_basic x = do
  x1 <- slice x dim=-1 start=0 end=(R / 2)
  x2 <- slice x dim=-1 start=(R / 2) end=R
  return concat (0 - x2) x1 dim=-1

rope_rotate_half_interleaved_basic :: Tensor[B,H,S,R] -> Tensor[B,H,S,R]
rope_rotate_half_interleaved_basic x = do
  pair <- reshape x shape=[B, H, S, (R / 2), 2]
  even <- slice pair dim=-1 start=0 end=1 |> reshape shape=[B, H, S, (R / 2)]
  odd <- slice pair dim=-1 start=1 end=2 |> reshape shape=[B, H, S, (R / 2)]
  neg_odd <- reshape (0 - odd) shape=[B, H, S, (R / 2), 1]
  even <- reshape even shape=[B, H, S, (R / 2), 1]
  return concat neg_odd even dim=-1 |> reshape shape=[B, H, S, R]
```

```axon
rope_apply_basic :: Tensor[B,H,S,R] -> Tensor[B,S] -> Float -> ?Bool -> Tensor[B,H,S,R]
rope_apply_basic x pos_ids theta ?interleaved=false = do
  ref <- slice x dim=1 start=0 end=1 |> reshape shape=[B, S, R]
  emb <- sinusoidal_positions ref pos_ids theta=theta offset=0 padding_idx=null
  sin_half <- slice emb dim=-1 start=0 end=(R / 2)
  cos_half <- slice emb dim=-1 start=(R / 2) end=R
  sin <- concat sin_half sin_half dim=-1 |> reshape shape=[B, 1, S, R]
  cos <- concat cos_half cos_half dim=-1 |> reshape shape=[B, 1, S, R]
  return (x * cos) + ((rope_rotate_half_basic x interleaved=interleaved) * sin)
```

```axon
rope_pair_base :: Tensor[B,H,S,HD] -> Tensor[B,H,S,HD] -> Tensor[B,S] -> Float -> ?Float -> ?Bool -> (Tensor[B,H,S,HD], Tensor[B,H,S,HD])
rope_pair_base q k pos_ids theta ?partial_rotary_factor=1.0 ?interleaved=false = do
  rot <- floor (HD * partial_rotary_factor)
  q_rot <- slice q dim=-1 start=0 end=rot |> rope_apply_basic pos_ids theta interleaved=interleaved
  k_rot <- slice k dim=-1 start=0 end=rot |> rope_apply_basic pos_ids theta interleaved=interleaved
  q_tail <- slice q dim=-1 start=rot end=HD
  k_tail <- slice k dim=-1 start=rot end=HD
  q_out <- concat q_rot q_tail dim=-1
  k_out <- concat k_rot k_tail dim=-1
  return q_out, k_out
```

For the remaining 4 variants, keep the same interfaces but compute angles via profile-specific helpers, then reuse `rope_apply_basic`:

```axon
rope_pair_freq_split q k pos_ids theta scale_factor low_freq_factor high_freq_factor original_context ?interleaved=false = do
  theta_eff <- rope_theta_freq_split_basic theta scale_factor low_freq_factor high_freq_factor original_context
  return rope_pair_base q k pos_ids theta_eff interleaved=interleaved
```

```axon
rope_pair_scaled_beta q k pos_ids theta scale_factor original_context beta_fast beta_slow ?mscale=null ?mscale_all_dim=null ?interleaved=false = do
  theta_eff <- rope_theta_scaled_beta_basic theta scale_factor original_context beta_fast beta_slow mscale mscale_all_dim
  return rope_pair_base q k pos_ids theta_eff interleaved=interleaved
```

```axon
rope_pair_longrope q k pos_ids theta original_context max_context long_factor short_factor ?attention_factor=1.0 ?partial_rotary_factor=1.0 ?long_mscale=null ?short_mscale=null ?interleaved=false = do
  theta_eff <- rope_theta_longrope_basic theta original_context max_context long_factor short_factor attention_factor long_mscale short_mscale
  return rope_pair_base q k pos_ids theta_eff partial_rotary_factor=partial_rotary_factor interleaved=interleaved
```

```axon
rope_pair_yarn q k pos_ids theta scale_factor low_freq_factor high_freq_factor original_context attention_factor ?interleaved=false = do
  theta_eff <- rope_theta_yarn_basic theta scale_factor low_freq_factor high_freq_factor original_context attention_factor
  return rope_pair_base q k pos_ids theta_eff interleaved=interleaved
```

## Ops Needed To Make All 5 Variants Fully Concrete

Already available and usable:
- `sinusoidal_positions`, `slice`, `reshape`, `concat`, `mul`, `add`, `floor`

Still needed for faithful full coverage (freq-split/beta/longrope/yarn exactness):
- `rotary_angles` (or equivalent): build RoPE angle tensor from a per-dim inverse-frequency profile, not just scalar `theta`
- `where`/comparison-driven frequency gating over angle index for Llama3-style split
- path/list-to-tensor helper for `long_factor`/`short_factor` lists in longrope modes

If you want, I can implement these in `Derived.axon` next as concrete modules plus the minimal new primitive(s), then migrate model call sites variant-by-variant.
