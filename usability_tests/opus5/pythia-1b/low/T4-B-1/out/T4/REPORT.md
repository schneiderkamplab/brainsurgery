# T4 self-report (condition B, BrainSurgery plan)

- Final artifact path: `out/T4/plan.yaml` -> `out/T4/model.safetensors`
- Number of times you executed the script or plan: 3
- Which executions failed, and why (one line each):
  1. `failed_assertion` — `equal failed: ft1::gpt_neox.layers.0.attention.masked_bias != base::...`: I passed `eps: 0.0` to the bit-exactness check, and `masked_bias` is `-inf`, so `|left-right|` is NaN and `NaN <= 0` is false. Dropping `eps` uses `torch.equal`, which is bit-exact and inf-safe.
  2. `failed_assertion` — `dtype failed: base::gpt_neox.layers.0.attention.bias has dtype torch.uint8, expected torch.float16`: my final dtype check covered all 244 tensors, but the causal-mask buffers `attention.bias` are `uint8`. Narrowed the check to the 64 merged MLP tensors.
- Pitfalls or surprises you hit (one line each):
  - `equal` with `eps` compares via subtraction, so `-inf` vs `-inf` is a false negative; omit `eps` when you want bit-exactness.
  - Not every tensor in a float16 checkpoint is float16 (`attention.bias` is `uint8`, and `masked_bias` is a scalar `-inf`).
  - Output alias inference: temporaries must be created *inside* the `base` alias (`to: 'base::tv1__\1'`), otherwise the plan writes to more than one alias and cannot pick an output.
  - Avoided the ordering hazard algebraically: `base + L*(ft1-base) + L*(ft2-base) == (1-2L)*base + L*ft1 + L*ft2`, so both task vectors are taken against the unmodified base by construction; no intermediate state of `base` can contaminate the second vector.
- Anything in the task text or documentation that was unclear:
  - The README does not say which of `from`/`to` is the matcher for the in-place binary transforms (`add_`, `subtract_`); `interfaces-reference.md` line 231 answers it (`from` matches, `to` is the rewrite).
  - The README does not mention that `equal` with `eps` is subtraction-based, which is what caused failure 1.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~15 minutes.

## Verification

Independent check of the written file: 244 tensors, key set identical to the
base, all dtypes matching the base, the 180 non-MLP tensors bit-identical to
the base, and worst-case relative Frobenius error over the 64 merged tensors
2.4e-4 (limit 1e-3).
