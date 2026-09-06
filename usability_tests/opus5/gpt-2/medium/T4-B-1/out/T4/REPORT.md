# T4 self-report (condition B: BrainSurgery plan)

- Final artifact path: `out/T4/plan.yaml` -> `out/T4/model.safetensors`
- Number of times you executed the script or plan: 1 (plus 2 separate scratch
  plans under `out/` for verification, since removed; only `out/verify_plan.yaml`
  is left in place)
- Which executions failed, and why (one line each): none; the plan passed on the
  first execution.
- Pitfalls or surprises you hit (one line each):
  - The ordering hazard: scaling the base in place first and only then adding
    lambda*ft1 and lambda*ft2 keeps both task vectors against the untouched
    base, so I used the collected form out = 0.2*base + 0.4*ft1 + 0.4*ft2
    instead of two sequential base-relative updates.
  - Output alias inference forced every write onto the `base` alias, so the
    scaled fine-tune tensors had to be staged as `base::tv1.*` / `base::tv2.*`
    temporaries and deleted before saving.
  - Regex refs are full-match, so `h\.\d+\.mlp\....` does not accidentally pick
    up the `tv1.h.0.mlp....` temporaries; dots still need escaping.
  - `assert: equal` rewrites `right` from the `left` match, which made the
    "identical outside the MLPs" precondition a single line with a negative
    lookahead (`base::(?!h\.\d+\.mlp\.).+` vs `ft1::\g<0>`).
- Anything in the task text or documentation that was unclear:
  - The task lists `h.<i>.mlp.c_fc.weight` as `[768, 3072]` (Conv1D layout);
    that did not matter here because the merge is elementwise, but it would for
    anything shape-sensitive.
  - The docs do not say explicitly whether `add_`/`scale_` may target a
    different alias than the output one; I inferred the rule from the
    "Output behavior" section's alias-inference paragraph.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~10 minutes, most of it reading
  `docpack/help.txt`.

## Verification performed

Beyond the in-plan asserts (name/count checks on all three checkpoints,
non-MLP bit-equality across base/ft1/ft2, 48 merged tensors, 160 output
tensors, float32 dtype), I re-checked the written file with a second plan
(`out/verify_plan.yaml`) that recomputes the merge in the literal
`base + l*(ft1-base) + l*(ft2-base)` order and asserts equality with the
output within eps 1e-6: it passes. A negative control (asserting the output
equals ft1) fails as expected, so the equality asserts are not vacuous.
