# T2 (Pythia-1B, condition B) participant self-report

- Final artifact path: `out/T2/plan.yaml` (output checkpoint `out/T2/model.safetensors`, 244 tensors)
- Number of times you executed the script or plan: 1 execution of `out/T2/plan.yaml` (succeeded). One additional run of a separate assert-only plan, `out/T2/verify.yaml`, that loads the input and the output and checks bit-exact equality of untouched tensors and of the kept slices; it writes nothing.
- Which executions failed, and why (one line each): none.
- Pitfalls or surprises you hit (one line each):
  - `concat` requires each source ref to resolve to exactly one tensor, so the per-layer slicing cannot be written once with a regex capture; the plan has 48 explicit `concat` entries (3 per layer, generated mechanically).
  - The pruned tensors have new shapes, so they cannot be assigned in place; they are built under temporary names (`pruned.layers.<i>...`), the originals are deleted by regex, then `move` with a regex capture renames them back in one entry.
  - Head 5's block boundaries follow from the interleaved layout given in the task: qkv rows 3840..4607, dense columns 1280..1535.
- Anything in the task text or documentation that was unclear:
  - The README does not say whether `assert.shape` accepts a pattern matching several tensors, so the cross-layer checks use `count` instead and shape checks are per named tensor.
  - Whether `equal`'s `right` accepts a slice suffix is not documented; it does (used in `verify.yaml`).
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: about 3 minutes wall clock.
