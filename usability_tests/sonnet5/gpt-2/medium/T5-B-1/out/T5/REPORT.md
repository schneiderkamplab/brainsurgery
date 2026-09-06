# T5 report

## Participant self-report

- Final artifact path: `out/T5/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single run succeeded.
- Pitfalls or surprises you hit (one line each):
  - `matmul`/`permute`/`scale`/`add_` are batch transforms: `from_a` (or `from`)
    supplies the regex match set, and `from_b`/`to` are rewrite templates over
    its capture groups, so all 12 layers could be merged with one transform
    each instead of a per-layer block.
  - Conv1D vs Linear layout: `B @ A` yields the `nn.Linear` `[out, in]`
    orientation; it has to be transposed (`permute` with `order: [1, 0]`)
    before adding into the Conv1D `[in, out]` base weight, per
    `fan_in_fan_out = true`.
  - Intermediate tensors (`merge.<i>.ba`, `merge.<i>.ba_t`, `merge.<i>.delta`)
    had to be written onto the `base` alias, not `lora`, otherwise the plan
    writes to two aliases and output-alias inference fails with "cannot infer
    output model uniquely"; they are deleted before saving along with all
    `lora::` tensors.
- Anything in the task text or documentation that was unclear: none; the
  scale formula, layout note, and sharding rule were sufficient to write the
  plan without additional lookups.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: one pass, well under the cap.
