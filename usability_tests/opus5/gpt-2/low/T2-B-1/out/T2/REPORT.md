# T2 self-report (condition B)

- Final artifact path: `out/T2/plan.yaml` (output `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - `concat` requires each `from` reference to resolve to exactly one tensor, so the plan cannot be written with a pattern over layers; it had to be unrolled to 12 layers x 3 tensors (generated with a shell loop into the YAML).
  - `concat`/`copy` destinations must not already exist, so each pruned tensor was built under a temporary name, the original deleted, then moved back into place.
  - Tensor names contain dots, which are regex metacharacters; `delete` targets were escaped (`h\.0\.attn\.c_attn\.bias`) so they could not also match the `h.<i>.attn.bias` mask buffer.
  - Conv1D `[in, out]` layout means heads are column blocks in `c_attn` (dim 1) but row blocks in `c_proj` (dim 0); the two concats use different `dim`.
- Anything in the task text or documentation that was unclear:
  - The README lists `concat` without stating the one-tensor-per-reference restriction; that only appears in `help: concat`.
  - The README does not say whether `move`/`delete` targets are regex-matched with escaping expected; it was inferred from the tensor-reference section.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~10 minutes.
