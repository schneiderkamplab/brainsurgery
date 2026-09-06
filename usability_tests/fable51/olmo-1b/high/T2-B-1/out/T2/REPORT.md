# T2 (OLMo-1B-0724-hf, condition B) — Participant self-report

- Final artifact path: `out/T2/plan.yaml` (output checkpoint: `out/T2/model.safetensors`, 114 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution passed all asserts and wrote the output.
- Pitfalls or surprises you hit (one line each):
  - `concat` requires each `from` reference to resolve to exactly one tensor, so it cannot be batched over layers with a regex; the plan needs one `concat` per layer per projection (64 in total), which I generated with a shell loop rather than writing by hand.
  - There is no in-place slice/narrow transform, so the pruned tensor has to be built under a temporary name, the original deleted, and the temporary moved back; `delete` and `move` do batch over layers with regex captures, which kept those steps to one transform each.
  - Tensor names contain dots, so I escaped them (`\.`) in every regex reference to avoid accidental matches.
- Anything in the task text or documentation that was unclear:
  - The `concat` and `split` help examples show an empty `from:`/`to:` list (`concat: { from: , to: a::xy, dim: 0 }`), which looks like a rendering bug; the second `concat` example with sliced references was the useful one.
  - The README does not state explicitly that a `.safetensors` file path in `output.path` produces a single unsharded file; it worked as hoped.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 5 minutes (reading the doc pack, generating the plan, one run, one independent verification pass).
