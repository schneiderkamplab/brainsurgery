# T2 participant self-report (condition B, GPT-2 124M)

- Final artifact path: `out/T2/plan.yaml` (output checkpoint: `out/T2/model.safetensors`, 160 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - `concat` requires each source reference to resolve to exactly one tensor and takes a single destination, so it cannot fan out over layers with a regex capture; the plan has one explicit concat block per layer and tensor (36 concats total), generated with a shell loop.
  - Tensor names cannot be edited in place, so pruned tensors are built under a temporary `pruned.*` prefix, the originals are deleted, and `move` with a regex capture renames them back to the original names.
  - The `delete` regex must be anchored on `attn.` so it does not catch `mlp.c_proj.weight`; the plan asserts the mlp shape afterwards as a guard.
  - Dots in tensor references are regex metacharacters (full-match regex); I escaped them (`h\.0\.attn\.…`) to avoid accidental matches.
- Anything in the task text or documentation that was unclear:
  - The `concat` and `split` help examples print an empty list for `from`/`to` (`concat: { from: , to: a::xy, dim: 0 }`), which looks like a rendering bug; the second example was enough to infer the list syntax.
  - The README does not say whether pattern references are allowed in `concat`; I had to infer from the "must resolve to exactly one tensor" wording that per-layer blocks are needed.
- Tools used (condition F): n/a (condition B)
- Approximate time spent, if you can tell: about 5 minutes (reading the doc pack, generating the plan, one run, an independent bit-exact check of the output against the input).
