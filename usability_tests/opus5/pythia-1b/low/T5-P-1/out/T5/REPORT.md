# T5 self-report (condition P)

- Final artifact path: `out/T5/solution.py` (output in `out/T5/`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why: none
- Pitfalls or surprises you hit:
  - Adapter names carry a `base_model.model.` prefix that must be stripped to match base names; `target_modules` in the config is `["query_key_value"]`, not the dotted `attention.query_key_value` from TASK.md.
  - Shard budget is tensor data only, so the greedy fill must be computed from `numel * element_size`, not file sizes (headers push shard 1 to 529 MB on disk).
  - `fan_in_fan_out=false` means `B @ A` is added directly; a transpose here would have silently produced a shape error only because the matrix is non-square.
- Anything unclear: the note that a tensor larger than 512 MiB is stored alone reads as if it applies to the 206 MB embeddings; they are well under the limit, so ordinary greedy packing covers it. The expected shard file naming was not specified; I used the HF `model-0000i-of-0000N.safetensors` convention.
- Tools used (condition F): n/a
- Approximate time spent: ~5 minutes
