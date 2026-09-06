# Participant self-report — T3 (condition F, repeat 1)

- Final artifact path: `out/T3/solution.py` (invoked via `out/T3/run.sh`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution
  succeeded and passed all in-script assertions.
- Pitfalls or surprises you hit (one line each):
  - The two >256 MiB tensors (`model.embed_tokens.weight`, `lm_head.weight`)
    each needed their own shard rather than being packed with anything else —
    handled with an explicit oversize branch in the greedy packer before
    applying the normal budget check.
  - Had to double check the 256 MiB budget is over *tensor data only*, not
    the safetensors header, so shard sizing uses `numel * element_size`
    rather than the on-disk file size.
- Anything in the task text or documentation that was unclear: none; the
  per-layer key list and shard-budget wording in TASK.md matched the actual
  checkpoint layout exactly.
- Tools used (condition F): `safetensors==0.5.3` (`safe_open`/`save_file`) and
  `torch==2.14.0` (`Tensor.to(torch.bfloat16)`) directly, in a plain script.
  No merging or key-renaming is required for this task — it's a per-tensor
  dtype cast plus a re-shard of the same key set — so `mergekit`,
  `torch-state-bridge`, and `transformers.save_pretrained` would only add
  indirection (extra config/YAML surface, or less control over exact shard
  boundaries) without doing anything a direct safetensors read/write doesn't
  already do exactly and verifiably. Verified afterward with an independent
  script comparing every output tensor bit-exactly against
  `orig.to(torch.bfloat16)` (bf16 tensors) or `orig` unchanged (fp32 tensors).
- Approximate time spent, if you can tell: ~10 minutes (single-pass script,
  no debugging iterations needed).
