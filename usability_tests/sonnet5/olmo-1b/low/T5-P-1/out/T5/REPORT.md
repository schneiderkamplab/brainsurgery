## Participant self-report

- Final artifact path: `out/T5/solution.py`
- Number of times you executed the script or plan: 2 (the second run followed a fix to the sharding logic, no crashes)
- Which executions failed, and why (one line each):
  - None crashed. Run 1 produced a valid checkpoint but did not honor the
    requirement that `model.embed_tokens.weight` and `lm_head.weight` each be
    stored alone in their own shard — my initial packer only forced a tensor
    alone when it exceeded the 512 MiB cap by itself (412 MB doesn't), so
    embed_tokens got packed with a second tensor. Caught this by inspecting
    the output shard groupings before finishing, not by an exception.
- Pitfalls or surprises you hit (one line each):
  - The "alone in its own shard" rule for the two big tensors is a named
    exception to the size-based packing rule, not implied by the 512 MiB cap
    alone (412 MB < 512 MiB), so it needs to be encoded explicitly.
  - Adapter tensor names use the `base_model.model.model.layers...` PEFT
    double-prefix, which needs stripping to `model.layers...` to match base
    key names.
  - `fan_in_fan_out=false` means no transpose is needed since both the base
    Linear weights and the LoRA factors already use `[out, in]`; easy to
    second-guess this and add an unneeded transpose.
- Anything in the task text or documentation that was unclear:
  - None; the task text explicitly gave the layout convention and scale
    formula, so no guessing was required there.
- Tools used (condition F): name, version, and why: N/A (condition P)
- Approximate time spent, if you can tell: ~10 minutes
