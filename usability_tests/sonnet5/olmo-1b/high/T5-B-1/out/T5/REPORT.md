## Participant self-report

- Final artifact path: `out/T5/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - `matmul`/`add_`/`scale_` resolve `from_a`/`from`/`target` as the driving
    regex and rewrite the other references from its capture groups (like
    `copy`/`assign`), so a single transform block can do the LoRA multiply
    or add for all 32 layer/module pairs at once instead of looping per layer.
  - Named the intermediate `B @ A` tensor `...merge_delta.weight` rather than
    anything containing `lora_`, since the "no tensor name containing
    `lora_`" check is easy to trip on an intermediate name by accident (I
    deleted the intermediate before the final asserts either way, but the
    naming made the intent clearer).
  - Verified the output alias inference rule: only the `base` alias's
    tensors get written to `out/T5`, so the `lora` alias's adapter tensors
    never needed explicit deletion.
- Anything in the task text or documentation that was unclear:
  - The task text says the 412 MB `embed_tokens`/`lm_head` tensors are
    "stored alone in its own shard" as if each individually exceeds the
    512 MiB budget, but neither does on its own (412 MB < 512 MiB). In
    practice the built-in greedy packer puts `embed_tokens` alone anyway
    (because adding the next tensor would exceed the budget) and packs
    `lm_head` with one more small tensor into the next shard (bit-verified
    combined shard is under 512 MiB). I did not change my plan for this —
    I just used `output.shard: 512MB` and trusted the tool's packing
    algorithm — but the wording could be read as requiring each of those
    two tensors to occupy a shard with nothing else in it, which is not
    what the tool actually does.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: ~10 minutes (single pass: read
  docs/help, inspect inputs, write plan, run once, verify numerically against
  the raw safetensors).
