# T5 participant self-report (condition B, OLMo-1B-0724-hf)

- Final artifact path: `out/T5/plan.yaml` (output checkpoint in `out/T5/`,
  10 shards plus `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution
  succeeded.
- Pitfalls or surprises you hit (one line each):
  - With two inputs, the output alias is inferred from which alias the
    transforms write to, so the `matmul` intermediate had to be created on
    `base::` (not `lora::`) and the adapter alias left untouched; otherwise the
    run would fail with "cannot infer output model uniquely".
  - `matmul` requires a new destination and `add_` requires an existing one,
    so the merge is matmul -> scale_ -> add_ -> delete rather than a single step.
  - The PEFT name prefix `base_model.model.` had to be stripped via a regex
    capture group that is reused in `from_b` and `to`.
- Anything in the task text or documentation that was unclear:
  - TASK.md says `model.embed_tokens.weight` and `lm_head.weight` (412 MB) are
    "larger than" the 512 MiB budget and stored alone; they are in fact
    smaller, so the tool packs `embed_tokens` together with one more tensor in
    its shard (479 MB total). Each shard still respects the 512 MiB budget.
  - The help text for `matmul` does not show a multi-match capture example;
    the interfaces reference's "Mapping note" was needed to confirm that
    `from_b` and `to` are rewrites of the `from_a` match.
- Tools used (condition F): not applicable (condition B).
- Approximate time spent, if you can tell: about 5 minutes, of which the plan
  execution took 13 seconds.
