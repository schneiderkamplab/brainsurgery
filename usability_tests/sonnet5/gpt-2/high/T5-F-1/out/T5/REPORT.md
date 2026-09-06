# T5 report

## Participant self-report

- Final artifact path: `out/T5/solution.py` (produces `out/T5/model-*-of-00005.safetensors`
  and `out/T5/model.safetensors.index.json`; invoked via `out/T5/run.sh`).
- Number of times you executed the script or plan: 1.
- Which executions failed, and why (one line each): none; the single run succeeded.
- Pitfalls or surprises you hit (one line each):
  - The Conv1D vs. Linear layout distinction (`fan_in_fan_out`) is easy to get backwards;
    wrote an explicit assertion on `h.0.attn.c_attn.weight`'s shape after the transpose to
    catch it early rather than trusting the arithmetic silently.
  - The adapter tensor names are prefixed with `base_model.model.` and use dotted module
    paths (`h.<i>.attn.c_attn`) rather than the bare base checkpoint names; matched them
    with a regex on `lora_A`/`lora_B` instead of assuming a fixed template.
  - The 100 MiB shard budget is tight against 12 similarly-sized attention blocks — a naive
    "N tensors per shard" split would not respect it; used strict greedy byte-packing instead
    (close the shard right before it would exceed the budget; an empty shard always takes the
    next tensor even if oversized, which is what puts `wte.weight` alone in its own shard).
- Anything in the task text or documentation that was unclear: no; the required checks and
  layout explanation in TASK.md were sufficient to write and verify the merge without
  consulting `peft` internals.
- Tools used (condition F): `torch` 2.14.0 and `safetensors` 0.5.3 only, via a plain script
  operating directly on the two checkpoints' state dicts. Did not use `peft.merge_and_unload`
  because that API is built around merging into a live `nn.Module` (build the GPT-2 model,
  wrap it in `PeftModel`, merge, re-export), which pulls in a full model instantiation for a
  merge that is fully specified as linear algebra on two safetensors files; a direct script
  is smaller and lets the required checks (pair count, no leftover `lora_` names, shape,
  tensor count) run against the exact tensors being written, before anything is written.
  `mergekit`'s task-arithmetic and passthrough recipes assume same-shaped dense checkpoints,
  not low-rank adapter factors, so it doesn't apply to this task. `torch-state-bridge`'s
  key-rewriting features weren't needed since no tensor renaming was required, only
  in-place addition to existing base names.
- Approximate time spent, if you can tell: one authoring pass plus verification, no retries.
