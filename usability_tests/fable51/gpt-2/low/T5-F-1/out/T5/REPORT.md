# T5 self-report (condition F)

- Final artifact path: `out/T5/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - `adapter_config.json` lists `target_modules = ["c_attn"]` while TASK.md says `attn.c_attn`; the script derives the base name from the adapter tensor names instead, so this did not matter.
  - `fan_in_fan_out = true` means the `B @ A` product (`[out, in]`) must be transposed onto the Conv1D `[in, out]` base weight; the script reads the flag from the config rather than hard-coding it.
- Anything in the task text or documentation that was unclear: nothing significant; the shard budget wording ("tensor data, not counting headers") was clear.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: load base and adapter, save shards with `metadata={"format": "pt"}`.
  - `torch` 2.14.0: float32 matmul, transpose, add, and `torch.equal` in a post-run verification.
  - Plain Python for name mapping, the four required checks (12 pairs, no `lora_` names, `h.0.attn.c_attn.weight` shape, 160 tensors), greedy 100 MiB sharding and the index file. I did not use `peft.merge_and_unload` because it requires instantiating the model and would round-trip through `transformers` naming and `save_pretrained`, which adds surface for surprises (weight tying, key prefixes) with no benefit for a 12-pair merge.
- Approximate time spent, if you can tell: about 3 minutes.
