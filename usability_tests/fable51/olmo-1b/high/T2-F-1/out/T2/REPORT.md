# T2 self-report (condition F)

- Final artifact path: `out/T2/solution.py` (writes `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - In a separate verification snippet (not the solution), constructing `OlmoConfig` with `num_attention_heads=15` made transformers derive `head_dim = 2048 // 15 = 136`, so `load_state_dict` reported `[2040, 2048]`; setting `head_dim=128` explicitly fixed it. Anyone loading the pruned checkpoint via HF needs to set `head_dim` in the config, not just the head count.
- Anything in the task text or documentation that was unclear: nothing; row/column ranges and the `[out, in]` layout were stated explicitly.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: read the two input shards listed in `model.safetensors.index.json`, write the single output file, and re-open it to verify keys and shapes on disk.
  - `torch` 2.14.0: `index_select` on rows (q/k/v) and columns (o_proj) with a keep-index of `0..639, 768..2047`, then `.contiguous()` so safetensors accepts the slices.
  - `transformers` 5.12.1: used only after the fact to confirm the file loads strictly into `OlmoForCausalLM` with 15 heads and `head_dim=128` and produces finite logits. I did not use `prune_heads` / `save_pretrained` for the output because grading is bit-exact on a single 114-tensor file with unchanged names, and a direct slice is the shortest path to that with no risk of resharding, config rewriting, or dtype changes.
  - Independent verification (ad hoc, not in the artifact): reloaded input and output and checked every one of the 114 tensors is bit-equal to the expected slice or to the untouched original (0 mismatches).
- Approximate time spent, if you can tell: about 3 minutes.

## Checks enforced by `solution.py` before writing

- input has exactly 114 tensors and no duplicate keys across shards;
- every q/k/v/o projection in all 16 layers has the expected pre-prune `[2048, 2048]` shape;
- exactly 64 tensors were touched;
- q/k/v in every layer are `[1920, 2048]`, o_proj is `[2048, 1920]`, all float32 (the required layer-0 checks are a subset);
- state dict has exactly 114 tensors;
- destination does not already exist.

After writing, the file is re-opened and the key count and shapes are checked again on disk.
