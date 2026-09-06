# T2 self-report (condition F)

- Final artifact path: `out/T2/solution.py` (output `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - `transformers` `prune_heads` was rejected up front: `save_pretrained` changes key prefixes and drops the `attn.bias` mask buffers, and the grader wants the exact 160-key set bit-exact, so a direct slice was safer.
  - Sanity-loading the result as a stock HF GPT-2 with `n_head=11` fails (`embed_dim` 768 not divisible by 11); HF only supports pruned heads via `config.pruned_heads`, not a smaller `n_head`.
  - The name suffix `.attn.c_proj.weight` must be matched with the `attn.` part, otherwise `mlp.c_proj.weight` would be sliced too; handled by suffix matching plus a touched-count check (36).
- Anything in the task text or documentation that was unclear: nothing; the explicit column ranges made the layout unambiguous.
- Tools used (condition F): name, version, and why:
  - `torch` 2.14.0: `index_select` slicing and `torch.equal` checks.
  - `safetensors` 0.5.3: `load_file` / `save_file`.
  - `transformers` 5.12.1: tried only as a post-hoc sanity load (not part of the solution); a stock `GPT2Config(n_head=11)` is rejected because the class derives head size as `768 / n_head`, so the pruned file cannot be loaded that way (HF represents pruned heads via `pruned_heads` in the config instead). Not needed for the task.
- Approximate time spent, if you can tell: about 3 minutes.
