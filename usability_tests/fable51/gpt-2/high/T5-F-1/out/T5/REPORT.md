# T5 participant self-report (condition F, GPT-2 124M)

- Final artifact path: `out/T5/solution.py` (run with `.venv/bin/python out/T5/solution.py`); output shards and `model.safetensors.index.json` in `out/T5/`.
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The base checkpoint stores the `h.<i>.attn.bias` causal-mask buffers and uses no `transformer.` prefix, so a transformers/PEFT `merge_and_unload` + `save_pretrained` route would have dropped those buffers and renamed every key; that ruled out the "obvious" PEFT route for exact key-set parity.
  - `fan_in_fan_out = true` means the Conv1D base is `[in, out]` while `B @ A` is `[out, in]`; the delta must be transposed before adding (the task text spells this out, so it was not a trap here).
  - `wte.weight` (154 MB) exceeds the 100 MiB shard budget and has to sit alone in its shard; greedy fill in base key order gives 5 shards.
- Anything in the task text or documentation that was unclear:
  - The task does not say whether the grader expects a specific shard assignment or filename pattern, only the size rule and the index file. I used greedy packing in base key order and HF-style `model-0000N-of-0000M.safetensors` names.
  - The adapter config lists `target_modules = ["c_attn"]` while TASK.md says `["attn.c_attn"]`; harmless since names were derived from the adapter tensor names, not the config.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: `safe_open` for reading base and adapter, `save_file` for writing shards. Chosen because it operates on the checkpoint files directly and preserves names, dtypes and bytes exactly.
  - `torch` 2.14.0: float32 matmul, transpose and add for the merge.
  - `numpy` 2.5.2: only in a separate float64 verification pass after the run (not part of the solution).
  - Not used: `peft`/`transformers` (would change the key set, see pitfalls), `mergekit` (no LoRA-merge-on-files mode that preserves this key set), `torch-state-bridge` (no key rewriting was needed).
- Approximate time spent, if you can tell: about 5 minutes wall clock.

## Result summary

- 12 adapter pairs merged with scale 2.0, delta transposed for Conv1D layout.
- 160 tensors written, same names as the base; no `lora_` names.
- Independent check after the run: 148 unchanged tensors bit-exact, worst relative Frobenius error on merged weights 2.6e-8.
- 5 shards; the four multi-tensor shards hold at most 104,718,336 bytes of tensor data, `wte.weight` alone in the fifth.
