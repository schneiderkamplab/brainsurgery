# T5 participant self-report (condition F, GPT-2 124M)

- Final artifact path: `out/T5/solution.py` (run with `.venv/bin/python out/T5/solution.py`; output shards and `model.safetensors.index.json` in `out/T5/`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - PEFT `merge_and_unload` on `GPT2LMHeadModel` silently loads nothing: the adapter names are `base_model.model.h.<i>...` (no `transformer.` prefix), so all 24 keys are "unexpected" and the merged delta is zero. Loading `GPT2Model` instead requires bypassing the `task_type=CAUSAL_LM` wrapper (which wants `prepare_inputs_for_generation`). I used this only as a verification, not as the solution.
  - Shard budget is on tensor bytes, not file size; the 154 MB `wte.weight` goes alone in its own shard even though it exceeds the budget.
- Anything in the task text or documentation that was unclear: nothing material. The task is explicit about scale, transpose and the shard rule. It does not specify shard file naming or tensor order within shards, so I used HF-style `model-0000i-of-0000n.safetensors` with greedy packing in base key order.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: `safe_open` to read base and adapter, `save_file` to write shards. Keeps unchanged tensors bit-exact and avoids instantiating a model.
  - `torch` 2.14.0: float32 matmul, transpose and add for the 12 merges.
  - `peft` 0.20.0 and `transformers` 5.12.1: only for an out-of-band verification (`get_peft_model` + `set_peft_model_state_dict` + `merge_and_unload` on `GPT2Model`), which agreed with my output with relative Frobenius error 0 on all 12 merged weights; 148 other tensors are bit-exact with the base.
  - Not used: mergekit, torch-state-bridge (no need for key rewriting or merge configs; a 100-line script was simpler and gives full control over the checks and sharding).
- Approximate time spent, if you can tell: about 5 minutes, most of it on the PEFT cross-check.
