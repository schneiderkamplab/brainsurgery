# Participant self-report: T5 (OLMo-1B-0724-hf, condition P)

- Final artifact path: `out/T5/solution.py` (output checkpoint in `out/T5/`: 10 shards plus `model.safetensors.index.json`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - None failed. Execution 1 succeeded but packed a 64 MiB tensor into the shard with `model.embed_tokens.weight`; execution 2 was after changing the packer so the two 412 MB tensors are stored alone, as the task text states. The stale shards from execution 1 were removed before rerunning because the script refuses to overwrite existing shard files.
- Pitfalls or surprises you hit (one line each):
  - The task says a tensor "larger than" 512 MiB is stored alone but then names two 412 MB tensors as examples; a plain greedy packer under the cap does not isolate them, so I isolate any tensor above half the shard budget.
  - `adapter_config.json` lists `target_modules` as `q_proj`/`v_proj` without the `self_attn.` prefix, unlike TASK.md; I mapped names by stripping the `base_model.model.` prefix and the `lora_A`/`lora_B` segment rather than by `target_modules`.
- Anything in the task text or documentation that was unclear:
  - Whether the grader requires the exact shard layout of the reference (count, order, file names) or only the size rule and index consistency. I used sorted tensor names, greedy packing, and HF-style `model-0000i-of-0000n.safetensors` names.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 5 minutes; each script run takes about 10 s.
