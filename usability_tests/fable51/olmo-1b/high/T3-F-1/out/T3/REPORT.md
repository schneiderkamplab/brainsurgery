# T3 participant self-report (condition F)

- Final artifact path: `out/T3/solution.py` (wrapper `out/T3/run.sh`); output checkpoint in `out/T3/` (10 shards + `model.safetensors.index.json`).
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution passed all checks.
- Pitfalls or surprises you hit (one line each):
  - Targeting is done with an explicit set of 112 fully spelled-out key names (16 layers x 7 projections), not a regex, so embeddings and lm_head cannot be caught by accident.
  - Each layer's 7 bf16 matrices sum to exactly 128 MiB, so two layers fill a 256 MiB shard exactly; the greedy fill uses `>` (not `>=`) against the budget, matching "at most".
  - `model.embed_tokens.weight` and `lm_head.weight` (412 MB float32 each) exceed the budget and are stored alone as shards 1 and 10.
- Anything in the task text or documentation that was unclear:
  - The task does not fix the shard iteration order or file naming; I used HF order (embed, layers 0..15, lm_head) and `model-0000N-of-0000M.safetensors` names, greedy fill, on the assumption that grading checks the sharding rules rather than exact file membership.
- Tools used (condition F): name, version, and why:
  - `torch` 2.14.0: loading tensors, `to(torch.bfloat16)` for RNE casting, `torch.equal` for the unchanged-value checks.
  - `safetensors` 0.5.3: `safe_open` to read the input shards and `save_file` to write the output shards.
  - No transformers/mergekit/huggingface_hub sharding helpers: the task fixes a dtype per exact tensor name and a specific shard budget, and a 100-line script makes both explicit and enforceable (`save_pretrained(dtype=...)` casts uniformly, and mergekit's dtype conversion is also checkpoint-wide).
- Approximate time spent, if you can tell: about 5 minutes; the script runs in ~7 s.

## Checks enforced at runtime (script exits non-zero if any fails)

Before writing: 114 input tensors all float32; all 112 projection keys present; exactly 112 bf16 outputs; `model.layers.0.self_attn.q_proj.weight` is bf16; `model.embed_tokens.weight` and `lm_head.weight` are float32; exactly 114 output tensors with the same key set and shapes; every non-cast tensor bit-equal to input; no shard over 256 MiB unless it holds a single tensor.
After writing: every shard is reopened, every tensor compared bit-exact to the in-memory result, the index maps 114 names to the files that actually hold them.
