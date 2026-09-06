# T3 participant self-report

- **Final artifact path:** `out/T3/solution.py` (writes `out/T3/model-0000N-of-00010.safetensors` + `out/T3/model.safetensors.index.json`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none — the single execution succeeded.
- **Pitfalls or surprises you hit:**
  - The obvious regex route (`.*weight`, or even `.*proj\.weight`) would sweep in `lm_head.weight` / `model.embed_tokens.weight`, so I built the 112 target names literally from `layers 0..15 x {q,k,v,o,gate,up,down}_proj` instead of matching patterns.
  - The `transformers` `save_pretrained(dtype=...)` route suggested in `F-allowed.md` is a whole-model cast; it cannot keep embeddings in float32 while casting only the projections, so it was not usable here.
  - `huggingface_hub`'s shard budget is on tensor bytes only, so the 268,437,072-byte shard *files* are correct: the tensor payload is exactly 268,435,456 bytes and the excess is the safetensors header.
  - The two 412 MB float32 embedding matrices each land alone in a shard, which is what the task requires; I asserted that invariant explicitly rather than trusting the splitter.
- **Anything in the task text or documentation that was unclear:**
  - The task lists "drop non-parameter buffers" and "upcast what must be float32" in the objective, but the Input section then states this checkpoint has no buffers, no norms and no biases. Those steps were therefore no-ops; I deliberately deleted and upcast nothing.
  - The shard budget is specified in tensor bytes ("not counting file headers") but the exact shard *assignment* rule (packing order, greedy vs. balanced) is not pinned down, so a valid answer may still differ file-by-file from the hidden reference. I used the input index's key order with the standard HuggingFace greedy packer, which yields 2 singleton shards + 8 shards of exactly 256 MiB.
- **Tools used (condition F):**
  - `safetensors` 0.5.3 — per-tensor read from the input shards and `save_file` for the output; the only thing that gives exact per-tensor dtype control.
  - `torch` 2.14.0+cu130 — `tensor.to(torch.bfloat16)` for the round-to-nearest-even cast required by the spec.
  - `huggingface_hub` 1.16.1 — `split_torch_state_dict_into_shards` for the 256 MiB sharding and the `weight_map`, so the layout matches what serving stacks expect rather than a hand-rolled splitter.
  - Considered and rejected: `transformers.save_pretrained(dtype=...)` (all-or-nothing cast, see above), `mergekit` (its dtype conversion is likewise global and it is built for merging, not selective export), `torch-state-bridge` (renames keys; no renaming needed here), `peft` (no adapters involved).
- **Approximate time spent:** ~10 minutes, of which ~18 s was the run itself.

## Checks enforced by the run

`out/T3/solution.py` aborts with `SystemExit("FAIL: ...")` before writing if any of these does not hold, and re-verifies them against the files on disk afterwards:

- exactly 112 bfloat16 tensors;
- `model.layers.0.self_attn.q_proj.weight` is bfloat16;
- `model.embed_tokens.weight` is float32;
- exactly 114 tensors in the output;
- every non-target tensor is still float32;
- all 112 expected projection names exist in the input;
- no shard exceeds 256 MiB of tensor data unless it holds a single tensor;
- no tensor appears in two shards, and the index `weight_map` matches the tensors actually on disk.
