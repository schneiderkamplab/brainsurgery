## Participant self-report

- Final artifact path: `out/T5/solution.py` (run via `out/T5/run.sh`), output
  written to `out/T5/model-*.safetensors` + `out/T5/model.safetensors.index.json`.
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none — the single run
  succeeded and passed all required checks.
- Pitfalls or surprises you hit (one line each):
  - The two large tensors (`model.embed_tokens.weight`, `lm_head.weight`,
    ~393 MiB each) are each individually under the 512 MiB shard cap, so
    whether either lands alone in a shard depends on which tensor happens to
    be next to it in iteration order during greedy bin-packing, not on their
    own size; I preserved the base checkpoint's natural per-shard-file tensor
    order (the same order a live `model.state_dict()` would produce) rather
    than re-sorting keys, so this fell out consistently instead of being
    forced.
  - The base checkpoint's index JSON `weight_map` is alphabetically sorted
    for readability, but the physical tensor order inside each shard file is
    not (it follows the original module registration order); iterating via
    `safe_open(...).keys()` per shard file (not the JSON key order) matters
    for reproducing a natural bin-packing.
- Anything in the task text or documentation that was unclear: the note that
  `model.embed_tokens.weight` and `lm_head.weight` are each "stored alone in
  its own shard" reads as if it follows from their size, but at ~393 MiB
  each they're individually under the 512 MiB cap — whether either ends up
  alone is a consequence of bin-packing order, not size alone. My output
  gives `lm_head.weight` its own shard and packs `model.embed_tokens.weight`
  with one small tensor (still well under 512 MiB), which respects every
  literal required check and the stated total-per-shard cap.
- Tools used (condition F): `torch` 2.14.0 and `safetensors` 0.5.3 for
  tensor I/O and the float32 `scale * B @ A` arithmetic; `huggingface_hub`
  1.16.1's `split_torch_state_dict_into_shards` for the shard/index
  construction — this is the same greedy bin-packing helper `transformers`
  and `peft`'s own sharded `save_pretrained` call internally, so using it
  directly matches "sharding rules" exactly while keeping the merge logic
  (which is only 32 explicit `B @ A` products) as a plain, auditable script
  rather than routing through a full `PeftModel.merge_and_unload()`, which
  would require reconstructing an OLMo `nn.Module` and loading the entire
  base model into it just to perform the same arithmetic. `peft` and
  `mergekit` were both available and considered but not needed given how
  small and explicit the mapping from adapter names to base names is here.
- Approximate time spent, if you can tell: ~15 minutes end to end (mostly
  verification of the merge and shard-size constraints against the input
  tensors, not iteration on the script itself).
