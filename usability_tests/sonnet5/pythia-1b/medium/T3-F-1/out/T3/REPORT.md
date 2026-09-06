# T3 report

## Participant self-report

- Final artifact path: `out/T3/solution.py` (invoked via `out/T3/run.sh`), output checkpoint in `out/T3/` (9 shards + `model.safetensors.index.json`).
- Number of times you executed the script or plan: 1.
- Which executions failed, and why: none.
- Pitfalls or surprises you hit:
  - The task says the embed tensors are "206 MB each" — that's their float16 *input* size; after the required upcast to float32 they're ~412 MiB (>256 MiB shard budget), so each must land alone in its own shard rather than exactly matching the quoted 206 MB figure. Worth double-checking against the actual output size rather than the input size when picking the oversized-tensor path.
  - `transformers.modeling_utils.split_torch_state_dict_into_shards` only sets `is_sharded=True` (and thus only builds an index) when there's more than one shard; had to handle the single-shard case explicitly to always emit an index file, though in this task there were always 9 shards so the fallback path never triggered here.
  - Regexes had to anchor on the exact 4 weight names (`query_key_value`, `dense`, `dense_h_to_4h`, `dense_4h_to_h`) rather than a broad `.*weight` pattern, since that would also catch `embed_in.weight`/`embed_out.weight`/layer-norm weights, which must stay float32.
- Anything in the task text or documentation that was unclear: the "206 MB each" figure for the embed tensors refers to the float16 input size, not the float32 output size actually written to the shard — this could be read as a check on the output shard size at first glance.
- Tools used (condition F): `torch` 2.14.0 (dtype casts, tensor ops), `safetensors` 0.5.3 (`safe_open`/`save_file` for reading the input and writing shards), `transformers` 5.12.1 (`split_torch_state_dict_into_shards`, the standard HF sharded-export helper, to compute shard membership and build the index/weight_map) — chosen because this is a plain, auditable script with no hidden model-family branching, and the sharding helper is exactly the HF-recommended routine for this format rather than reimplementing shard-packing logic by hand.
- Approximate time spent: ~10 minutes.
