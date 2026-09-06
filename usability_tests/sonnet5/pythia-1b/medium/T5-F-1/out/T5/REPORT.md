# T5 report (condition F)

## Participant self-report

- Final artifact path: `out/T5/solution.py`
- Number of times you executed the script or plan: 2 (1 failed, 1 succeeded)
- Which executions failed, and why (one line each):
  - Execution 1: `SystemExit: adapter targets 'gpt_neox.layers.0.attention.query_key_value', not present in base checkpoint` — the adapter key (after stripping the `base_model.model.` prefix and the `.lora_A.weight` suffix) names the module, not the tensor; I forgot to append the trailing `.weight` before looking it up in the base state dict, and derived `lora_B`'s key with a fresh f-string instead of a substring replace on the already-matched key.
- Pitfalls or surprises you hit (one line each):
  - PEFT adapter keys are prefixed with `base_model.model.` and have no `.weight` on the module path itself — easy to get the base-name mapping subtly wrong (see above).
  - The task's shard example ("embed_in/embed_out, 206 MB each, stored alone") doesn't actually hold under the stated 512 MiB budget and standard greedy shard-packing (each is only ~196.5 MiB, well under budget) — my output packs them alongside other tensors instead, following `huggingface_hub.split_torch_state_dict_into_shards`'s documented behavior (a tensor is only forced into its own shard when it individually exceeds `max_shard_size`). I did not chase this further since it isn't one of the four required checks and matches the tool `transformers.save_pretrained` itself uses.
  - Pythia-1B/GPT-NeoX is not tied-embedding (`tie_word_embeddings: false`), so `embed_in.weight` and `embed_out.weight` are genuinely distinct 244th/243rd tensors — worth checking `config.json` before assuming a HF model-instantiation route would round-trip all 244 tensors.
- Anything in the task text or documentation that was unclear:
  - See the shard-packing note above: the parenthetical about embed tensors being "stored alone in its own shard" doesn't follow from the stated 512 MiB threshold and their actual 196.5 MiB size, at least under the standard packing algorithm.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3 — direct, dependency-free read of both the base and adapter checkpoints as raw tensors, and writing the merged shards, without instantiating a model.
  - `torch` 2.14.0 — the float32 matmul (`B @ A`), scaling, and add-then-cast-to-float16 required by the spec.
  - `huggingface_hub` (pinned, a `transformers`/`peft` dependency) — `split_torch_state_dict_into_shards`, the same shard-bin-packing routine `PreTrainedModel.save_pretrained(..., safe_serialization=True)` uses internally, so the shard layout and index format match standard HF tooling rather than a bespoke implementation.
  - Did not use `peft.merge_and_unload`: it requires instantiating the full `GPTNeoXForCausalLM`, attaching the adapter, merging, then saving — more moving parts than necessary here, and it does not give explicit control over the float32-then-cast-to-float16 accumulation the spec calls for. Operating on the two state dicts directly was smaller and easier to check against the four required assertions before writing anything.
- Approximate time spent, if you can tell: ~10 minutes (mostly script authoring; one execution to catch the key-naming bug, one clean run).
