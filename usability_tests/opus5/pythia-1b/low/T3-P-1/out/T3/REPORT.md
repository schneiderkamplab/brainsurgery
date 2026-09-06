# T3 participant self-report

- **Final artifact path:** `out/T3/solution.py` (output: `out/T3/model-0000{1..9}-of-00009.safetensors` + `out/T3/model.safetensors.index.json`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none; the single execution passed all checks and graded PASS.
- **Pitfalls or surprises you hit (one line each):**
  - Over-broad matching is the obvious trap: I anchored one regex to the four exact projection names (`attention.query_key_value|attention.dense|mlp.dense_h_to_4h|mlp.dense_4h_to_h` + `.weight$`) instead of anything like `.*weight`, so biases, layer norms and embeddings were never touched.
  - `attention.bias` looks like a parameter but is the uint8 causal mask; it is dropped along with `masked_bias` and `rotary_emb.inv_freq` by a separate anchored regex.
  - The two embedding matrices are 412 MB each in float32 (the 206 MB figure in TASK.md is the float16 input size), so both exceed the 256 MiB budget and the greedy packer must let an oversized tensor sit alone rather than erroring.
  - `save_file` wants contiguous, non-aliased storage, so I called `.contiguous()` on every output tensor after the dtype cast.
- **Anything in the task text or documentation that was unclear:**
  - The shard file naming convention is not stated; I used the HuggingFace `model-<i>-of-<n>.safetensors` pattern and it matched the reference.
  - The 256 MiB budget is stated as "at most", which leaves the packing order implicit; I used greedy fill in the input's key order (HF's `split_torch_state_dict_into_shards` behaviour).
  - "206 MB each" for the embeddings describes the input float16 size, not the float32 output size, which briefly made the shard arithmetic look wrong.
- **Tools used (condition F):** n/a (condition P: torch 2.14.0, safetensors 0.5.3 only).
- **Approximate time spent, if you can tell:** roughly 5 minutes — read the task and template, write the script once, run it once, grade.
