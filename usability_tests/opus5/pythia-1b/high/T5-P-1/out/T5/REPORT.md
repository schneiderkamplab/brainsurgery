# T5 — Participant self-report (condition P: Python / PyTorch)

- **Final artifact path:** `out/T5/solution.py`
  (output: `out/T5/model-0000{1..5}-of-00005.safetensors` + `out/T5/model.safetensors.index.json`)

- **Number of times you executed the script or plan:** 1 (passed on the first
  execution). Before running I did a syntax-only `py_compile` of the file and
  inspected the two input safetensors headers with shell tools (`od`/`dd`/`jq`)
  — no partial computation of the result was executed.

- **Which executions failed, and why (one line each):** none.

- **Pitfalls or surprises you hit (one line each):**
  - The shard spec is internally inconsistent: it says a tensor *larger than
    512 MiB* is stored alone, then names `gpt_neox.embed_in.weight` /
    `embed_out.weight` (206,045,184 B = 196 MiB each) as such tensors — no
    tensor in this checkpoint exceeds the budget. I read the sentence as a
    description of the emergent behaviour of greedy packing and used plain
    greedy packing, under which those two share shard 1 with 20 small tensors.
  - The base checkpoint holds 16 `attention.bias` causal-mask buffers of dtype
    `U8` `[1,1,2048,2048]` (4 MiB each); a `numel * 2` size estimate would have
    over-counted them, so shard sizes are computed with `element_size()`.
  - safetensors does not store tensors in name order: the header is ordered by
    data offset, which is *dtype-size descending, then name*, so the 16 `U8`
    masks sit at the very end. I sharded in that on-disk order rather than in
    `safe_open().keys()` order (which is sorted) so the layout follows the
    checkpoint's own order; the 4 MiB tail tensor lands alone in shard 5.
  - PEFT names carry a `base_model.model.` prefix that has to be stripped, and
    the pattern must tolerate an optional `.default` adapter-name segment.
  - `masked_bias` is a 0-d scalar tensor; anything that assumes `ndim >= 1`
    over the base state dict trips on it.

- **Anything in the task text or documentation that was unclear:**
  - The 512 MiB / 206 MB contradiction above. It is the one place where a
    grader comparing shard assignment (rather than shard *rules*) could
    disagree with a correct-looking solution.
  - "no intermediate tensor appears in the output" is unambiguous here but does
    not say whether the index file may carry extra `metadata` keys; I wrote
    only `total_size`, plus `{"format": "pt"}` in each shard's own metadata to
    match the base file.
  - The task does not state a shard-file naming convention; I used the HF
    convention `model-<i>-of-<n>.safetensors`.

- **Tools used (condition F):** n/a — condition P.
  Environment: `torch 2.14.0`, `safetensors 0.5.3`, `numpy 2.5.2`.

- **Approximate time spent, if you can tell:** ~6 minutes wall clock; the
  script itself runs in ~9 s on CPU.

## What the script does

1. Reads `adapter_config.json` for `r`, `lora_alpha`, `fan_in_fan_out`; derives
   `scale = lora_alpha / r = 2.0`.
2. Loads the base in its on-disk (data-offset) order and the adapter, pairs
   `lora_A`/`lora_B` per module after stripping `base_model.model.`.
3. For each pair: `delta = scale * (B.float() @ A.float())` (transposed only if
   `fan_in_fan_out`), added to the base in float32, cast back to float16.
4. Fails loudly unless: exactly 16 complete pairs; every adapter target exists
   in the base; each delta matches the base shape; no `lora_` key in the
   output; `gpt_neox.layers.0.attention.query_key_value.weight` is still
   `[6144, 2048]` float16; the output has exactly 244 tensors with the base's
   key set.
5. Greedy-packs into shards of at most 536,870,912 B of tensor data, asserts
   the budget per shard, writes shards plus `model.safetensors.index.json`
   (`weight_map` over all 244 names, `metadata.total_size = 2090673184`).
6. Re-opens every shard and checks the on-disk key/shape set agrees with the
   index and contains no adapter tensor.

Result: 5 shards (521,220,164 / 528,761,226 / 503,583,050 / 532,914,440 /
4,194,304 bytes of tensor data), 244 tensors, 2,090,673,184 bytes total.
