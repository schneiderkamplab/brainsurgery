# T5 self-report (condition B: BrainSurgery plan)

- **Final artifact path:** `out/T5/plan.yaml` (output checkpoint in `out/T5/`:
  4 shards + `model.safetensors.index.json`, 244 tensors).

- **Number of times you executed the script or plan:** 1
  (`brainsurgery out/T5/plan.yaml`, passed on the first run). Two additional
  runs used throwaway plans under `tmp/`, not the task plan: one `dump` to read
  the adapter's tensor names, one `diff`/`assert` plan to verify the result
  afterwards.

- **Which executions failed, and why:** none.

- **Pitfalls or surprises you hit:**
  - Output alias inference: with two inputs, every write has to land on one
    alias, so the adapter factors had to be `copy`-ed onto `base::` first
    rather than computing the delta on the `lora` alias and adding across.
  - `assert: count` raises "matched zero tensors" when the pattern matches
    nothing, so "no `lora_` in the output" cannot be written as `count: 0`;
    `not: { exists: ... }` is the form that tolerates an empty match. I probed
    this on a scratch plan rather than burning an attempt on the real one.
  - `cast_` takes the dtype under the key `to` (not `dtype` as `cast` does).
  - Intermediates are ordinary tensors on the alias, so they must be `delete`-d
    before the output is written; the checks for that have to sit after the
    delete but still inside the plan, which the transform ordering makes easy.
  - Keeping the float32 accumulation meant a second temporary (`W32` = base
    weight cast up), then `cast_` back to float16 and `assign` into the
    original slot. `assign` (not `move`) is what preserves the tensor's
    position in the state dict, which the shard packing order depends on.
  - `fan_in_fan_out = false` plus both sides in `[out, in]` layout meant no
    transpose: `matmul` with `from_a = lora_B`, `from_b = lora_A` directly.
  - Shard budget: `512MB` in the tool is binary (536,870,912 bytes of tensor
    data), which is exactly the task's limit, so no conversion was needed.

- **Anything in the task text or documentation that was unclear:**
  - The task says a tensor larger than 512 MiB "is stored alone in its own
    shard" and then names `embed_in`/`embed_out` at 206 MB each, which are not
    over the budget. Reading it as an illustration of the general rule, not a
    requirement that those two be alone, matched what the tool produces.
  - The docs say ternary transforms (`matmul`, `add`) support "capture-based
    rewrite across `from_a`, `from_b`, and `to`" without spelling out that
    `from_a` is the matched pattern and the other two are replacement
    templates; the `add` example (`from_a: '.*.weight'`, `from_b: '.*.delta'`)
    reads as if `from_b` were a pattern too. It behaves as a rewrite.

- **Tools used (condition F):** n/a (condition B).

- **Approximate time spent:** roughly 10 minutes, most of it reading
  `help.txt` for exact key names and checking the empty-match assert semantics.

## Verification performed

- Plan-internal checks (all before the write): 16 `lora_A` and 16 `lora_B`
  matched, factor shapes `[16,2048]`/`[6144,16]`, 16 deltas produced with shape
  `[6144,2048]` and dtype float32, no `lora_`/temporary tensor left,
  `gpt_neox.layers.0.attention.query_key_value.weight` still `[6144, 2048]` and
  float16, 244 tensors total.
- Post-hoc, with a scratch plan (`diff` + `assert`): `orig` vs `out` differ in
  exactly the 16 `query_key_value.weight` tensors and nothing else (no missing
  names on either side, other 228 bit-identical); an independently recomputed
  `(W.float() + 2 * B @ A).half()` is bit-identical to what the shards hold.
