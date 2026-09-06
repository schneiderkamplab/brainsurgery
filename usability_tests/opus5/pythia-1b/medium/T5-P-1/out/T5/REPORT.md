# T5 self-report (condition P)

- **Final artifact path:** `out/T5/solution.py` (output checkpoint in `out/T5/`:
  4 shards + `model.safetensors.index.json`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none; the single execution passed all
  four required checks and the post-write disk verification.
- **Pitfalls or surprises you hit:**
  - The task text says the 206 MB embedding tensors are "larger than" the
    512 MiB shard budget and must sit alone; they are actually 206,045,184
    bytes ≈ 196 MiB, i.e. well under the budget. I followed the operative rule
    (no shard over 536,870,912 bytes of tensor data) with greedy packing in
    checkpoint key order, so the embeddings share shard 1 with other tensors.
  - Adapter names carry the PEFT `base_model.model.` prefix and a
    `.lora_A/.lora_B` infix; the base name is the stem plus `.weight`.
  - `fan_in_fan_out = false` with both base and factors in `[out, in]` layout
    means `B @ A` is added directly — no transpose. Getting this backwards
    would be a silent shape error only because `[6144, 2048]` is not square.
  - Merging must happen in float32 before the cast back to float16; adding the
    delta in float16 loses accuracy against the 1e-3 Frobenius tolerance.
  - Sizes for the shard plan are computed from the safetensors header
    (`get_slice().get_shape()/get_dtype()`) rather than by materialising every
    tensor, so the plan is made before any 2 GB of data is loaded.
  - `attention.masked_bias` is a rank-0 scalar tensor; it round-trips fine but
    is easy to trip over if code assumes at least one dimension.
- **Anything in the task text or documentation that was unclear:**
  - The oversized-tensor claim above contradicts the actual file (see first
    pitfall). It left the intended shard boundaries ambiguous: a reference that
    isolates the embeddings would produce a different shard assignment than
    greedy packing, even though both respect the stated 512 MiB rule.
  - The task does not say whether the index should carry a
    `metadata.total_size` field or which shard filename pattern is expected; I
    used the HuggingFace conventions (`model-000ii-of-000nn.safetensors`,
    `metadata.total_size` = sum of tensor bytes).
- **Tools used (condition F):** n/a (condition P: torch 2.14.0,
  safetensors 0.5.3 only).
- **Approximate time spent:** ~10 minutes.
