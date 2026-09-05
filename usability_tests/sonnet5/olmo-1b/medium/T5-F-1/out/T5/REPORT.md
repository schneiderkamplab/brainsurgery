# T5 report

## Participant self-report

- Final artifact path: `out/T5/solution.py`
- Number of times you executed the script or plan: 2 (first run produced a
  valid checkpoint but packed `model.embed_tokens.weight` into a shard with
  another tensor instead of giving it its own shard; second run, after fixing
  that, succeeded)
- Which executions failed, and why (one line each):
  - Execution 1 did not crash or fail an assertion, but on inspection it
    violated the sharding spec: it only forced tensors that literally exceed
    536,870,912 bytes into solo shards, and `model.embed_tokens.weight` /
    `lm_head.weight` (412,090,368 bytes each) are under that cap, so the
    generic bin-packer happily merged `model.embed_tokens.weight` into a
    shard with `model.layers.0.mlp.down_proj.weight`. Caught by manually
    listing `weight_map` and comparing to the spec's explicit statement that
    those two tensors must be alone.
- Pitfalls or surprises you hit (one line each):
  - TASK.md says these two tensors are "larger than that limit" and must be
    solo-sharded, but at fp32 they are 412 MB (390 MiB), which is smaller
    than the 512 MiB (536,870,912-byte) shard cap; the rule as written isn't
    actually about size, it names the two tensors explicitly, so I hardcoded
    them as always-alone rather than relying purely on a size threshold.
  - Everything else in the spec matched the data exactly (r=16, alpha=32,
    scale=2, `fan_in_fan_out=false`, `[out, in]` layout, 32 pairs across
    16 layers x {q_proj, v_proj}), no LoRA-side surprises.
- Anything in the task text or documentation that was unclear:
  - The "larger than that limit" wording for the two 412 MB tensors is
    inconsistent with the stated 512 MiB shard cap; it reads as a
    size-triggered rule but is actually a named-tensor rule for this
    checkpoint. Worth rephrasing to "these two tensors are always given
    their own shard" rather than tying it to the size comparison.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3 — reading the sharded base checkpoint and the
    adapter file, writing the sharded output. Chosen because it gives direct
    control over exactly which tensors land in which shard file and lets me
    assert the byte-budget per shard precisely; the task's per-tensor,
    per-shard byte constraints (and the explicit two-tensor exception) are
    easier to guarantee with an index/weight_map built by hand than by
    trusting a higher-level `save_pretrained(..., max_shard_size=...)` call.
  - `torch` 2.14.0 — the actual `float32` matmul/add (`scale * B @ A`) and
    tensor introspection (`numel`, `element_size`, shape/dtype checks).
  - Considered `peft.merge_and_unload`, but it requires instantiating the
    full HF `AutoModelForCausalLM` (loading the whole model into a
    transformers module graph) just to merge two `nn.Linear`-shaped LoRA
    factors per targeted layer, and then re-extracting a state dict for
    sharded export gives no control over the exact byte-threshold packing
    and the embed/lm_head exception this task requires. A direct script
    over raw safetensors state dicts was more direct and let every
    "Required check" be asserted explicitly in code before anything is
    written.
- Approximate time spent, if you can tell: ~15 minutes, most of it re-reading
  the sharding spec and reconciling the "412 MB … larger than that limit"
  wording against the stated 512 MiB cap.
