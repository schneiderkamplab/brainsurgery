# T3 self-report (condition P, OLMo-1B-0724-hf)

- **Final artifact path:** `out/T3/solution.py` (output checkpoint in `out/T3/`:
  10 shard files `model-0000N-of-00010.safetensors` + `model.safetensors.index.json`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none; the single execution passed.
- **Pitfalls or surprises you hit (one line each):**
  - The obvious `.*weight` / "cast every 2-D weight" targeting would also hit
    `model.embed_tokens.weight` and `lm_head.weight`, so I built the 112 names by
    construction (16 layer indices read from the checkpoint x 7 named suffixes) and
    asserted all 112 exist, rather than matching a pattern.
  - Both embedding matrices are 412 MB, i.e. larger than the whole 256 MiB shard
    budget, so the greedy packer needs an explicit "oversized tensor gets its own
    shard" branch before the accumulate-and-flush branch, otherwise it emits an
    over-budget shard or an empty one.
  - Shard sizes are computed from the *output* dtype, not the input dtype; a layer is
    exactly 128 MiB in bf16, so two layers pack to exactly 256 MiB and the budget
    comparison has to be `>` (inclusive "at most"), not `>=`, or every shard would
    hold one layer and the layout would be wrong.
  - The checkpoint is ~5 GB in float32, so I streamed it: shapes/dtypes are read via
    `get_slice` without materialising data, and tensors are loaded one shard at a
    time, keeping peak memory near one shard (~412 MB) instead of the full model.
  - `.contiguous().clone()` before `save_file` to avoid safetensors' shared-storage
    rejection on tensors sliced out of a memory-mapped shard.
- **Anything in the task text or documentation that was unclear:**
  - The shard *ordering* is unspecified. The stated rules are properties (every name
    mapped, <= 256 MiB per shard, oversized tensors alone), so I iterated the input
    index's `weight_map` order and packed greedily; a different but equally valid
    tensor order would produce a different name-to-shard assignment.
  - Shard file naming is not specified either; I used the HF convention
    `model-<i>-of-<n>.safetensors`.
  - Whether the index's `metadata.total_size` is required, and whether the tokenizer /
    config files should be copied alongside, is not stated. I wrote `total_size`
    (2,971,664,384 bytes, verified against the tensors on disk) and did not copy the
    config/tokenizer files, since the task only asks for shards plus index.
  - Items 2-4 of "Required result" (no buffers to delete, no norms/biases, names
    unchanged) are no-ops for this checkpoint; I confirmed the input is 114 tensors,
    all float32, with no non-parameter buffers.
- **Tools used (condition F):** n/a (condition P: torch 2.14.0, safetensors 0.5.3 only).
- **Approximate time spent, if you can tell:** ~10 minutes, of which 15 s was the run.
