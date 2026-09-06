# T5 participant self-report

- **Final artifact path:** `out/T5/solution.py` (produces `out/T5/model-*.safetensors` and `out/T5/model.safetensors.index.json`).

- **Number of times you executed the script or plan:** 2.

- **Which executions failed, and why (one line each):**
  1. Did not fail, but produced output that violated a stated requirement: the
     first version used plain size-based greedy bin-packing (512 MiB cap),
     which is mathematically correct for the general rule but does not put
     `gpt_neox.embed_in.weight` / `embed_out.weight` alone in their own shard,
     as the task text requires by name (each is ~196.5 MiB, under the cap, so
     no greedy packer isolates them on general grounds -- see below). Caught
     by manual inspection before considering the task done, not by a crash.

- **Pitfalls or surprises you hit (one line each):**
  - The task states embedding tensors are isolated because they are "larger
    than" the 512 MiB (536,870,912-byte) shard cap, but they are actually
    ~206,045,184 bytes (~196.5 MiB) each, well under the cap; I verified this
    against the actual file and against the real bin-packing algorithms in
    `huggingface_hub.split_torch_state_dict_into_shards` and mergekit's
    `TensorWriter` (both only isolate a tensor that itself exceeds
    `max_shard_size`), under both alphabetical and natural model-definition
    tensor order -- none isolate these two tensors on size grounds alone.
    I resolved this by treating the named requirement as authoritative and
    isolating `gpt_neox.embed_in.weight` and `embed_out.weight` explicitly,
    since the spec names them and their measured byte size matches exactly;
    everything else uses generic greedy packing under the 512 MiB cap. This
    is documented inline in `solution.py`.
  - PEFT's key prefix (`base_model.model.`) and the `.lora_A.weight` /
    `.lora_B.weight` suffixes have to be stripped in the right order to
    recover the base tensor name; got this right on the first try because the
    example key was given in full in TASK.md.
  - The `attention.bias` / `attention.masked_bias` buffers (a U8 causal mask
    and a scalar) are not adapted and must pass through unchanged -- easy to
    overlook since they live right next to `query_key_value.weight` in the
    same submodule but aren't in `target_modules`.

- **Anything in the task text or documentation that was unclear:** Yes -- see
  the shard-isolation pitfall above: the stated justification ("larger than
  [512 MiB]") doesn't hold arithmetically for the 206 MB embedding tensors,
  even though the named tensors and their exact byte count do match the
  actual checkpoint. Worth tightening: either give the real threshold that
  triggers isolation in the reference implementation, or state the rule as
  "these two tensors are always isolated" without tying it to the 512 MiB
  cap.

- **Tools used (condition F): name, version, and why:** Plain Python script
  on top of `torch` 2.14.0 and `safetensors` 0.5.3 only (both from
  `requirements-F.txt`). No `peft` (`merge_and_unload` requires instantiating
  the full HF model just to get a state dict back, then still needs the same
  custom sharding logic afterward -- it adds a dependency and a load step
  without removing any of the actual work); no `mergekit` (its task-arithmetic
  merge configs assume the two checkpoints share overlapping key spaces,
  which a PEFT adapter's key names deliberately don't -- read on-checkpoint
  tensor manipulation and its own `TensorWriter` sharding directly, but a raw
  script gives the same result with less indirection); no
  `torch-state-bridge` (built for renaming/regex key rewriting, not for
  reading two safetensors files, computing `B @ A`, and merging into one).

- **Approximate time spent, if you can tell:** Short: reading the two input
  files' key layouts, writing the merge + shard script, one run that revealed
  the shard-isolation issue, one fix, then verification (independent
  recomputation of two merged layers' weights, bit-exact check against
  unmerged tensors, shard budget/isolation check).
