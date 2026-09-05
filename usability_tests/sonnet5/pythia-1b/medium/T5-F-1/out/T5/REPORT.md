# Participant self-report — T5 (condition F)

- **Final artifact path:** `out/T5/solution.py` (invoked via `out/T5/run.sh`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none; the single execution succeeded.
- **Pitfalls or surprises you hit:**
  - The adapter tensor names carry a `base_model.model.` prefix and split
    `lora_A`/`lora_B` into separate keys per module; had to group them by
    stripping the prefix and suffix before pairing.
  - Naive shard packing (pack a tensor into the current shard, then check if
    the running total overflowed) would have merged an oversized tensor's
    bytes into the shard before it, corrupting the "oversized tensor gets its
    own shard" rule — checked tensor size against the budget *before* adding
    it, not after.
  - `fan_in_fan_out = false` and the task statement both confirm `B @ A`
    without transposition, since both the base and the adapter factors use
    the `nn.Linear` `[out, in]` layout — no ambiguity here, but worth
    double-checking since PEFT's `fan_in_fan_out` flag exists specifically to
    flip this convention for `Conv1D`-style weights.
- **Anything in the task text or documentation that was unclear:** no.
- **Tools used (condition F):** `safetensors` 0.5.3 (direct load/save,
  `safe_open` for streaming reads) and `torch` 2.14.0 (float32 matmul and
  dtype casts) only. Considered `peft.merge_and_unload`, but that requires
  instantiating the full `GPTNeoXForCausalLM` and its tokenizer just to fold
  two small factors into 16 linear layers, and PEFT's own sharded save is a
  thin wrapper over `transformers.save_pretrained` with no direct control
  over the shard-membership rule the task specifies (oversized tensor alone
  in its shard). A ~180-line script against the safetensors file format
  directly was simpler, more auditable, and let me implement the exact
  sequential shard-packing rule instead of trusting a library default.
- **Approximate time spent, if you can tell:** ~10 minutes of tool time
  (single pass, no debugging needed beyond writing and verifying the script).
