# T3 self-report (condition B, Pythia-1B)

- **Final artifact path:** `out/T3/plan.yaml` (output written to `out/T3/`:
  9 shards + `model.safetensors.index.json`)

- **Number of times you executed the script or plan:** 1

- **Which executions failed, and why (one line each):** none; the single
  execution passed all asserts and wrote the output.

- **Pitfalls or surprises you hit (one line each):**
  - The obvious hazard is regex overreach: `.*weight` would also catch
    `gpt_neox.embed_in.weight`, `embed_out.weight` and every layer-norm
    weight, so the bfloat16 target is spelled out as
    `gpt_neox\.layers\.\d+\.(attention\.(query_key_value|dense)|mlp\.(dense_h_to_4h|dense_4h_to_h))\.weight`
    with escaped dots, guarded by a `count: is: 64` before the cast.
  - `assert` has no way to select tensors *by* dtype, so "exactly 64 tensors
    are bfloat16" had to be expressed as two halves: the 64-name pattern is
    bfloat16, and the complement (a negative-lookahead regex, 132 tensors) is
    float32. Getting the lookahead right needed a trailing `$` inside it,
    because references are full-matched.
  - Deleting the buffers first and only then doing `cast_: {target: '.*', to:
    float32}` avoids having to exclude the uint8 causal-mask buffer from the
    blanket upcast.
  - The task text sizes `embed_in`/`embed_out` at 206 MB (their float16 input
    size); in the float32 output they are 412 MB each, so each still lands
    alone in its own shard — the shard budget applies after the casts.
  - Shard units: the README states `KB/MB/GB` are binary, so `shard: 256MB`
    is exactly the 268,435,456-byte budget the task asks for. Verified from
    the written headers: max non-solo shard is 260,317,184 data bytes.
  - Order matters for `cast_`: fp16 -> fp32 -> bf16 was chosen because the
    task confirms it is bit-identical to a direct fp16 -> bf16 cast.

- **Anything in the task text or documentation that was unclear:**
  - The docs do not say whether `assert: {dtype: ...}` applies to *every*
    tensor matched by a pattern or only the first; behaviour on the 64-name
    pattern suggests all of them, but it is worth stating explicitly.
  - Nothing documents a dtype-based tensor selector, which is what a check
    like "exactly N tensors are bfloat16" naturally wants.
  - `output.shard` vs `--shard-size` precedence is documented, but it is not
    stated that the budget is evaluated on post-transform dtypes (it is).

- **Tools used (condition F):** n/a (condition B).

- **Approximate time spent, if you can tell:** ~10 minutes, most of it
  reading `docpack/help.txt` and dumping the input safetensors header with
  `dd`/`grep` to confirm the exact key set; the plan itself ran once in ~9 s.
