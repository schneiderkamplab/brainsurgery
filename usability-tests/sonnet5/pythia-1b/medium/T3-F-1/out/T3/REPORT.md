# Participant self-report — T3 (condition F)

- **Final artifact path:** `out/T3/solution.py` (writes `out/T3/*.safetensors` +
  `out/T3/model.safetensors.index.json`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none; the single execution succeeded.
- **Pitfalls or surprises you hit:**
  - `attention.masked_bias` is stored as `F16` in this checkpoint, not the
    scalar-buffer dtype the task blurb implies for the other two buffers —
    doesn't change the fix (still dropped by name), but worth noting when
    writing the buffer regex so it doesn't get mistaken for a value that
    needs casting.
  - The "206 MB each" figure for `embed_in`/`embed_out` in the task text is
    the float16 input size; after the required float32 upcast they're
    ~393 MiB each, well over the 256 MiB shard budget either way, so they
    still land alone in their own shard — just wanted to flag the number
    doesn't match the output dtype so the next reader isn't confused
    double-checking shard sizes against it.
- **Anything in the task text or documentation that was unclear:** No —
  spec, buffer list, and dtype rules were unambiguous once cross-checked
  against the actual tensor list in the checkpoint.
- **Tools used (condition F):** `torch` 2.14.0 and `safetensors` 0.5.3 only,
  via a plain script. Chose this over `mergekit`/`transformers` dtype-export
  paths because the task needs per-tensor dtype control (bf16 for exactly 64
  named projection matrices, f32 for everything else) plus a custom
  byte-budget shard packer with a hard-coded fallback for oversized tensors —
  none of the higher-level tools in F-allowed.md expose that combination
  directly, whereas `safe_open`/`save_file` give exact control with little
  code.
- **Approximate time spent:** ~10 minutes (inspect checkpoint keys, write
  script with built-in assertions, run once, verify output).
