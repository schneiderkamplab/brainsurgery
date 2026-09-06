# T2 (Pythia-1B, condition F) — participant self-report

- Final artifact path: `out/T2/solution.py` (output: `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; execution 1 succeeded.
- Pitfalls or surprises you hit (one line each):
  - `gpt_neox.layers.<i>.attention.bias` (the U8 causal-mask buffer) shares the
    `attention.` prefix with the head-bearing tensors; matched on the exact
    suffix (`attention.query_key_value.weight/.bias`, `attention.dense.weight`)
    rather than a prefix or regex to avoid touching it.
  - Post-hoc load test with `transformers` 5.12.1 failed at config
    construction: `GPTNeoXConfig` validates `hidden_size % num_attention_heads == 0`,
    so a 7-head, 2048-hidden model cannot be instantiated through HF even though
    the checkpoint itself is correct. This is a library constraint, not a
    checkpoint defect; the task's "loadable with 7 heads" claim holds only for
    an implementation that takes an explicit head_dim. Not needed for grading.
  - Transformers' `prune_heads` route (suggested in `F-allowed.md`) was
    deliberately not used: it requires instantiating the HF model, rewrites the
    Linear modules in float32 by default, and would need a separate re-export,
    which risks dtype and bit-exactness drift under bit-exact grading.
- Anything in the task text or documentation that was unclear: nothing; the
  row/column ranges and the interleaved per-head q|k|v layout were spelled out.
  Minor: the task says the result "must be loadable as the same architecture
  with 7 heads", which the pinned `transformers` version cannot do (see above).
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3 — `safe_open` to read, `save_file` to write; preserves
    dtype and the `{'format': 'pt'}` metadata, keeps everything bit-exact.
  - `torch` 2.14.0 — index-select slicing of the head blocks in float16 without
    any conversion; `torch.equal` for the built-in row/column-order checks.
  - `transformers` 5.12.1 — attempted only for a post-hoc loadability check
    (failed at config validation as described); not part of the solution.
- Approximate time spent, if you can tell: about 5 minutes (one header
  inspection, one script, one run of ~4 s, one independent verification pass
  confirming all 244 keys, dtypes, and bit-equality of the 196 untouched
  tensors against the input).
