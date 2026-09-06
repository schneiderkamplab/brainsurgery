# Participant self-report

- Final artifact path: `out/T2/solution.py` (writes `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none — first execution succeeded
- Pitfalls or surprises you hit (one line each):
  - The row layout for the pruned head is a single contiguous 768-row block
    per the spec (rows `768*5..768*5+767` = `3840..4607`), so the qkv weight
    and bias can be sliced identically without needing to interleave/reorder
    q, k, v separately; got this right on the first pass by reading the spec
    carefully before writing code.
- Anything in the task text or documentation that was unclear: none; the
  "keep rows/columns" ranges in "Required result" were explicit enough to
  code directly and cross-check against the derived slice boundaries.
- Tools used (condition F): name, version, and why:
  - `torch` 2.14.0 and `safetensors` 0.5.3 only. The transform is a pure,
    fully-specified tensor-layout edit (contiguous row/column slices with
    known boundaries), so a direct script using `safe_open`/`save_file` is
    simpler and more auditable than routing through `transformers`'
    `prune_heads` (which assumes a generic, non-fused attention layout and
    doesn't know GPT-NeoX's fused/interleaved `query_key_value` block
    structure) or `mergekit`/`torch-state-bridge` (built for cross-checkpoint
    merging and key rewriting, not intra-tensor slicing). Plain `torch`
    slicing plus explicit shape asserts before writing gave the required
    "fail loudly" checks directly.
- Approximate time spent, if you can tell: ~10 minutes
