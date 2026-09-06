# T2 participant self-report (condition F)

- **Final artifact path:** `out/T2/solution.py` (produces `out/T2/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none; the first execution produced the output.
- **Pitfalls or surprises you hit:**
  - `transformers.prune_heads` is not the route it looks like: `OlmoAttention` has no
    `prune_heads` implementation, and the generic `prune_linear_layer` path rebuilds
    layers with a head mask rather than giving plain contiguous slices, so I used
    safetensors directly instead.
  - The head/axis split is the whole task: q/k/v produce heads (row blocks, dim 0),
    o consumes them (column blocks, dim 1). Slicing o on dim 0 would give a checkpoint
    that loads and runs with garbage attention.
  - Not part of the required output, but worth noting for anyone loading the result:
    OLMo derives `head_dim = hidden_size // num_attention_heads`, so a config with
    `num_attention_heads=15` computes 136 and rejects the weights. The loaded config
    needs an explicit `head_dim: 128`. My load test hit this before I pinned it.
  - OLMo-1B has no per-layer norm parameters, so 16 layers x 7 tensors + embed + lm_head
    = 114 exactly; the count check is a real check, not a formality.
- **Anything in the task text or documentation that was unclear:** nothing. The task
  states the axis for each projection and the exact keep ranges; the only thing it does
  not mention is the `head_dim` config point above, which is outside the required output.
- **Tools used (condition F):**
  - `safetensors` 0.5.3 — load the two input shards and write the single output file.
    Chosen because the task is a pure tensor-slicing rewrite with a bit-exactness
    requirement; going through a model object risks dtype or layout changes.
  - `torch` 2.14.0 — `index_select` for the keep-index slicing, `contiguous()` before
    saving (safetensors rejects non-contiguous views).
  - `transformers` 5.12.1 — only to read `config.json` for hidden size / head count /
    layer count so the script derives the block boundary instead of hard-coding it, and
    (outside the solution script) to confirm the result loads as a 15-head OLMo and runs
    a finite forward pass.
  - Not used: `mergekit` (layer-level slicing and merging, no intra-tensor head slicing),
    `peft` (no adapters here), `torch-state-bridge` (renames keys; key names are unchanged).
- **Approximate time spent:** ~10 minutes.

## Checks the run enforces

`solution.py` raises `SystemExit` before writing if any of these fail:
input tensor count matches the index; each of the 64 attention tensors exists and is
`[2048, 2048]` before slicing; exactly 64 tensors were edited; the four layer-0 shapes
required by the task; the output has exactly 114 tensors and the count is unchanged from
the input; the MLP tensors were not touched. After writing, it reopens the file and
re-verifies the tensor count and the four shapes on disk.
