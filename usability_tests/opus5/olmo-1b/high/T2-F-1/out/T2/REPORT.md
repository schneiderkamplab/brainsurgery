# T2 participant self-report (condition F)

- **Final artifact path:** `out/T2/solution.py` (driver: `out/T2/run.sh`); output `out/T2/model.safetensors`
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none — the first execution passed all checks and wrote the output.
- **Pitfalls or surprises you hit:**
  - The o_proj asymmetry is the only real trap: q/k/v are sliced on dim 0 (rows,
    the head-concat axis of the output) but o_proj on dim 1, because it *consumes*
    the concatenated head outputs on its input axis. Slicing o_proj by rows would
    still produce a loadable checkpoint with garbage attention.
  - OLMo-1B-0724 is MHA (`num_key_value_heads == num_attention_heads == 16`), so
    k/v are pruned identically to q. With GQA the block boundary for k/v would be
    a different head index and this script would be wrong; I checked the config
    rather than assuming.
  - Nothing is fused here (separate q/k/v tensors), so no interleaved-QKV
    unpacking was needed.
  - `config.json` is not part of the required output (spec: a single
    `model.safetensors`), but the pruned checkpoint is only *loadable* as
    15 heads if a config sets `num_attention_heads: 15` **and** an explicit
    `head_dim: 128` — 2048/15 is not an integer, so a naive config edit would
    fail. I left the config alone as the spec asks; flagging it as a real gap
    between "required result" and "loadable as the same architecture".
- **Anything in the task text or documentation that was unclear:**
  - Item 6 says one file with 114 tensors, while the objective says the result
    "must be loadable as the same architecture with 15 heads per layer". Those
    pull in different directions (see the `head_dim` point above); I followed
    item 6 literally since grading compares `out/T2` tensor-by-tensor.
  - Otherwise the spec was fully explicit — the exact kept-row ranges were given,
    so there was no layout ambiguity left to discover.
- **Tools used (condition F):**
  - `safetensors` 0.5.3 — streamed the two input shards via `safe_open` and wrote
    the single unsharded output with `save_file`.
  - `torch` 2.14.0+cu130 — `torch.cat` for the slice-and-reassemble, `torch.equal`
    for bit-exact verification.
  - Read `transformers` 5.12.1 config only (to confirm head count / MHA); did not
    use it to build the output.
  - **Why a plain script over `transformers.prune_heads`:** `prune_heads` would
    have required materialising a 5 GB `OlmoForCausalLM`, it re-derives its own
    keep-index (so the spec's exact row ranges would be assumed rather than
    asserted), pruning support is not uniformly maintained across models in
    transformers 5.x, and `save_pretrained` re-shards and rewrites the layout
    instead of emitting the single required file. The transform itself is one
    index slice per tensor; a script makes both the block boundaries and the
    required checks explicit and auditable. mergekit and torch-state-bridge
    operate on layers and key names respectively, not on intra-tensor axes, so
    neither can express this edit.
- **How the required checks are enforced:** `solution.py` raises `AssertionError`
  (no fallbacks) before writing if layer-0 q/k/v are not `[1920, 2048]`, o_proj is
  not `[2048, 1920]`, or the tensor count is not 114. It additionally asserts the
  same shapes for all 16 layers, an unchanged key set and dtypes, byte-identical
  contents for every non-head-bearing tensor, and that each kept slice is
  bit-equal to the correct source range in the correct order. After writing it
  re-opens the file and re-verifies keys, shapes, dtypes and values.
- **Approximate time spent:** ~5 minutes; the run itself takes ~20 s.
