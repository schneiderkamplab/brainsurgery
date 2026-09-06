# T2 — Participant self-report

- **Task / condition / repeat:** T2 (structured attention-head pruning, Pythia-1B) / F (free choice of tooling) / 1
- **Outcome:** solved; `out/T2/model.safetensors` written with 244 tensors.
- **Executions:** 1 (`.venv/bin/python out/T2/solution.py`). Retries: 0. First execution succeeded.
- **Failed executions:** none.
- **Artifact:** `out/T2/solution.py`, ~50 non-blank, non-comment lines.

## Tools used and why

| Tool | Version | Why |
|---|---|---|
| `safetensors` | 0.5.3 | direct `safe_open` / `save_file` — the task is a pure tensor-slice rewrite of one file, and this keeps key set, order and float16 dtype bit-exact |
| `torch` | 2.14.0 | `index_select` on the keep-index lists, `.contiguous()` before saving |

Rejected routes:

- **`transformers.prune_heads`** (the route the condition sheet suggests). It
  prunes at the module level on a loaded `GPTNeoXForCausalLM` and rewrites the
  fused `query_key_value` through its own index bookkeeping; it also updates
  `config.num_attention_heads` and re-saves the whole model. That gives no
  control over the exact row order the task specifies, risks a dtype or
  key-set change on `save_pretrained`, and materializes the model in memory
  for what is a 48-tensor slice. Bit-exact grading makes an opaque helper the
  wrong bet.
- **mergekit** — expresses layer-level slicing, not intra-tensor head slices.
- **torch-state-bridge** — rewrites keys, and no key changes here.

## What I had to get right

- The fused projection is GPT-NeoX **interleaved**: head `h` owns rows
  `768h..768h+767` as `[q|k|v]` *within* the head, not three global `[q|k|v]`
  segments. Pruning head 5 is therefore a single contiguous 768-row cut at
  `3840..4607` — not three separate 256-row cuts. Treating it as `[q|k|v]`
  segments would have produced a loadable checkpoint with garbage attention.
- `attention.dense` is `[out, in]` (`nn.Linear`), so heads are **column**
  blocks of 256 — the cut is on dim 1 at `1280..1535`, a different width and
  a different axis from the qkv cut.
- `attention.dense.bias`, the three attention buffers and all MLP tensors are
  not head-bearing and were left untouched; the tensor count stays 244.
- Preserved the file's safetensors metadata and float16 dtype; sliced views
  were made `.contiguous()` so `save_file` accepts them.

## Checks enforced in the run

The script raises before writing if any of: the three layer-0 shapes are not
`[5376, 2048]`, `[5376]`, `[2048, 1792]`; the tensor count is not 244; the
count changed relative to the input; or an expected per-layer tensor name is
missing.

Post-run spot check (separate from the script): key sets identical to the
input, dtype `float16`, and layer 7's three edited tensors bit-equal to the
independently recomputed `cat` of the kept slices, with `dense.bias` unchanged.
