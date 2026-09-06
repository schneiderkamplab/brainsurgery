# T2 report (condition F)

## Participant self-report

- **Tools used**: `safetensors` (load_file/save_file) and `torch`
  (index_select) directly, in a plain Python script (`out/T2/solution.py`).
  No mergekit/transformers/peft/torch-state-bridge.
- **Why**: the spec pins down exact row/column block boundaries and ordering
  (interleaved 768-row qkv blocks, 256-column dense blocks, drop head 5,
  keep the rest in original order). `transformers.prune_heads` prunes heads
  on a live `nn.Module` and doesn't expose a guarantee about matching this
  exact raw-tensor block layout, so I couldn't easily verify it against the
  spec bit-for-bit. mergekit's passthrough/task-arithmetic operate at the
  layer or model level, not sub-tensor column/row slices. A direct
  `index_select` over the loaded state dict is easy to reason about and to
  check against the spec directly (I recomputed the expected slices with
  `torch.cat` and compared with `torch.equal` before finishing).
- **Approach**: for each of the 16 layers, build a "keep" row index for
  `query_key_value.{weight,bias}` (rows `0..3839` + `4608..6143`) and a
  "keep" column index for `dense.weight` (columns `0..1279` + `1536..2047`),
  then `index_select` and `.contiguous()`. All other tensors are copied
  unchanged. Required checks (shapes on layer 0's three tensors, and total
  tensor count == 244) run and `assert` before the file is written, so a
  regression fails loudly instead of silently writing a bad checkpoint.
- **Attempts**: 1 execution, succeeded on the first run
  (`executions_until_first_success = 1`, `first_execution_success = yes`).
- **Verification beyond the required checks**: after writing, I independently
  recomputed the expected sliced tensors for layer 0 from the raw input and
  compared with `torch.equal`, confirming both the row/column selection and
  ordering match the spec exactly.
- **Output**: `out/T2/model.safetensors`, 244 tensors, dtypes preserved
  (float16), pruned tensors at `[5376, 2048]`, `[5376]`, `[2048, 1792]` as
  required.
