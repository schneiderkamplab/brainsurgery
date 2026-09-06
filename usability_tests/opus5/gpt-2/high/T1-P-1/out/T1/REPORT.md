# Participant self-report — T1 (condition P)

- **Final artifact path:** `out/T1/solution.py` (output: `out/T1/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none; the single execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - Renumbering collision hazard: avoided by writing into a fresh dict keyed by the new
    names instead of renaming in place, plus an explicit collision check on each insert.
  - The block-index regex has to be anchored (`^h\.(\d+)\.`) with escaped dots so it
    matches only the leading block index and never a digit elsewhere in a name.
  - Blocks own 13 tensors each, including the non-parameter causal-mask buffer
    `attn.bias`; it has to move with the block like everything else (12*13 + 4 = 160).
  - `save_file` needs `metadata={"format": "pt"}` to match the usual HF-written header;
    tensors loaded via `load_file` are already contiguous and unshared, so no clone needed.
- **Anything in the task text or documentation that was unclear:**
  - Nothing blocking. The keys have no `transformer.` prefix, which the task states; worth
    noting since the HF `GPT2LMHeadModel` state dict does carry that prefix.
  - The Conv1D `[in, out]` layout note is informational here — T1 is a pure rename, no
    tensor is transposed or reshaped.
- **Tools used (condition F):** n/a (condition P: Python 3.13 + torch 2.14.0 + safetensors 0.5.3).
- **Approximate time spent, if you can tell:** ~5 minutes.

## What the script checks (fails loudly, before writing anything)

- Input has exactly blocks 0..11 and all of 2, 5, 8 are present.
- No destination key collides during renumbering.
- No tensor of blocks 9, 10, 11 remains, and output block indices are exactly 0..8.
- Exactly 9 tensors match `h.<i>.attn.c_attn.weight`; every block has 13 tensors.
- Output has exactly 121 tensors.
- Every output tensor is bit-identical (`torch.equal`), same shape and dtype, to its
  source tensor under the old name.
- After writing, the file is re-read and the key set, shapes, dtypes and values are
  re-verified against what was intended.
