# T1 participant self-report (condition F)

- **Task / condition / repeat**: T1 (depth pruning + renumbering), GPT-2 (124M), condition F, repeat 1.
- **Approach**: a single script, `out/T1/solution.py`, run once.
- **Tools used**:
  - `safetensors` 0.5.3 (`safe_open`, `save_file`) — direct, lossless tensor-level
    read/write; dtypes, shapes and bit values pass through untouched.
  - `torch` 2.14.0 — only as the tensor framework behind safetensors.
  - Considered and rejected: `mergekit` passthrough slicing (it rebuilds a model
    through transformers and would require a config edit and a full re-serialize,
    with no guarantee of bit-exactness for the `attn.bias` mask buffers);
    `torch-state-bridge` (key rewriting alone still leaves persistence to me, so
    it would add a dependency without removing any work).
- **Executions**: 1. First execution succeeded; 0 retries, 0 failed executions.
- **How the required checks are enforced**: all checks run in-process *before*
  `save_file`, and each calls `fail()` which prints to stderr and exits 1, so a
  violation means non-zero exit and no output file written. Checks:
  1. no key matching `h.9|10|11.*` remains;
  2. exactly 9 keys match `h.<i>.attn.c_attn.weight`, and their indices are exactly 0..8;
  3. the output dict has exactly 121 tensors;
  4. extra guard: a renumbered key colliding with an already-emitted key aborts.
- **Collision hazard**: avoided structurally rather than by ordering — the script
  builds a *new* dict keyed by the remapped name (`old -> new` from
  `[0,1,3,4,6,7,9,10,11] -> 0..8`) instead of renaming in place, so no surviving
  block can be overwritten; the collision check would still catch a bad map.
- **Pitfalls encountered**: none that cost an attempt. Points I deliberately
  handled: the block regex is anchored (`^h\.(\d+)\.`) with escaped dots so it
  cannot touch `wte`/`wpe`/`ln_f` or overreach into `mlp.c_proj`; the `attn.bias`
  mask buffer is treated as an ordinary block tensor (13 per block, so 9*13+4=121);
  tensors are `.contiguous()` on save to avoid safetensors rejecting views.
- **Verification beyond the required checks**: spot-compared the source and output
  tensors for every surviving block (`ln_1.weight`, `attn.c_attn.weight`,
  `attn.bias`, `mlp.c_proj.bias`) and all 4 non-block tensors with
  `torch.equal` — all bit-identical. Conv1D `[in, out]` layout is irrelevant here
  since no tensor is reshaped or transposed.
- **Result**: `out/T1/model.safetensors`, 121 tensors, 9 blocks indexed 0..8.
  `grade.py` lives outside the sandbox, so I did not run it.
