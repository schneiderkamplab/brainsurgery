# T1 participant self-report

- Final artifact path: `out/T1/model.safetensors` (script: `out/T1/solution.py`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Renumbering collisions are the real hazard; avoided entirely by building a fresh
    output dict keyed by new names instead of renaming in place, plus an explicit
    collision check on every insert.
  - Non-parameter buffers (`attention.bias`, `masked_bias`, `rotary_emb.inv_freq`) are
    just block-scoped tensors here, so the generic `gpt_neox.layers.<i>.<rest>` regex
    handles them with no special casing.
  - `save_file` needs `metadata={"format": "pt"}` to match HF-style checkpoints.
- Anything in the task text or documentation that was unclear:
  - The task describes the QKV row layout in detail, but T1 never touches tensor
    values, so that detail was irrelevant here.
  - It says "a single file `out/T1/model.safetensors`" but not whether config/tokenizer
    files should be copied along; I wrote only the safetensors file, as specified.
- Tools used (condition F): n/a (condition P: Python 3.13, torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: ~3 minutes.
