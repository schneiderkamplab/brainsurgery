# T1 self-report (condition P)

- Final artifact path: `out/T1/model.safetensors` (script: `out/T1/solution.py`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Renumbering collision hazard: avoided by building a fresh output dict keyed by
    the new name instead of renaming in place, plus an explicit collision check.
  - The three non-parameter buffers (`attention.bias`, `attention.masked_bias`,
    `attention.rotary_emb.inv_freq`) must be carried along too; a naive
    "parameters only" filter would drop them and land on 136 tensors, not 184.
  - `save_file` needs `metadata={"format": "pt"}` for the file to be loadable by
    HF loaders; easy to forget in a one-off script.
- Anything in the task text or documentation that was unclear:
  - The task describes the fused QKV row layout in detail, but T1 never slices it;
    that paragraph is not needed for this task.
  - Not stated whether the 12-layer `config.json`/tokenizer should also be written
    to `out/T1`; I took "a single file" literally and wrote only the safetensors.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~4 minutes
