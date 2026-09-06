# T4 participant self-report (condition F, GPT-2 124M)

- Final artifact path: `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - None material. The key names have no `transformer.` prefix (`h.<i>.mlp.c_fc.weight` etc.), confirmed by listing the base checkpoint before writing the script.
  - Tensor names were selected by an explicit 48-name allowlist rather than a regex, so `attn.c_proj` cannot be over-matched.
  - After the successful run I reworded one comment in `solution.py` (no code change); the script was not re-executed since the output already existed and the script refuses to overwrite it.
- Anything in the task text or documentation that was unclear: nothing. The formula and the ordering hazard (task vectors against the unmodified base) were explicit.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: load the three checkpoints and save the merged one (`safe_open`, `save_file`).
  - `torch` 2.14.0: float32 arithmetic and bit-exact `torch.equal` comparisons.
  - Not used: `mergekit` 0.1.4 task arithmetic. Its merge does not perform the required precondition check (non-MLP tensors identical across all three checkpoints), it rebases the model through the HF loader, and it would have required extra care to guarantee that the 112 unchanged tensors are bit-exact and that exactly 48 tensors were touched. A ~120-line script gives full control over the checks the task requires and is simpler to verify.
- Approximate time spent, if you can tell: about 5 minutes.

## What the script enforces (all raise `SystemExit` with an `ERROR:` message)

1. Same tensor name set in base, ft1, ft2; 160 tensors; all 48 MLP names present; per-tensor shape and dtype equality across the three checkpoints; every non-MLP tensor bit-identical in all three (checked before any merge).
2. Merge `out = base + 0.4*(ft1-base) + 0.4*(ft2-base)` in float32; both task vectors are taken against the untouched `base` dict.
3. Exactly 48 tensors merged, exactly 160 tensors in the output dict.
4. Refuses to overwrite an existing output file. After saving, reloads the file from disk and re-verifies: 160 tensors, same key set, shapes and dtypes, the 112 unchanged tensors bit-equal to base, and the 48 merged tensors bit-equal to a recomputed merge.

Independent post-hoc check (separate one-off snippet, not part of the artifact): exactly 48 tensors differ from base in the output, all of them MLP tensors, all float32.
