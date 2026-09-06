# T1 self-report

- Final artifact path: `out/T1/solution.py` (output: `out/T1/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why: none.
- Pitfalls or surprises you hit:
  - Renumbering collision hazard: avoided by building a fresh output dict keyed
    by the new name and failing loudly if a name is ever written twice, instead
    of renaming in place in the source dict.
  - Input is sharded, so keys have to be gathered across both shard files via
    `model.safetensors.index.json`; the output is a single unsharded file.
  - `save_file` rejects non-contiguous/shared storage, so tensors are written
    with `.contiguous()`.
- Anything in the task text or documentation that was unclear: nothing; the
  old->new index mapping was given explicitly, so it was only a matter of
  implementing it without collisions. Minor: the task asks for exactly
  `out/T1/model.safetensors` while grading talks about the directory `out/T1`,
  so I wrote only the single file as specified.
- Tools used (condition F): `safetensors` 0.5.3 (load the sharded input,
  write the single-file output) and `torch` 2.14.0 (tensor objects only), from
  a plain Python script. I skipped mergekit and torch-state-bridge: mergekit's
  passthrough slicing expresses contiguous layer ranges and would need four
  slices plus a full HF export, and torch-state-bridge would still leave the
  I/O and every required check to me. The task is a keyed rename with three
  explicit assertions, so a ~90-line script with the checks inline was the
  shortest path that fails loudly.
- Approximate time spent: about 5 minutes.
