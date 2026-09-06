# T1 report

- Final artifact path: `out/T1/solution.py` (invoked via `out/T1/run.sh`)
- Number of times you executed the script or plan: 2 (first attempt failed on
  an input-path bug in the script, second attempt succeeded)
- Which executions failed, and why (one line each):
  - Run 1: `FileNotFoundError` on `inputs/base/model.safetensors.index.json`
    — wrong number of `.parent` hops from `out/T1/solution.py` back to the
    sandbox root when computing `INPUT_DIR`.
- Pitfalls or surprises you hit (one line each):
  - None on the actual renumbering logic; the only snag was the relative
    path bug above, fixed by counting directory levels correctly
    (`out/T1` -> `out` -> sandbox root).
- Anything in the task text or documentation that was unclear:
  - None; the block list, removal set, and renumbering mapping were given
    explicitly in TASK.md, which made the mapping unambiguous.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3, used directly (`safe_open`/`save_file`) rather than
    mergekit or torch-state-bridge. The task is a single flat rename/filter
    over a small, fully-specified tensor set with an explicit index mapping;
    a ~110-line plain script over `safetensors` gave full control over the
    exact key regex, the collision check, and the required-check assertions
    (blocks 12-15 fully removed, exactly 12 surviving blocks, exactly 86
    tensors, non-block tensors byte-identical) without needing to learn a
    merge-config DSL for a one-off bulk rename.
- Approximate time spent, if you can tell: a few minutes (one script write,
  one path-bug fix, one clean run).
