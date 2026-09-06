# T1 self-report

- Final artifact path: `out/T1/solution.py` (invoked via `out/T1/run.sh`), output at `out/T1/model.safetensors`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single run succeeded on the first try
- Pitfalls or surprises you hit (one line each):
  - None of consequence; the main hazard (renumbering collisions if blocks are shifted in the wrong order) is avoided structurally by building a brand-new dict keyed by new names from an old->new index map, rather than renaming keys in place.
  - Double-checked the fused `query_key_value` note (per-head interleaved q/k/v) but it doesn't matter for this task: whole tensors are copied verbatim per surviving block, no slicing inside `query_key_value`.
- Anything in the task text or documentation that was unclear: no, the old->new index mapping was given explicitly in the task, which removed any ambiguity about ordering.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3 (`safetensors.torch.load_file`/`save_file`) to read and write the checkpoint directly.
  - Plain Python (stdlib `re`) for pattern matching block indices and renumbering.
  - Chose a hand-written script over `mergekit` layer-slicing or `torch-state-bridge` regex rewriting because this task is a single flat state-dict rename/delete with an explicit index map: a ~15-line loop is easier to verify by inspection than translating the same map into a merge-tool YAML config or a regex-capture rewrite rule, and it lets the required checks (no blocks 12-15, exactly 12 blocks, exactly 184 tensors) live in the same script that produces the output, so it fails loudly and writes nothing on any violation.
- Approximate time spent, if you can tell: ~10 minutes including verification
