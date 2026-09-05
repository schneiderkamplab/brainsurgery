# Participant self-report — T2 (condition B)

- Final artifact path: `out/T2/plan.yaml`
- Number of times you executed the script or plan: 2 (first run succeeded; a second run was
  a verification re-run of the identical plan, both passed)
- Which executions failed, and why (one line each): none failed.
- Pitfalls or surprises you hit (one line each):
  - `concat`'s `from` list requires each reference to resolve to exactly one tensor — regex
    broadcasting (as used by `copy`/`move`/`delete` with a capture group) is not supported for
    `concat`, so the query_key_value/dense slicing had to be spelled out once per layer (16
    repetitions of the same 3-transform block) instead of one broadcast rule.
  - `move` requires the destination not to already exist and does not support slicing, so I
    built the pruned tensors under a temporary `__pruned` suffix name, deleted the originals by
    regex, then `move`d the temporary names back onto the original tensor names using a
    regex-with-capture (which `move` does support).
  - Row/column math needed care: head 5 occupies query_key_value rows `3840..4607` (768-wide
    interleaved q/k/v block) and dense.weight columns `1280..1535` (256-wide); both were
    excluded while keeping everything before and after in original order.
- Anything in the task text or documentation that was unclear: none; the row/column boundaries
  in TASK.md were precise enough to compute directly.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: ~10 minutes, most of it reading `help.txt` for
  `concat`/`move` semantics and confirming the pruning arithmetic with a small Python check
  against the raw safetensors file before finalizing the plan.
