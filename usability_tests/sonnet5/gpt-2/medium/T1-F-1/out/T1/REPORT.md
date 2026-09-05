# Participant self-report — T1 (condition F)

- **Final artifact path:** `out/T1/solution.py` (invoked via `out/T1/run.sh`),
  output at `out/T1/model.safetensors`.
- **Number of times you executed the script or plan:** 2 (first run failed a
  self-check I had written incorrectly; second run, after fixing the check
  itself, succeeded and was re-verified by deleting the output and re-running
  `run.sh` from scratch).
- **Which executions failed, and why (one line each):**
  1. `crash`-adjacent `failed_assertion`: my own "no stale old-block name"
     check flagged block 8 in the *output*, but that was the legitimately
     renumbered old block 11 (11 → 8), not a leftover of dropped block 8 —
     bug in the check, not in the rename logic; fixed by checking for stale
     `h.9.`/`h.10.`/`h.11.` prefixes specifically instead of reusing
     `DROP_BLOCKS`.
- **Pitfalls or surprises you hit (one line each):**
  - Easy to conflate "dropped block" with "block index that also appears as
    a *new* renumbered index" (8 is both) when writing a leftover-name check;
    the check must key off the original indices that were removed by
    renumbering (9, 10, 11), not the literal `DROP_BLOCKS` set.
  - Confirmed the Conv1D `[in, out]` weight layout doesn't matter here since
    values are copied verbatim, not transposed.
- **Anything in the task text or documentation that was unclear:** No.
- **Tools used (condition F):** Plain Python + `safetensors` 0.5.3 (load with
  `safe_open`, save with `safetensors.torch.save_file`) and `torch` 2.14.0
  as the tensor container. Chose this over `mergekit` layer-slicing or
  `torch-state-bridge` regex rewriting because the operation is a single
  regex-keyed rename over a flat state dict with no cross-model merge or
  arithmetic involved — a ~15-line loop is simpler and more auditable than
  authoring a YAML merge config or a rule-engine invocation for one
  renumbering pass, and it makes the required checks (block count, index
  contiguity, tensor count, non-block tensors unchanged) trivial to embed
  directly as asserts before the file is written.
- **Approximate time spent, if you can tell:** ~10 minutes including the
  fix-and-reverify cycle.
