## Participant self-report

- Final artifact path: `out/T4/solution.py` (invoked via `out/T4/run.sh`)
- Number of times you executed the script or plan: 2 (first run succeeded
  functionally but wrote the output to the wrong path (`out/model.safetensors`
  instead of `out/T4/model.safetensors`) due to an off-by-one in a `Path`
  computation; fixed and re-ran, second run wrote to the correct path).
- Which executions failed, and why (one line each):
  1. Not a crash/assertion failure — ran and produced correct tensor content,
     but at the wrong output location (`out/model.safetensors`), which would
     have failed grading since `grade.py` expects `out/T4/model.safetensors`.
- Pitfalls or surprises you hit (one line each):
  - Easy to get `Path.parent` chains wrong when the script lives inside the
    output directory it needs to write to (`out/T4/solution.py` writing to
    `out/T4/model.safetensors`); fixed by pointing `OUT_DIR` at the script's
    own directory instead of a `.parent` of it.
- Anything in the task text or documentation that was unclear: none — the
  spec for the merge formula, the precondition check, and the required
  checks were all unambiguous.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3 — load/save the three input checkpoints and the output
    checkpoint directly by tensor name.
  - `torch` 2.14.0 — tensor arithmetic in float32, `torch.equal` for the
    bit-exact precondition check, dtype casting back to the original storage
    dtype after merging.
  - Chose a plain script over `mergekit`'s task-arithmetic merge method
    because this task's hard requirement is the *precondition check*
    (identical non-MLP tensors, bit-exact, across all three checkpoints,
    aborting loudly if violated) plus a hard count check on the merged/total
    tensor sets. `mergekit` computes task-arithmetic merges but doesn't
    expose that specific three-way identity precondition as a single
    built-in gate, so it would still need this same custom check bolted on
    beforehand; doing the load/verify/merge/save all in one ~110-line script
    against `base`/`ft1`/`ft2` state dicts is more direct and easier to audit
    end-to-end than wiring a mergekit YAML plus a separate verification pass.
- Approximate time spent, if you can tell: ~10 minutes (script authored once,
  one path bug found and fixed, then verified merge math and unchanged
  tensors by hand-checking one MLP tensor and one non-MLP tensor against
  independently recomputed expected values).
