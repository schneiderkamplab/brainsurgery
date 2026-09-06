## Participant self-report

- Final artifact path: `out/T1/plan.yaml` (output written to `out/T1/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single run passed all asserts and produced 121 tensors.
- Pitfalls or surprises you hit (one line each):
  - Renumbering-order hazard: moves are listed low-index-first (3→2, 4→3, 6→4, 7→5, 9→6, 10→7, 11→8) so each destination slot is always empty (already deleted or already vacated by an earlier move) before it is written, since `move` refuses to overwrite an existing destination.
  - `attn.bias` (the causal-mask buffer) is a plain per-block tensor like the rest, so the same `h.<i>.` regex prefix on delete/move covers it without special-casing.
- Anything in the task text or documentation that was unclear: none; the README's note that `move`/`copy` destination rewrites support regex backreferences (`\1`) exactly like `assert.equal`'s `right` made the whole-block renumbering a single `move` per block.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: a few minutes, mostly reading `docpack/README.md` and `docpack/help.txt` for `move`/`delete`/`assert` semantics before writing the plan.
