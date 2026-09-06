## Participant self-report

- Final artifact path: `out/T4/plan.yaml` (output written to `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - `add`/`subtract`/`multiply`/`assign` all require a pre-existing destination
    tensor, so the task vectors had to be staged as scratch copies (via `copy`)
    before `subtract_`/`scale_`/`add_` could accumulate into them.
  - All scratch tensors had to be created under the `base` alias (with a
    `__b_`/`__d1_`/`__d2_` name prefix) rather than a separate alias, because
    the output alias is inferred as the single alias every transform writes
    to; using a second alias for scratch space would have made the alias
    inference ambiguous.
  - Confirmed empirically (with a small synthetic checkpoint) that regex
    capture-group correspondence in `subtract_`/`scale_`/`add_`/`assign` is
    matched by the captured name, not by insertion order, so per-tensor
    ordering across the 48 MLP tensors was not a hazard.
- Anything in the task text or documentation that was unclear: none; the
  README's `equal`-with-negative-lookahead example (`(?!h\.\d+\.mlp\.).+`) was
  exactly the pattern needed for the "everything outside the MLP tensors"
  check.
- Tools used (condition F): n/a (condition B)
- Approximate time spent, if you can tell: ~20 minutes, most of it verifying
  the exact multi-tensor matching semantics of `subtract_`/`add_`/`scale_`/
  `assign` against a synthetic checkpoint before touching the real inputs.
