# Participant self-report

- Final artifact path: `out/T4/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single run passed all asserts and wrote the output.
- Pitfalls or surprises you hit (one line each):
  - `add`/`subtract`/`scale`/`assign` require destinations to already exist, so
    the task vectors had to be materialized first with `copy` (which requires
    the destination to *not* exist) before `subtract` could fill them in.
  - The output alias is inferred from which alias every write targets, so the
    scratch task-vector tensors had to live under the `base` alias (as
    `base::tv1__...` / `base::tv2__...`) rather than a separate alias, or the
    run would have failed with "cannot infer output model uniquely".
  - `assert: equal` needs a capture group to use `\1`; the negative-lookahead
    pattern for "every non-MLP tensor" (`(?!h\.\d+\.mlp\.).+`) has no capturing
    group, so the right side must use `\g<0>` (whole match) instead, as shown
    in the README's own example for excluding `h.<i>.mlp.*`.
- Anything in the task text or documentation that was unclear: no; the README's
  worked example for `equal` with a negative-lookahead pattern was exactly what
  was needed for the step-1 precondition check.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: one exploration pass through
  `docpack/help.txt` for `assert`/`add`/`subtract`/`scale`/`copy`, then one
  plan draft that ran clean on the first execution.
