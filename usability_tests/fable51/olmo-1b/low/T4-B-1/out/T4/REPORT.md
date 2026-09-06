# T4 Participant self-report

- Final artifact path: `out/T4/plan.yaml` (plan), `out/T4/model.safetensors` (output, 114 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - `subtract` (non in-place) requires the destination to already exist, so task vectors were built as `copy` from the fine-tune into `base::tv<k>.<name>` followed by `subtract_` of the base, `scale_` by 0.4, then `add_` into the base tensor and `delete` of the temporaries.
  - All writes have to stay on one alias (`base::`) for the output alias to be inferred, hence the temporaries live under the `base` alias rather than on `ft1`/`ft2`.
  - Both task vectors are computed before either is added, so each is taken against the unmodified base.
- Anything in the task text or documentation that was unclear: nothing significant; the README example of `equal` with a negative lookahead mapped directly onto the shared-tensor check.
- Tools used (condition F): n/a (condition B)
- Approximate time spent, if you can tell: about 3 minutes
