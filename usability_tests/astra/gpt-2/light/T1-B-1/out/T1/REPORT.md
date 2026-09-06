# Participant self-report

- Final artifact path: `out/T1/model.safetensors`; plan: `out/T1/plan.yaml`.
- Number of times you executed the script or plan: 2.
- Which executions failed, and why (one line each): Execution 1 failed with `TransformError: count.of matched zero tensors` on the absence check; no checkpoint was written. Execution 2 succeeded with all assertions passing.
- Pitfalls or surprises you hit (one line each): `count` with `is: 0` rejects an empty match; changed the check to `not: exists`. Ascending source-index moves avoid destination collisions after deletion.
- Anything in the task text or documentation that was unclear: The count documentation says exact match count but does not explain that zero matches raise an error even for an expected count of zero.
- Tools used (condition F): Not applicable (condition B); BrainSurgery CLI and shell file operations only, no Python.
- Approximate time spent, if you can tell: About 3 minutes.
