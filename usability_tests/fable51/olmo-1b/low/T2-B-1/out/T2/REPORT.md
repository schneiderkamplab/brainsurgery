# T2 self-report (condition B, OLMo-1B-0724-hf)

- Final artifact path: `out/T2/plan.yaml` (output: `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 3
- Which executions failed, and why (one line each):
  - Execution 1: crash (YAML ParserError) — unquoted regex `[qkv]` inside a flow mapping was read as a YAML sequence.
  - Execution 2: crash (YAML ParserError) — my quoting sed missed the two preflight shape asserts on lines 10-11; same cause.
  - Execution 3: success; all asserts passed, 114 tensors written to a single file.
- Pitfalls or surprises you hit (one line each):
  - `concat` requires each source reference to resolve to exactly one tensor, so no regex-per-layer concat; I generated 64 explicit concat transforms (16 layers x q/k/v/o) into temporary `pruned.*` names, then regex-deleted the originals and regex-moved the pruned tensors back to the original names.
  - Regex references containing character classes must be quoted in YAML flow style.
  - Cross-alias `move`/`concat` was not needed; I used `pruned.` as a name prefix within the single `model` alias, which kept output-alias inference unambiguous.
- Anything in the task text or documentation that was unclear:
  - The docs do not say whether `concat` may write to the same tensor it reads from (in-place), so I used a temporary name to be safe.
  - Whether `output.path` ending in `.safetensors` guarantees a single unsharded file was inferred from "Output behavior" (directory-like path shards); it worked.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 5 minutes.
