# T5 report

## Participant self-report

- Final artifact path: `out/T5/plan.yaml` (run via `brainsurgery out/T5/plan.yaml`), output written to `out/T5/`
- Number of times you executed the script or plan: 3
- Which executions failed, and why (one line each):
  1. `matmul`'s `from_b` field is a replacement template applied to the name matched by `from_a`, not an independent regex — escaping the literal dots in it (`\.`) produced literal backslashes in the resolved name, so the tensor lookup failed (`matmul source_b missing: lora::base_model\.model\.h\.0...`).
  2. `output.shard: 100MiB` was rejected — the parser only accepts a bare number plus `b|kb|mb|gb|tb` (checked in `docpack/help.txt`/README; unit suffix must be exactly `MB`, no `MiB`/`i` form).
  3. (none — third run passed all asserts and wrote the sharded output.)
- Pitfalls or surprises you hit (one line each):
  - `from_a`/`from_b`/`to` play different roles: `from_a` (and `from`, `left`, etc.) is matched as a full regex with capture groups, but paired-slot fields like `from_b`/`to` are template strings where `\1` is substituted in — they should use plain literal dots, not `\.`-escaped ones.
  - `MB` in `output.shard` is binary (1024²), not decimal, so `100MB` is exactly the 104,857,600-byte budget the task specifies — no need for `MiB`.
  - Batch regex+backreference matching works transparently across matmul/permute/scale/add_, so all 12 layers can be merged with one transform each instead of 12 separate blocks per step.
  - Since `output` model inference is based on which alias the write-destination transforms target, keeping every write (`matmul`/`permute`/`scale`/`add_`) pointed at the `base` alias was enough to keep the `lora` alias's adapter tensors out of the saved checkpoint; explicit deletion of the `merge.*` intermediates was still required so they didn't end up in the 160-tensor output.
- Anything in the task text or documentation that was unclear:
  - The doc pack doesn't spell out that `from_b`/`to` are replacement templates (vs. `from_a` being a real regex); this had to be discovered by trial and error.
  - Shard-size unit semantics (binary vs. decimal MB) aren't documented; discovered by trial and error and by the concrete required byte budget in TASK.md.
- Tools used (condition F): n/a (condition B)
- Approximate time spent, if you can tell: ~15 minutes of iteration (three plan executions plus verification scripting)
