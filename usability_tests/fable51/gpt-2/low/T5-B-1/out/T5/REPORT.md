# Participant self-report: T5 (GPT-2 124M), condition B

- Final artifact path: `out/T5/plan.yaml` (output checkpoint in `out/T5/`, 5 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 4
- Which executions failed, and why (one line each):
  - 1: `no_match` — `matmul` `from_b` is a capture rewrite, not a regex; escaped dots (`\.`) were taken literally, so `lora::base_model\.model\.h\.0\....` was not found.
  - 2: `failed_assertion` — my extra (non-required) guard `writes: {ge: 1, lt: 2}` on merged weights fired: `add_` records writes=2.
  - 3: `failed_assertion` — replacement guard `writes < 1` on untouched tensors fired: every loaded tensor already has writes=1, so the counter is unusable for "unchanged" checks. Guard removed.
  - 4: success.
- Pitfalls or surprises you hit (one line each):
  - In ternary transforms, `from_b`/`to` are rewrite templates (plain dots), only `from_a` is a regex.
  - `writes` access counts include the initial load, and `add_` counts as 2 writes; not documented in the doc pack (no `help: {assert: writes}` entry in help.txt).
  - Intermediates had to live on the `base` alias so that output-alias inference stays unambiguous; deleted them before saving.
  - Transpose is done with `permute` `order: [1, 0]` (no dedicated transpose transform).
- Anything in the task text or documentation that was unclear:
  - help.txt lacks entries for `assert.reads`/`assert.writes` keys and semantics.
  - Whether `from_b` in ternary transforms is regex or rewrite is only implied by the interfaces reference mapping note.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~5 minutes

Verification (outside the plan, read-only): 160 tensors, no `lora_` names, all shards ≤100 MiB of tensor data except `wte.weight` alone in shard 5, merged weights match `W + 2*(B@A).T` with relative error 0.0, all other tensors bit-identical to the base.
