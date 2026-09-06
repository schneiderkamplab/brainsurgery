# Participant self-report

- Final artifact path: `out/T5/plan.yaml`
- Number of times you executed the script or plan: 3
- Which executions failed, and why (one line each):
  - Attempt 1: plan failed to compile — used `${i}`/`${mod}` interpolation in
    structured-expression `to:` fields, which OmegaConf's YAML loader tried to
    resolve itself before brainsurgery ever saw it (`InterpolationKeyError`).
  - Attempt 2: escaped the `${...}` as `\${...}` to dodge OmegaConf, but then
    hit "matmul source_a matched zero tensors" — structured list references
    (`["a","$i","b"]`) always resolve against the transform's single default
    model alias, so mixing `lora::[...]` and `model::[...]` in the same
    from_a/from_b/to triple isn't possible; the alias tag on a structured list
    is silently ignored.
- Pitfalls or surprises you hit (one line each):
  - Structured-expression syntax (`$i`, `${i}`) cannot carry an explicit
    `alias::` prefix per-reference — it only works when every ref in the
    transform shares one implicit default alias, which fails as soon as you
    need to read from one alias (`lora`) and write to another (`model`) in
    the same `matmul`/`add_`/`scale` call.
  - Switched to plain regex references with capture groups (`\1`, `\2`) and
    explicit `alias::` prefixes on every side (from_a/from_b/to), which
    supports cross-alias rewriting fine and let one `matmul`, one `scale`,
    and one `add_` transform cover all 32 (layer, module) pairs at once.
  - `matmul`/`scale` write new tensors, so I routed B@A into scratch tensors
    (`tmp.<i>.<mod>.weight`, then `tmp2.<i>.<mod>.weight`) and merged with an
    in-place `add_` into the existing base weight, then `delete`d both
    scratch families before asserting no `lora_` names remain and saving.
- Anything in the task text or documentation that was unclear:
  - The doc pack's structured-expression example (`README.md`) shows `$i`/
    `${i}` usage but never shows it combined with an explicit `alias::`
    prefix or with more than one input alias, which is what actually broke;
    a note that structured refs use one implicit default model per transform
    would have saved an attempt.
- Tools used (condition F): name, version, and why: n/a (condition B)
- Approximate time spent, if you can tell: ~20 minutes
