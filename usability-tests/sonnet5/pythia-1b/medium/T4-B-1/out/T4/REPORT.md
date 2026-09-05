## Participant self-report

- Final artifact path: `out/T4/plan.yaml`
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  1. `output: { path: ..., alias: base }` — `PlanLoaderError: output received unknown keys: ['alias']`; the top-level `output` mapping only accepts `path`/`format`/`shard`, not `alias`.
- Pitfalls or surprises you hit (one line each):
  - With three inputs there is no default alias, so every tensor reference in the plan must be fully qualified with `alias::`.
  - `output` has no way to pick which alias to save; the engine infers the sole alias that data-transforms wrote into, so scratch work has to happen in a separate alias (created with `prefixes: { mode: add, alias: scratch }`) and that alias must be removed with `prefixes: { mode: remove, alias: scratch }` before `output`, otherwise output-model inference is ambiguous between `base` and `scratch`.
  - `add`/`subtract`/`scale` create/overwrite the whole destination tensor (not accumulate); only `add_`/`subtract_`/`scale_` are in-place accumulators. `subtract`'s destination must already exist, so I pre-created the delta tensor's slot with a `copy` before overwriting it with `subtract`.
  - `assert: { equal: { left: 'ft1::<exclude-mlp regex>(.+)', right: 'base::\1' } }` (as used in the doc pack's own `validation.yaml` example) turned out to be the right way to verify "same names, shapes, dtypes, values" across the whole checkpoint pattern-wise in one assertion, rather than listing 180 tensor names by hand.
- Anything in the task text or documentation that was unclear:
  - The README shows `output: { path, format, shard }` but doesn't state there's no `alias` key, nor how the output alias is chosen when multiple aliases are loaded; I had to read the `PlanLoaderError` and infer the output-model-selection rule (unique destination alias across data transforms) from behavior.
- Tools used (condition F): n/a (condition B)
- Approximate time spent, if you can tell: about 25 minutes, most of it reading `help.txt`/README before writing the plan; the two runs above happened in the same short interval.
