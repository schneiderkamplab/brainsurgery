# T4-B-1 participant self-report

- Final artifact path: `out/T4/plan.yaml` (output written to `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 3
- Which executions failed, and why (one line each):
  - #1: `PlanLoaderError: transform #8: unknown model alias: 'diff1'` — a
    `copy` into a brand-new alias (`diff1`) isn't allowed until the alias is
    declared first.
  - #2: `TransformError: cannot infer output model uniquely; expected exactly
    one destination model across all transforms` — with a top-level `output:`
    block, brainsurgery infers which alias to save from the transforms'
    destinations, and having `diff1`/`diff2`/`base` all as destinations made
    that ambiguous.
- Pitfalls or surprises you hit (one line each):
  - New aliases must be created explicitly with `prefixes: { mode: add, alias:
    ... }` before any transform can write into them; you can't introduce one
    implicitly via a `copy`'s `to`.
  - In `equal`/`copy`/`subtract_`/etc., when `left`/`from` is a regex with
    capture groups, `right`/`to` is a *substitution template* on that match
    (`\g<0>`, `\1`, ...), not an independent regex — writing a mirrored regex
    with its own `\.`/`\d` escapes on the destination side raises "invalid
    regex rewrite: bad escape".
  - The top-level `output:` block requires the destination alias to be
    inferable uniquely from the transforms; once more than one alias is
    written to, you must switch to an explicit `save: { alias: ..., path: ...
    }` transform instead.
- Anything in the task text or documentation that was unclear:
  - The README's `output:` section doesn't mention the "exactly one
    destination model" inference rule or that it can fail with multiple
    aliases in play; discovered this only by hitting the error and reading
    `save`'s help text, which does support an explicit `alias`.
- Tools used (condition F): name, version, and why: n/a (condition B)
- Approximate time spent, if you can tell: ~15 minutes of interactive
  exploration (help text + a scratch precondition test outside `out/`) plus
  3 plan executions.
