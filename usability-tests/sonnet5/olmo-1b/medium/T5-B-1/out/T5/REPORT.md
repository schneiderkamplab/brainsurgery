# Participant self-report — T5 (Condition B)

- Final artifact path: `out/T5/plan.yaml`
- Number of times you executed the script or plan: 1 (of `out/T5/plan.yaml` itself). Before writing
  the final plan I ran two throwaway probe plans under a scratch `tmp/` directory (deleted afterward)
  to confirm two undocumented mechanics: (1) the exact escaping convention for regex vs. replacement
  strings in batch transforms, and (2) how `shard:` size suffixes are parsed. Those probe runs are not
  counted as attempts on the real task plan.
- Which executions failed, and why: none for `out/T5/plan.yaml` — it succeeded on the first run.
  One of the earlier scratch probes failed once with `matmul source_b missing: ...lora_A...` because I
  had escaped the dots (`\.`) in a replacement-side reference (`from_b`); replacement strings use plain
  literal dots while match-side references use regex-escaped dots.
- Pitfalls or surprises you hit:
  - `matmul`/`add_`/`scale` all support batch regex mapping via capture groups (like the `copy`/`assign`
    examples in the doc pack), but the source side (`from_a`) needs `\.` for a literal dot while the
    generated/replacement side (`from_b`, `to`) needs a plain `.` — mixing the two conventions up causes
    a "missing" error that looks like a naming problem rather than an escaping problem.
  - `matmul` requires a destination that does not already exist, so the low-rank product has to land in
    a brand-new intermediate tensor name (`...lora_delta`) before it can be scaled and added into the
    existing base weight with `add_`; there's no single fused "scaled matmul-accumulate" transform.
  - An `assert`-only transform doesn't count as writing an output alias, so a plan with only asserts
    before `output` fails with "cannot infer output model uniquely" — the plan needs at least one
    transform that actually touches the destination alias.
  - The `shard:` size suffix (`MB`) is parsed as binary mebibytes (2^20), not decimal megabytes; I
    confirmed this empirically with a probe run before trusting `shard: 512MB` to produce shards whose
    tensor payload is bounded by exactly 536,870,912 bytes.
- Anything in the task text or documentation that was unclear: the doc pack doesn't state whether
  `output` picks a specific alias by name or "the first input alias" when a plan has multiple aliases
  loaded (base + lora here); I inferred it from the worked MoE example, where the plan comments describe
  the first-listed alias as "the output anchor", and confirmed by running a plan with two aliases and
  checking that only the first (`model`) alias's tensors, not the second (`lora`), appear in the output.
- Tools used (condition F): n/a, this is condition B.
- Approximate time spent: one exploration pass through the doc pack plus two short probe runs, then one
  successful run of the final plan — well under the medium-effort budget.
