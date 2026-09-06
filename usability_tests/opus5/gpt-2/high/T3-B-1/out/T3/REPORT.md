# T3 self-report (condition B: BrainSurgery plan)

- Final artifact path: `out/T3/` — `model-0000{1..4}-of-00004.safetensors` plus
  `model.safetensors.index.json`; plan at `out/T3/plan.yaml`.

- Number of times you executed the script or plan: 2 executions of
  `out/T3/plan.yaml` (both succeeded; the second was an identical re-run only to
  read the transform log, which the first run's summary output had scrolled past).
  Plus 3 executions of throwaway plans that never touch `out/T3`: two
  `out/inspect.yaml` runs (a `dump` of the input tree, and a probe to confirm
  whether `assert: dtype` checks *every* regex match or just the first) and one
  `out/verify.yaml` run to re-check the finished output against the input. Both
  scratch plans were deleted afterwards.

- Which executions failed, and why (one line each):
  - `out/inspect.yaml` probe #2 failed *by design*: it cast one tensor to
    bfloat16 and then asserted `dtype: { of: '.*', is: float32 }`, confirming the
    assert iterates all matches rather than short-circuiting on the first.
  - No execution of `out/T3/plan.yaml` failed.

- Pitfalls or surprises you hit (one line each):
  - The buffer name `h.<i>.attn.bias` is a prefix-sibling of the parameters
    `h.<i>.attn.c_attn.bias` / `h.<i>.attn.c_proj.bias`; the delete pattern is
    only safe because tensor refs are *full-match* regexes, so
    `h\.\d+\.attn\.bias` cannot reach the projection biases. A substring match
    would have silently deleted 36 parameters.
  - "Exactly 48 tensors are bfloat16" is not directly expressible: `count` selects
    by name, not by dtype. I encoded it as a partition instead — the projection
    pattern matches exactly 48 and all of them are bfloat16, and the complementary
    negative-lookahead pattern `(?!...$).*` covers everything else and all of those
    are float32. Together those three asserts pin the bfloat16 count at 48.
  - The negative lookahead needs an explicit `$` inside it. Under `re.fullmatch`
    the outer pattern is anchored but the lookahead is not, so
    `(?!h\.\d+\....\.weight).*` would exclude nothing useful; `(?!...\.weight$).*`
    is what actually complements the cast set.
  - Shard units are binary, so the task's 64 MiB budget is written `shard: 64MB`
    (64 x 1024 x 1024 = 67,108,864), not `67108864` or `67MB`. The engine put
    `wte.weight` (154 MB) alone in shard 4 on its own, as the docs promise for a
    single tensor over budget.
  - No upcast step was needed: every input tensor is already float32, so
    "keep everything else float32" is satisfied by simply not touching it. I kept
    the float32 assert anyway so the plan fails loudly if that ever changes.

- Anything in the task text or documentation that was unclear:
  - The README documents shard units and the oversized-single-tensor rule clearly,
    but neither the README nor `help.txt` says whether `assert: dtype` with a
    multi-match pattern checks all matches; I had to probe for it. That property is
    what makes the "everything else is float32" check meaningful, so it is worth
    stating in the docs.
  - `assert: count` is name-based only. A dtype-filtered count (or a `dtype`
    variant that takes an expected number of matches) would let the four required
    checks be written literally instead of as a partition argument.

- Tools used (condition F): n/a — condition B, only `brainsurgery` and its YAML plans.

- Approximate time spent, if you can tell: ~15 minutes, most of it reading
  `docpack/README.md` and `help.txt` and pinning down the two regex questions
  (full-match semantics and multi-match `dtype`).
