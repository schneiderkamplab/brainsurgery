# T2 participant self-report (condition B: BrainSurgery plan)

- **Final artifact path:** `out/T2/plan.yaml` (output written to `out/T2/model.safetensors`)
- **Number of times you executed the script or plan:** 2
- **Which executions failed, and why (one line each):**
  - Execution 1: `no_match` — my "no scratch tensor survived" check used
    `count: { of: 'pruned\..*', is: 0 }`, but `count.of` resolves its reference
    first and raises `count.of matched zero tensors` instead of counting zero;
    replaced with `not: { exists: 'pruned\..*' }`. All other transforms and all
    required checks had already passed at that point; nothing was written.
- **Pitfalls or surprises you hit (one line each):**
  - `count` cannot express "exactly zero matches": a zero-match reference is an
    error in the resolver before the count is compared, so emptiness must be
    written as `not: { exists: ... }`.
  - `concat` requires each source reference to resolve to exactly one tensor, so
    the rebuild cannot be batched across layers with a pattern; the plan is 12
    explicit per-layer groups rather than one generic rule.
  - `concat` cannot write into a name that still exists, and `delete` cannot run
    before the sources are read, so each tensor needs the three-step
    concat-to-scratch / delete-original / move-back dance.
  - Conv1D `[in, out]` layout: the head axis is dim 1 (columns) for
    `c_attn.weight`/`c_attn.bias` but dim 0 (rows) for `c_proj.weight`; the three
    q/k/v blocks of head 5 are at columns 320..383, 1088..1151 and 1856..1919.
  - Name overreach was the obvious trap: `c_proj.weight` also exists under
    `h.<i>.mlp.`, so every delete target is a fully anchored, dot-escaped regex
    naming `attn` explicitly, and `attn.bias` (the `[1,1,1024,1024]` mask buffer)
    and `attn.c_proj.bias` are asserted to keep their original shapes.
- **Anything in the task text or documentation that was unclear:** No. The task
  gave the exact column/row ranges to keep, which removed the ambiguity about
  whether heads are interleaved or contiguous inside the fused projection. The
  README's assert section does not say that a zero-match reference is an error
  rather than a zero count, which is what cost me the first execution.
- **Tools used (condition F):** n/a (condition B).
- **Approximate time spent, if you can tell:** about 10 minutes.

## Notes on the checks

The plan asserts the four required conditions (`h.0.attn.c_attn.weight` is
`[768, 2112]`, `h.0.attn.c_attn.bias` is `[2112]`, `h.0.attn.c_proj.weight` is
`[704, 768]`, and `count: { of: '.*', is: 160 }`) before `output` is written, and
additionally asserts the same shapes for all 12 layers, that the untouched
`attn.c_proj.bias` and `attn.bias` keep their shapes, that all tensors are still
float32, and that no scratch `pruned.*` tensor survived.
