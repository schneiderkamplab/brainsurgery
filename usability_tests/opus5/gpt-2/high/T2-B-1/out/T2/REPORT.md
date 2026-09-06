# T2 self-report (condition B: BrainSurgery plan)

- **Final artifact path:** `out/T2/plan.yaml` (output written to `out/T2/model.safetensors`)

- **Number of times you executed the script or plan:** 1

  `out/T2/plan.yaml` was executed once and passed on the first run. Before
  writing it I ran three throwaway probe plans under `tmp/` (`inspect.yaml`,
  `probe.yaml`, `probe2.yaml`, `probe3.yaml`) to check tensor names and to pin
  down the semantics of `concat`/`delete`/`move` and of the assert operators,
  and two verification plans (`tmp/verify.yaml`, `tmp/verify2.yaml`) after the
  run. Those are not the task plan and produce no output checkpoint.

- **Which executions failed, and why (one line each):**
  - None for `out/T2/plan.yaml`.
  - (Probe, not the plan) `tmp/probe3.yaml`, `failed_assertion`/`no_match`: `assert: count: {of: 'model::pruned_.*', is: 0}` raised `count.of matched zero tensors` instead of succeeding, so the "no scratch tensors left over" check was rewritten as `assert: {not: {exists: ...}}`.

- **Pitfalls or surprises you hit (one line each):**
  - `count` with `is: 0` cannot express "nothing matches": zero matches is an error before the count is compared, `not: exists` is the way to say it.
  - Tensor references are full-match regexes, so unescaped `.` in `h.0.attn.c_attn.weight` is a wildcard; harmless for these names but I escaped every dot in source and `target` refs anyway, and left them unescaped only in `move`/`concat` destinations, which are literal names.
  - `concat` accepts sliced source references directly (`'model::name::[:, :320]'`), so no intermediate `split` and no cleanup of nine scratch pieces per tensor was needed: six slices straight into one new tensor.
  - `concat`/`copy` destinations must not exist, so a tensor cannot be rebuilt onto its own name in one step; the sequence has to be concat to a scratch name, `delete` the original, `move` the scratch back.
  - The Conv1D `[in, out]` layout is the whole trick: heads are *columns* of `c_attn.weight` (dim 1, three separate q/k/v blocks at 320:384, 1088:1152, 1856:1920) but *rows* of `c_proj.weight` (dim 0, one block at 320:384). Slicing `c_proj.weight` on dim 1 would have produced a plausible-looking `[768, 704]` and garbage attention.
  - `attn.bias` is the `[1, 1, 1024, 1024]` causal mask buffer, not a projection bias; a pattern like `h\.\d+\.attn\..*bias` would have swept it up. I never used such a pattern, and asserted its shape is untouched.
  - A multi-match `assert: shape` does check every match (verified deliberately with a wrong expected shape in `tmp/probe2.yaml`), so the per-layer checks over `h\.\d+\.` are real checks and not just a check of the first match.

- **Anything in the task text or documentation that was unclear:**
  - The task text was unambiguous; it gives the kept column ranges explicitly, so the only real work was mapping them onto the tool's slice syntax and the right `dim`.
  - The README lists reference forms `alias::expr`, `expr` and `alias::expr::[slice]` but not `expr::[slice]`, so it is not stated whether the slice form works without an explicit alias. I sidestepped it by always writing `model::`.
  - Nothing in the docs says what `count` does when a pattern matches nothing; the error above was the only way to find out.

- **Tools used (condition F):** n/a (condition B).

- **Approximate time spent, if you can tell:** about 10 minutes, most of it reading `help.txt` and probing assert semantics rather than writing the plan.

## What the plan does

For each layer `i` in 0..11, three tensors are rebuilt from the slices that survive:

| tensor | dim | kept slices | result |
|---|---|---|---|
| `h.i.attn.c_attn.weight` | 1 | `[:, :320]`, `[:, 384:768]`, `[:, 768:1088]`, `[:, 1152:1536]`, `[:, 1536:1856]`, `[:, 1920:2304]` | `[768, 2112]` |
| `h.i.attn.c_attn.bias` | 0 | `[:320]`, `[384:768]`, `[768:1088]`, `[1152:1536]`, `[1536:1856]`, `[1920:2304]` | `[2112]` |
| `h.i.attn.c_proj.weight` | 0 | `[:320, :]`, `[384:768, :]` | `[704, 768]` |

each via `concat` into a scratch name, `delete` of the original, `move` of the
scratch back onto the original name, so names and tensor count are preserved.

## Checks in the plan (all before the write)

Required: shape of `h.0.attn.c_attn.weight` is `[768, 2112]`; shape of
`h.0.attn.c_attn.bias` is `[2112]`; shape of `h.0.attn.c_proj.weight` is
`[704, 768]`; `count` of `model::.*` is 160.

Added: the same three shape checks over `h\.\d+\.` (all 12 layers); the
untouched `attn.c_proj.bias` `[768]` and `attn.bias` `[1, 1, 1024, 1024]` still
have their original shapes; exactly 12 matches for each rebuilt tensor kind; no
`pruned_*` scratch tensor survives; every tensor is still `float32`.

## Post-run verification (outside the plan)

- `tmp/verify.yaml`: `diff` of the input against the output reports nothing
  missing on either side and exactly 36 differing tensors, i.e. only the three
  head-bearing tensors per layer; an `assert: equal` over every *other* tensor
  name confirms they are bit-identical to the input.
- `tmp/verify2.yaml`: 23 slice-level `assert: equal` checks confirm each kept
  block of the output lines up bit-exactly with the block it came from in the
  input (`c_attn.weight` all six blocks for layers 0 and 11, `c_attn.bias` all
  six blocks for layer 7, `c_proj.weight` both blocks for layers 0 and 11),
  plus a negative control that the columns at the old seam `320:384` no longer
  match the input, i.e. head 5 really is gone rather than the tensor merely
  being truncated at the end.
