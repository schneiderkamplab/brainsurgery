# T1 self-report

- Final artifact path: `out/T1/model.safetensors` (plan: `out/T1/plan.yaml`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: `crash` — `assert: { count: { of: 'gpt_neox\.layers\.1[2-5]\..*', is: 0 } }` cannot express "zero matches": `count.of` resolves its reference first and raises `TransformError: count.of matched zero tensors` before comparing to `is: 0`. Replaced with `assert: { not: { exists: ... } }`.
- Pitfalls or surprises you hit (one line each):
  - `count` is unusable for asserting absence; `not: exists:` is the right operator, and its payload is a bare tensor-ref, not a `{ of: ... }` mapping.
  - Renumbering collision hazard: the `move`s must run in ascending old-index order (3->2, 4->3, ... 15->11) so each destination is already free; `move` rejects existing destinations, so a wrong order would have failed loudly rather than silently overwriting.
  - Dots in tensor names must be escaped in the regex (`gpt_neox\.layers\.2\..*`), otherwise `.` would also match e.g. a hypothetical `layers-2`.
  - Deleting/renaming whole blocks with a trailing `.*` after the escaped index dot cleanly covers all 15 tensors per block including the `attention.bias` / `masked_bias` / `rotary_emb.inv_freq` buffers.
- Anything in the task text or documentation that was unclear:
  - The docs list `count` with `is` but do not warn that a zero expectation is unreachable; the "matched zero tensors" resolver error applies to assert references too.
  - The README does not state the payload shape of each assert operator; `help.txt` ("Payload: tensor-ref") was needed for `exists`.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~10 minutes

## Plan summary

`delete` blocks 2, 6, 10, 14 by pattern; `move` the 10 shifted blocks in ascending
order with a capture-group rewrite of the trailing name; then assert no
`layers.1[2-5]` tensor remains, exactly 12 `query_key_value.weight` tensors,
180 block tensors, and 184 tensors total.
