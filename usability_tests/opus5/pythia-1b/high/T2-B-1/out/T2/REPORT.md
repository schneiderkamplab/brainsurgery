# T2 self-report (condition B: BrainSurgery plan)

- **Final artifact path:** `out/T2/plan.yaml` (output written to `out/T2/model.safetensors`)
- **Number of times you executed the script or plan:** 1 (succeeded on the first execution)
- **Which executions failed, and why (one line each):** none.
- **Pitfalls or surprises you hit (one line each):**
  - `concat` requires each `from` reference to resolve to exactly one tensor and a single
    `to`, so it cannot be pattern-expanded over layers; the plan needs 48 explicit `concat`
    blocks even though every other step is one pattern-based transform for all 16 layers.
  - Because `concat` creates a new destination, the full-width originals have to be
    `delete`d before the pruned tensors can be written back under the same names; the
    surviving slices therefore have to be parked in temporary tensors first.
  - Temporary names were kept dot-free (`tmp_qkvw_lo_3`) on purpose: references are
    full-match regexes, so a dot in a scratch name is a wildcard and invites overmatching
    between e.g. layer 1 and layer 11.
  - Every dot in a real tensor name has to be escaped in `from`/`target`/`of` patterns,
    while the `to` side is a replacement template where dots are literal — the two sides of
    the same transform look inconsistent and are easy to get wrong.
  - The GPT-NeoX interleaved fused QKV layout means head 5 is one contiguous 768-row block
    (3840..4607), so the fused weight and bias need only a single cut; had it been
    `[q | k | v]` segments it would have required three cuts per tensor.
  - `attention.dense.weight` is pruned on its *input* axis (columns, `[:, :1280]` +
    `[:, 1536:]`, `dim: 1`), the opposite axis from the QKV projection — the one place
    where a wrong axis still produces a plausible-looking tensor.
  - The output has no `__metadata__` entry, whereas the input checkpoint carries one; the
    244 tensor keys, shapes, dtypes and values are unaffected and the plan format offers no
    control over it.
- **Anything in the task text or documentation that was unclear:**
  - The task text was precise (explicit row/column ranges, explicit layout warning); nothing
    was ambiguous.
  - The README lists `concat` without noting that it is the only reassembly primitive and
    that it is not pattern-expandable, which is the single fact that shapes the whole plan.
  - The `concat` and `split` help examples are rendered with an empty `from:`/`to:` list
    (`concat: { from: , to: a::xy, dim: 0 }`), so the intended list syntax has to be inferred
    from the second example.
  - The README's documentation links point at an absolute path on someone else's machine.
- **Tools used (condition F):** n/a (condition B).
- **Approximate time spent, if you can tell:** ~15 minutes, of which ~14 s was the plan run.

## Verification performed

Beyond the required `assert` transforms (which also cover all 16 layers, dtypes and the
untouched tensors), the output header and payload were checked directly against the input
without any further plan execution:

- 244 tensor keys, identical key set to the input;
- `layers.{0,15}.attention.query_key_value.weight` `[5376, 2048]`, `.bias` `[5376]`,
  `attention.dense.weight` `[2048, 1792]`, all `F16`;
- byte-exact comparison of the kept QKV blocks (rows `0..3839` and `4608..6143`) for layer 3;
- byte-exact comparison of the kept `dense.weight` column blocks (`0..1279`, `1536..2047`)
  on rows 0, 1, 1023 and 2047 of layer 7;
- byte-exact equality of untouched tensors: `embed_in`, `embed_out`, `attention.dense.bias`,
  `mlp.dense_4h_to_h.weight`, the `attention.bias` U8 buffer and `rotary_emb.inv_freq`.
