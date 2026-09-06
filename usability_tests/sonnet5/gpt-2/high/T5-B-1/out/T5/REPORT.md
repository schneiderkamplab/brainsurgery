## Participant self-report

- Final artifact path: `out/T5/plan.yaml`, executed against `out/T5/` (shards +
  `model.safetensors.index.json`).
- Number of times you executed the script or plan: 1.
- Which executions failed, and why (one line each): none; the single run
  passed all asserts and wrote the output.
- Pitfalls or surprises you hit (one line each):
  - Ternary math transforms (`add`/`multiply`/`subtract`/`matmul`) treat
    `from_a` as the regex to match/capture and rewrite `from_b`/`to` from
    those captures (same model as `from`/`to` on binary transforms) — this
    isn't spelled out with a worked capture example in the docs, so I
    verified it on a throwaway two-layer scratch checkpoint before trusting
    it on the real inputs.
  - `matmul`'s `to` must not already exist and can't be sliced, so the
    `B @ A` product and its transpose each need their own new intermediate
    tensor name before the final `add_`; all three intermediates
    (`lora_ba`, `lora_ba_t`, `lora_delta`) have to be deleted explicitly so
    they don't leak into the sharded output.
  - Conv1D `[in, out]` vs. Linear `[out, in]`: `fan_in_fan_out=true` means
    `(B @ A)` (shape `[out, in]`) must be transposed with `permute` before
    adding into `h.<i>.attn.c_attn.weight` (shape `[in, out]`).
  - Output-alias inference is by "who gets written": as long as every
    write (`matmul`/`permute`/`scale`/`add_`/`delete` destinations) stays on
    the `base` alias and nothing is ever written to the `lora` alias, the
    output resolves to `base` unambiguously even with two inputs loaded.
- Anything in the task text or documentation that was unclear: the README
  and `help.txt` state that ternary transforms "support the same
  capture-based rewrite model" as binary transforms but don't show a worked
  regex-capture example for `from_a`/`from_b`/`to` together, unlike binary
  `copy`/`equal` which do; I confirmed the direction (matched-then-rewritten)
  empirically rather than from documentation text alone.
- Tools used (condition F): n/a.
- Approximate time spent, if you can tell: one plan draft, one scratch
  verification, one execution; no retries needed.
