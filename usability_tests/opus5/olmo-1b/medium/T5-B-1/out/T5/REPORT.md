# T5 self-report (Condition B: BrainSurgery plan)

- Final artifact path: `out/T5/` (10 shards + `model.safetensors.index.json`);
  plan at `out/T5/plan.yaml`.
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none — the single run
  passed all asserts and wrote the checkpoint.
- Pitfalls or surprises you hit (one line each):
  - Output alias inference: every write (`matmul` destination, `scale_`,
    `add_`, `delete`) had to be on the `base` alias, otherwise the run would
    fail with "cannot infer output model uniquely".
  - The temporary delta tensors could not be named with a `lora_` substring,
    since the required check forbids `lora_` anywhere in the output; I used
    the prefix `tmp_delta.` and deleted it before writing.
  - Regex references are full-match on the source side but plain rewrite
    strings on the `from_b`/`to` side, so escaped dots belong only in the
    matching pattern.
  - `shard: 512MB` is binary (512 * 1024 * 1024), which is exactly the
    536,870,912-byte budget the task asks for; no oversized shard resulted,
    and the two 412 MB tensors were placed alone.
- Anything in the task text or documentation that was unclear:
  - The docs state that `from_b`/`to` in ternary transforms use the same
    capture-rewrite model as `to` in `copy`, but the `add` help example
    (`from_a: '.*.weight', from_b: '.*.delta'`) reads like independent
    pattern matching; a capture-based example would be clearer.
  - Nothing in the task text was unclear; the `fan_in_fan_out` note removed
    the usual ambiguity about transposing `B @ A`.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~10 minutes, most of it reading
  `docpack/README.md` and `help.txt`.

## Verification performed

Independent check (outside the plan) against the inputs:
114 tensors, key set identical to the base, all float32 with unchanged
shapes; the 32 merged `q_proj`/`v_proj` weights match
`W + 2 * (B @ A)` with relative Frobenius error 0.0, and the other 82
tensors are bit-exact copies of the base. All shards hold at most
536,870,912 bytes of tensor data.
