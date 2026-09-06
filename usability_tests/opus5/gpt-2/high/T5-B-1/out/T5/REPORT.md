# T5 — LoRA adapter merge with sharded export (GPT-2 124M), condition B

## Participant self-report

- **Final artifact path:** `out/T5/plan.yaml` (output checkpoint in `out/T5/`:
  `model-0000{1..5}-of-00005.safetensors` + `model.safetensors.index.json`)

- **Number of times you executed the script or plan:** 1
  (plus one execution of a separate verification-only plan, `out/verify.yaml`,
  which does not write anything and is not part of the solution)

- **Which executions failed, and why (one line each):** none — the single
  execution of `out/T5/plan.yaml` succeeded.

- **Pitfalls or surprises you hit (one line each):**
  - Conv1D `[in, out]` vs Linear `[out, in]`: `fan_in_fan_out = true` means the
    product `B @ A` is `[2304, 768]` and must be transposed to `[768, 2304]`
    before `add_`; I did this with `permute: { order: [1, 0] }`, since no
    transpose transform exists and matmul refs cannot be transposed via slicing.
  - The shapes pin down almost everything: `A [16,768]`, `B [2304,16]` only
    compose as `B @ A`, and `add_` would fail loudly on a wrong transpose, so the
    single genuinely free parameter is the scale `alpha / r = 32 / 16 = 2`.
  - Intermediates must live on the alias that gets written (`base::`), otherwise
    output-alias inference sees writes on two aliases; they are `delete`d before
    the output is saved so nothing extra reaches the checkpoint.
  - I first named the intermediates `lora_merge.<i>.delta`, which would have made
    the "no `lora_` in the output" assert self-referential; renamed to `merged.*`.
  - `100MB` in `output.shard` is binary (104,857,600 bytes) and counts tensor data
    only — the resulting shards hold 101,840,896 / 104,718,336 / 100,021,248 /
    87,120,896 bytes, and `wte.weight` (154,389,504 bytes) lands alone in shard 5.
  - Regex-mode capture rewriting works across all three refs of `matmul`
    (`from_a` drives, `from_b`/`to` are rewrites), which is what makes the whole
    12-layer merge four transforms instead of 48.

- **Anything in the task text or documentation that was unclear:**
  - The README documents capture-based rewriting explicitly only for `copy`/`move`
    and `assert.equal`; that `permute`'s `to` and `matmul`'s `from_b` behave the
    same way is stated only in passing in `interfaces-reference.md` §9
    ("including ..."), so I could not tell from the docs alone whether the
    pattern form would work before running it.
  - `TASK.md` says `target_modules = ["attn.c_attn"]` while
    `adapter_config.json` actually says `["c_attn"]`; the adapter tensor names
    resolve the ambiguity, but the two texts disagree.
  - The docs do not say whether tensors created and then deleted mid-plan can
    perturb the state-dict order used for shard packing (they do not, but that
    had to be confirmed from the resulting index).

- **Tools used (condition F):** n/a

- **Approximate time spent, if you can tell:** ~6 minutes, most of it reading
  `help.txt` and the example plan; the plan ran in ~10 s.

## What the plan does

1. Preflight asserts: base has 160 tensors; 12 `lora_A` and 12 `lora_B` tensors
   with shapes `[16, 768]` / `[2304, 16]`; base `c_attn` weights are
   `[768, 2304]` float32.
2. `matmul` `B @ A` per layer into `base::merged.<i>.delta` `[2304, 768]`.
3. `permute` with `order: [1, 0]` to `base::merged.<i>.deltaT` `[768, 2304]`,
   drop the untransposed intermediate, `scale_` by `2.0` (= `lora_alpha / r`).
4. `add_` the 12 deltas into `h.<i>.attn.c_attn.weight`, then `delete` them.
5. Required checks, all as `assert` transforms before the write:
   `count(merged.<i>.deltaT) == 12` (exactly 12 pairs merged),
   `not exists base::.*lora_.*`, `shape(h.0.attn.c_attn.weight) == [768, 2304]`,
   `count(base::.*) == 160`.
6. `output: { path: out/T5, format: safetensors, shard: 100MB }`.

## Independent verification (`out/verify.yaml`, no output written)

Re-opens the written checkpoint next to the untouched base and asserts:
160 tensors, no `lora_` names, all 148 non-`c_attn` tensors bit-equal to the
base, all 12 `c_attn` weights *not* equal to the base, and their shape/dtype
unchanged. All six assertions passed.
