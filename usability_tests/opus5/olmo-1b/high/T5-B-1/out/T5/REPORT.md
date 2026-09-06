# T5 self-report (condition B: BrainSurgery plan)

- **Final artifact path:** `out/T5/plan.yaml` (checkpoint written to `out/T5/`,
  index `out/T5/model.safetensors.index.json`). `out/T5/verify.yaml` is an
  extra read-only plan I wrote to double-check the result; it writes nothing.

- **Number of times you executed the script or plan:** 1 execution of
  `out/T5/plan.yaml` (passed on the first run). Plus 3 executions of the
  separate read-only `verify.yaml`, of which the first failed (see below).

- **Which executions failed, and why (one line each):**
  - `plan.yaml` execution 1: succeeded, no failures.
  - `verify.yaml` execution 1: `crash` / `PlanLoaderError: transform #3: unknown
    model alias: 'chk'` — a scratch alias must be created with
    `prefixes: { mode: add, alias: chk }` before it can be a destination.
  - `verify.yaml` executions 2 and 3: succeeded (3 was 2 plus two extra
    `count` asserts to prove the negative-lookahead pattern really selected 82
    and 32 tensors rather than passing vacuously).

- **Pitfalls or surprises you hit (one line each):**
  - Output alias inference: the intermediate `B @ A` products had to be created
    inside the `model` alias (under a `merge_delta.*` scratch prefix, deleted
    before writing), because writing to both `model` and `lora` would trip
    `cannot infer output model uniquely`.
  - `matmul` requires its destination to be new, `add_`/`scale_` require it to
    exist — so the sequence is matmul (new delta) -> `scale_` by 2 -> `add_`
    into the base weight -> `delete` the deltas, not a single fused step.
  - PEFT's `base_model.model.model.layers.<i>....` prefix means the adapter and
    base names differ by more than a prefix strip; a capture-group rewrite
    (`\1`, `\2`) across `from_a` / `from_b` / `to` handles it in one transform
    for all 32 pairs.
  - Assert patterns are full-match regexes over dotted names; unescaped dots
    would still match here, but I escaped them so `count` can't overmatch.
  - A `count`/`equal` assert over a pattern that matches nothing would pass
    silently, so I added explicit `count` asserts (32 pairs, 82 untouched, 114
    total) rather than relying on `equal` alone.
  - Shard budget: `512MB` in plan units is binary (536,870,912 bytes), which is
    exactly the required 512 MiB. Result is 10 shards; `model.embed_tokens.weight`
    ends up alone in shard 1, `lm_head.weight` shares shard 2 with one 67 MB MLP
    tensor and the shard still fits the budget.

- **Anything in the task text or documentation that was unclear:**
  - TASK.md says a tensor "larger than" 512 MiB is stored alone in its own
    shard and names `model.embed_tokens.weight` and `lm_head.weight` (412 MB
    each) as such tensors — but 412 MB is *smaller* than 512 MiB, so they are
    not oversized. The tool's documented greedy packing puts `embed_tokens`
    alone (the next tensor would overflow) and pairs `lm_head` with one more
    tensor. Every shard is within the 512 MiB budget, but the exact
    tensor-to-shard partition may differ from a reference built by a different
    packing order.
  - The README documents capture-group rewrite for `to` in `copy`/`move`; that
    it also applies to `from_b` in ternary transforms is only stated in
    `interfaces-reference.md`.
  - Nothing in the docs said a scratch alias must be registered with
    `prefixes: { mode: add }` before use as a destination.

- **Tools used (condition F):** n/a (condition B).

- **Approximate time spent, if you can tell:** roughly 10 minutes, most of it
  reading `help.txt`, the README and the example plans before writing anything.

## What the plan does

`W += (lora_alpha / r) * B @ A` with `scale = 32 / 16 = 2`, computed in float32.
`fan_in_fan_out = false` and both the base weights and the adapter factors use
the `nn.Linear` `[out, in]` layout, so `B @ A` (`[2048,16] @ [16,2048]` ->
`[2048,2048]`) is added with no transpose.

Required checks, all implemented as `assert` transforms in `plan.yaml`:

| Required check | Implementation |
|---|---|
| exactly 32 adapter pairs found and merged | `count` of `lora_A` = 32, `count` of `lora_B` = 32, `count` of `lora::.+` = 64, and `count` of the 32 computed `merge_delta.*` products = 32 (the matmul itself reported "affected 32 site(s)") |
| no `lora_` name in the output | `assert: { not: { exists: 'model::.*lora_.*' } }` (plus the same for `merge_delta`) |
| `model.layers.0.self_attn.q_proj.weight` still `[2048, 2048]` | `assert: shape ... is: [2048, 2048]`, before and after the merge |
| output has exactly 114 tensors | `assert: { count: { of: 'model::.+', is: 114 } }`, after the deletes |

Independent post-check (`verify.yaml`, read-only): the 82 untouched tensors are
bit-equal to the base; the 32 merged weights match a freshly recomputed
`base + 2 * B @ A` within `eps: 1e-4`; the merge is not a no-op; dtype is
float32; the output has 114 tensors and no `lora_` names. All asserts passed.
