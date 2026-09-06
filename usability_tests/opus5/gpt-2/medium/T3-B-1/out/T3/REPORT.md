# T3 self-report (condition B: BrainSurgery plan)

- **Final artifact path:** `out/T3/plan.yaml` (output checkpoint in `out/T3/`)

- **Number of times you executed the script or plan:** 2 executions of
  `out/T3/plan.yaml`, plus one throwaway dump-only plan (`out/dump.yaml`) used
  to list the tensor names before writing anything.

- **Which executions failed, and why (one line each):**
  1. `failed_assertion` / `no_match` — `assert: { count: { of: 'h\.\d+\.attn\.bias', is: 0 } }`
     aborted with `count.of matched zero tensors`: a reference that matches nothing is a
     hard error in the resolver, so `count ... is: 0` can never succeed; replaced with
     `not: { exists: ... }`. Nothing was written, so no partial output.
  2. passed.

- **Pitfalls or surprises you hit (one line each):**
  - `count: is: 0` is unusable as an "it is gone" check; the reference resolver raises on
    zero matches before the count is compared. `not: { exists: ... }` is the working form.
  - The same failing run also carried a wrong hand-computed bias count (74 instead of 73);
    I had forgotten that `h.<i>.attn.bias` had just been deleted. Fixed in the same edit.
  - Targeting hazards were as advertised but easy to avoid with full-match regexes:
    `h\.\d+\.attn\.bias` does not touch `h.<i>.attn.c_attn.bias`, and the projection
    pattern `h\.\d+\.(?:attn\.(?:c_attn|c_proj)|mlp\.(?:c_fc|c_proj))\.weight` does not
    touch `wte`/`wpe`/layer norms. `mlp.c_proj` and `attn.c_proj` both had to be listed.
  - The "everything else is float32" half of the "exactly 48 bfloat16" check needed a
    negative lookahead (`(?!...$).*`); there is no dtype-based tensor selector, so the
    only way to bound the bf16 set from above is to assert the complement is float32.
  - Sharding needed no special handling: `output.shard: 64MB` is binary (64 MiB), and the
    oversized `wte.weight` was placed alone in its own shard automatically.
  - My first dump-only plan wrote into `out/T3/`; I moved it to `out/` so the output
    directory contains only checkpoint files.

- **Anything in the task text or documentation that was unclear:**
  - The docs do not say that a zero-match reference is an error rather than an empty
    result, which is what made the `count: is: 0` idiom fail.
  - The README documents shard packing as "state-dict order", but not whether `delete`
    and `cast_` preserve that order; the task's bit-exact sharding comparison depends on
    it. It did hold here.

- **Tools used (condition F):** n/a.

- **Approximate time spent, if you can tell:** a few minutes; roughly two thirds of it
  reading `README.md` and `help.txt` before writing the plan.

## Verification performed

After the passing run I re-read the output (outside the plan) and confirmed: 148 tensors,
48 bfloat16 and 100 float32, no `h.<i>.attn.bias`, names and shapes identical to the
input, values bit-exact against `src.to(torch.bfloat16)` / `src`, 4 shards each within
64 MiB except the single-tensor `wte.weight` shard, and an index whose `weight_map`
covers all 148 names.
