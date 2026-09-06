# Participant self-report — T4 (condition F)

- **Final artifact path:** `out/T4/solution.py` (invoked via `out/T4/run.sh`)
- **Number of times you executed the script or plan:** 1 (the run against the
  real inputs succeeded on the first attempt; I separately smoke-tested the
  abort path against a hand-corrupted copy of the inputs in a scratch
  directory outside `out/`, which is not counted as an attempt on this task).
- **Which executions failed, and why:** none, on the real inputs.
- **Pitfalls or surprises you hit:**
  - `inputs/base` is sharded (`model.safetensors.index.json` + two shard
    files) while `inputs/ft1`/`inputs/ft2` are single `model.safetensors`
    files, so the loader has to handle both layouts.
  - The frozen-backbone precondition has to be checked with `torch.equal`
    (dtype + shape + bit-exact values), not `torch.allclose`, since the
    grader requires bit-exact preservation of the 66 unchanged tensors.
- **Anything in the task text or documentation that was unclear:** no.
- **Tools used (condition F):** `torch` 2.14.0 and `safetensors` 0.5.3
  directly, in a plain script — no `mergekit`/`torch-state-bridge`/`peft`.
  I chose a direct script over `mergekit`'s task-arithmetic merge method
  because this task's required checks (verify identical non-MLP tensors
  across all three checkpoints before touching anything, abort loudly,
  assert exactly 48 merged / 114 total) needed to be first-class,
  fail-loud steps under my control, and the arithmetic itself
  (`base + lambda*(ft1-base) + lambda*(ft2-base)`, float32, per-tensor) is
  a few lines with `safetensors`/`torch` — a general-purpose merge config
  would have added a dependency without buying reliability for such a
  narrow, precisely specified transform.
- **Approximate time spent:** ~10 minutes (read task + inputs, write and
  run the script, verify output bit-exactness and the abort path).
