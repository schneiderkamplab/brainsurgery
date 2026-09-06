# T4 — participant self-report

- **Final artifact path:** `out/T4/solution.py` (entry point `out/T4/run.sh`);
  output at `out/T4/model.safetensors`. `out/T4/verify.py` is an independent
  post-hoc checker and does not produce the output.

- **Number of times you executed the script or plan:** 1
  (one execution of `out/T4/run.sh`, which succeeded).

- **Which executions failed, and why:** none.

- **Pitfalls or surprises you hit:**
  - The tensor names have no `transformer.` prefix (`h.0.mlp.c_fc.weight`, not
    `transformer.h.0...`), so this is a bare `GPT2Model` state dict; anything
    routed through `transformers`/`mergekit` would have had to rename keys
    back on the way out. Checked the key list before writing any code.
  - The ordering hazard is real but avoidable by construction: I compute
    `base + λ*(ft1-base) + λ*(ft2-base)` from the three source tensors in a
    single expression and only then store it, so no intermediate ever becomes
    the base for the second task vector. I never mutate a loaded base tensor
    in place.
  - `h.<i>.attn.bias` is a causal-mask buffer, not a weight; it sits in the
    112 "unchanged" tensors and is copied bit-exact, which the grading wants.
    A merge tool that filters by "is this a parameter" could have dropped it.
  - Conv1D layout: GPT-2's `c_fc.weight` is `[768, 3072]` (`[in, out]`), not
    the Linear `[out, in]`. Irrelevant for elementwise task arithmetic, but it
    would have mattered had I tried to reshape anything — I did not.
  - Guarding against a vacuous pass: my verifier also asserts each of the 48
    merged tensors actually differs from the base, so a run that silently
    copied everything could not report success.

- **Anything in the task text or documentation that was unclear:** nothing
  material. Two small things I decided myself: the output is only
  `model.safetensors` (no config/tokenizer copied, as the task names a single
  file), and I wrote `{"format": "pt"}` metadata to match the inputs.

- **Tools used (condition F):**
  - `safetensors` 0.5.3 — load/save. Used directly because the task is defined
    on exact tensor names and requires bit-exact preservation of 112 tensors;
    reading and writing the state dict as-is is the only route that guarantees
    the key set and dtypes survive untouched.
  - `torch` 2.14.0 — float32 arithmetic and `torch.equal` for the bit-exact
    backbone comparison.
  - **`mergekit` 0.1.4 — considered and rejected.** Its `task_arithmetic`
    method is the nominal route for this task, but it applies the merge to
    *every* tensor rather than to a named subset of 48, it loads and re-saves
    through `transformers` (so the bare `h.*` key names would be rewritten and
    the output resharded), and it has no notion of the precondition this task
    centres on — verifying that the backbone is identical across all three
    checkpoints and aborting if not. Bending it into shape would have cost
    more than the 90-line script and would have made the required checks
    weaker, not stronger.
  - **`torch-state-bridge` 0.1.0 — not applicable**; no key rewriting is
    needed here, the names are unchanged.

- **Approximate time spent:** ~5 minutes.

## What the run enforces

`out/T4/solution.py` raises `CheckFailed` (exit code 1, nothing written) on:

1. key sets of base/ft1/ft2 differing, or the base not having exactly 160 tensors;
2. any of the 48 expected MLP names being absent;
3. any per-tensor shape or dtype mismatch between the three checkpoints;
4. any non-MLP tensor not being bit-identical in all three (the frozen-backbone
   precondition, step 1 of the task);
5. merged count != 48, copied count != 112, or output tensor count != 160;
6. post-write re-read: the file on disk not having the same 160 keys, or a
   non-MLP tensor on disk not being bit-identical to the base.

`out/T4/verify.py` additionally recomputes the 48 merged tensors with an
algebraically different expression, `(1-2λ)·base + λ·ft1 + λ·ft2`, and confirms
the worst relative Frobenius error is **6.901e-08** (tolerance 1e-5); it also
runs a negative test on synthetic checkpoints with one tampered backbone tensor
and confirms the solution aborts.
