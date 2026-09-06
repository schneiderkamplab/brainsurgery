# Participant self-report — T1 (condition F)

- **Final artifact path:** `out/T1/solution.py` (invoked via `out/T1/run.sh`), output at `out/T1/model.safetensors`.
- **Number of times you executed the script or plan:** 1.
- **Which executions failed, and why:** None failed.
- **Pitfalls or surprises you hit:**
  - Renumbering must be done as a single dict-based rename (old index -> new
    index computed from the full surviving-block list) rather than sequential
    in-place shifts, to avoid a block overwriting a not-yet-moved survivor
    (e.g. moving old 3 -> 2 before old 2 is deleted would collide).
  - The causal-mask buffer `attn.bias` is also named `h.<i>.attn.bias` and
    must be dropped/renumbered along with the rest of the block, not treated
    as a "non-block" tensor (the non-block tensors are only `wte.weight`,
    `wpe.weight`, `ln_f.weight`, `ln_f.bias`, none of which start with `h.`).
- **Anything in the task text or documentation that was unclear:** No,
  the explicit old->new index list in TASK.md removed any ambiguity about
  ordering.
- **Tools used (condition F):** Plain script on top of `safetensors` 0.5.3
  (`safetensors.torch.load_file`/`save_file`) and `torch` 2.14.0 tensor ops
  (`.clone().contiguous()`). Chose a direct script over `mergekit`'s
  passthrough layer-slicing or `torch-state-bridge`'s regex rewriting because
  the required checks (exact key set, block count, tensor count, collision
  detection) are simple to express directly against a `dict[str, Tensor]`
  and I wanted the collision/failure-mode guarantees (fail loudly, no
  partial output) to be explicit and auditable in one small script rather
  than mediated by a merge-config DSL.
- **Approximate time spent, if you can tell:** ~10 minutes.
