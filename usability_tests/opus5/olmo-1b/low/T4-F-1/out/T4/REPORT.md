# T4 — Participant self-report

- **Task / condition / repeat:** T4 / F (free choice of tooling) / 1
- **Approach:** a single plain Python script, `out/T4/solution.py`, on top of
  `safetensors` + `torch`.
- **Tools used:**
  - `safetensors` 0.5.3 — lazy `safe_open` reads of the sharded base
    directory (via `model.safetensors.index.json`) and the two single-file
    fine-tunes, and `save_file` for the output.
  - `torch` 2.14.0 — float32 arithmetic and `torch.equal` for the bit-exact
    comparison of the 66 shared tensors.
- **Why these tools:** mergekit 0.1.4 has a `task_arithmetic` method that
  matches the arithmetic, but it goes through a full HF model load/save, it
  applies the task vector to *every* tensor rather than a named subset, and
  it has no place to express the required preconditions. The task's checks
  (identical key sets, every non-MLP tensor bit-identical across all three
  checkpoints, exactly 48 merged, exactly 114 written) are the substance of
  this task, so a script that asserts them directly was both shorter and
  safer than configuring a merge toolkit and then re-verifying its output.
  torch-state-bridge and peft do not apply: nothing is being renamed and
  there is no adapter here.
- **How the required checks are enforced:** all four are `raise SystemExit`
  paths in `solution.py`, so a violation aborts before or during the write:
  1. key sets of base/ft1/ft2 must be equal and of size 114; the 48 expected
     MLP names must be present; per-tensor shape and dtype must agree.
  2. every tensor outside the MLP set must satisfy
     `torch.equal(base, ft1) and torch.equal(base, ft2)`.
  3. the merged counter must equal 48.
  4. the output dict must have 114 entries, and the file is reopened after
     writing and its key set re-checked against the base's.
- **Ordering hazard:** both task vectors are computed against the freshly
  read, unmodified `base[X]`; the accumulator `out[name]` is only assigned
  once, so no partial merge can feed into the second task vector.
- **Executions / attempts:** 1 execution of `solution.py`, succeeded.
  (Plus two read-only verification snippets that do not write output; the
  first of those had a wrong expected coefficient in my own check —
  `0.6*base` instead of `0.2*base` — which was an error in the check, not in
  the solution, and was corrected in the second.)
- **Result:** `out/T4/model.safetensors`, 114 tensors, float32. Independent
  re-check: max relative Frobenius error over the 48 merged tensors
  5.4e-08, and all 66 unchanged tensors bit-identical to the base.
- **Pitfalls hit:** the base is sharded, so the reader has to resolve names
  through the index's `weight_map` rather than assuming one file; and
  writing safetensors requires contiguous, unshared tensors, so copied
  tensors are `.clone().contiguous()`.
- **Time:** a few minutes; no retries of the solution itself.
