# T4 — Participant self-report

- **Final artifact path:** `out/T4/solution.py` (run via `out/T4/run.sh`);
  output at `out/T4/model.safetensors`.
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none — the single execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - Ordering hazard: both task vectors must be taken against the *unmodified* base, so the script loads the base tensor fresh per name and never writes into it — a naive in-place `base += lambda*(ft1-base)` would fold the first merge into the second.
  - The base is sharded (two files + index) while the fine-tunes are single-file, so the reader had to handle both layouts before any comparison was possible.
  - The three checkpoints are ~5 GB each in float32; loading them whole would be ~15 GB, so everything streams tensor-by-tensor through `safe_open`.
  - "Identical" for the 66 backbone tensors is checked bitwise (`view(torch.uint8)`), not with `==`, so NaN and `-0.0` cannot silently pass as equal.
  - `safe_open(...).get_slice(...).get_dtype()` returns safetensors' own dtype names (`"F32"`), not torch names — needed an explicit mapping table.
  - `lm_head.weight` and `model.embed_tokens.weight` are untied here (`tie_word_embeddings: false`) and are read as independent tensors, so safetensors' shared-storage rejection never triggered.
- **Anything in the task text or documentation that was unclear:** nothing
  blocking. Two points I resolved by choice: the task fixes the output as a
  single `model.safetensors` and says nothing about copying `config.json` /
  tokenizer files, so I wrote only the tensor file; and the merged tensors are
  already float32 in all three inputs, so "computed in float32" needed no dtype
  promotion (the script asserts float32 rather than assuming it).
- **Tools used (condition F): name, version, and why:**
  - `safetensors` 0.5.3 — lazy `safe_open` for per-tensor streaming reads of the sharded base and the two fine-tunes, and `save_file` for the single-file output. Chosen because it is the only allowed package that gives random per-tensor access without materialising a 5 GB checkpoint.
  - `torch` 2.14.0 — tensor arithmetic for the task vectors, bitwise equality via `view(torch.uint8)`, and Frobenius norms for the post-write check.
  - Considered and rejected: **mergekit** 0.1.4 `task_arithmetic`, which is the
    obvious catalogue answer for this task and computes the right formula. I did
    not use it because it does not express this task's actual requirements: it
    cannot verify the frozen-backbone precondition (that all 66 non-MLP tensors
    are identical across the three checkpoints) and would silently produce a
    merge if that assumption were false; it has no notion of "exactly 48 tensors
    were merged"; and it emits a sharded HF model directory with its own dtype
    and config handling rather than the single `out/T4/model.safetensors` the
    task requires. Every required check would have had to be written as a
    separate script anyway, at which point the merge itself is three lines.
    **transformers** `save_pretrained` was rejected for the same output-shape
    reason (sharding, tied-weight rewriting), and **peft** does not apply — these
    are full fine-tunes, not adapters.
- **Approximate time spent, if you can tell:** ~8 minutes end to end; the merge
  run itself is 18 s wall clock, including reading ~15 GB and the post-write
  re-verification pass.

## How the required checks are enforced

All checks raise `CheckFailed` and exit non-zero; there is no fallback path.

1. **Shared-tensor verification (step 1)** — before any arithmetic:
   name sets of base/ft1/ft2 must be equal and of size 114; the 48 MLP names the
   task lists must all be present; shape and dtype must agree per name across
   all three; and each of the 66 non-MLP tensors must be bit-identical in all
   three checkpoints.
2. **Exactly 48 tensors merged** — a counter incremented in the merge loop is
   asserted `== 48`, and the post-write pass independently recounts how many
   output tensors differ from the base and asserts that count is also 48.
3. **Output has exactly 114 tensors** — asserted on the in-memory dict before
   writing and again by reopening the written file, which must also have exactly
   the base's tensor names, shapes and dtypes.

The post-write pass additionally recomputes the merge formula for each MLP
tensor and requires a relative error `<= 1e-6` against what was written, and
requires the other 66 tensors to be bit-identical to the base.
