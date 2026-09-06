# T4 run record — participant self-report

- **Final artifact path:** `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none — the single execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - The checkpoints use raw GPT-2 keys (`h.<i>.mlp...`), with no `transformer.` prefix, so mergekit's task-arithmetic path (which wants a loadable HF repo/model) was more friction than value here.
  - The ordering hazard the task warns about: both task vectors must be differences against the *unmodified* base, so I read base/ft1/ft2 lazily from the three files and never mutate a base tensor in place.
  - `attn.bias` mask buffers live in the 112 "unchanged" tensors and must be copied bit-exactly — they are handled by the same base-clone path as every other non-MLP tensor, not regenerated.
  - Verification of "identical outside the MLPs" has to be bit-exact (`torch.equal`), not `allclose`, or a small genuine drift would pass silently.
- **Anything in the task text or documentation that was unclear:**
  - The task says "exactly 48 tensors were merged". A task vector could in principle be numerically zero for some tensor, so I check that 48 tensors went through the merge computation, and separately that at most 48 tensors differ from the base and that none of them is outside the MLP set. (In this data all 48 do in fact differ.)
- **Tools used (condition F):**
  - `safetensors` 0.5.3 — lazy per-tensor read (`safe_open`) of three 548 MB checkpoints and the final `save_file`; avoids holding three full state dicts in memory.
  - `torch` 2.14.0 — float32 tensor arithmetic and bit-exact comparison (`torch.equal`).
  - Considered and rejected: `mergekit` 0.1.4 task arithmetic — it operates on HF model repos and its own dtype/output conventions, and it would not by itself enforce the three required checks (shared-tensor identity, exactly 48 merged, exactly 160 out); a ~120-line script over safetensors expresses the checks directly and keeps the 112 unchanged tensors provably bit-exact. `transformers`/`peft` are not relevant to a plain task-vector merge of raw state dicts.
- **Approximate time spent:** ~5 minutes.

## What the script enforces (fails loudly, non-zero exit, via `CheckFailed`)

1. Base/ft1/ft2 key sets identical, and 160 tensors.
2. The 48 expected MLP names all present; the remaining 112 tensors have matching shape and dtype and are **bit-exactly equal** across all three checkpoints.
3. Shapes match and dtype is float32 for each merged MLP tensor.
4. Exactly 48 tensors merged; output dict has exactly 160 tensors with the input key set.
5. Post-write re-read of `model.safetensors`: 160 tensors, key set matches, shapes/dtypes unchanged, and every tensor that differs from the base is one of the 48 MLP tensors.
