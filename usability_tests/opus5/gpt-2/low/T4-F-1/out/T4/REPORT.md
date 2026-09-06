# T4 participant self-report

- **Final artifact path:** `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none; the single execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - The MLP name list has to be built explicitly rather than by regex: `mlp.c_proj` and `attn.c_proj` share a suffix, so a loose pattern would overreach onto attention.
  - The stated ordering hazard is real but avoided structurally: both task vectors are taken against `base[X]` read once into a local, so the first merge cannot contaminate the second.
  - Checkpoints are already float32, so the `.to(torch.float32)` casts are no-ops here, but they keep the arithmetic correct if the dtype ever changed.
  - `safetensors` rejects non-contiguous or shared storage, so every output tensor is cloned/made contiguous before saving.
- **Anything in the task text or documentation that was unclear:** nothing material. The spec says "identical in all three" for non-MLP tensors; I read that as bit-exact and used `torch.equal`, which is consistent with the grader's bit-exactness requirement for the 112 unchanged tensors. It does not say whether MLP tensors are *required* to differ, so I did not assert that.
- **Tools used (condition F):**
  - `torch` 2.14.0 — tensor arithmetic and bit-exact comparison (`torch.equal`).
  - `safetensors` 0.5.3 — `load_file` / `save_file` for direct, lossless tensor I/O.
  - I deliberately did **not** use mergekit, despite it being the suggested route for T4. Its task-arithmetic path goes through HuggingFace model loading and a YAML config, which would make the required pre-verification (that all 112 non-MLP tensors are bit-identical across three checkpoints, aborting otherwise) awkward to express and hard to make fail loudly; it also risks re-serializing untouched tensors through a dtype/format path rather than copying them verbatim. A ~90-line script over raw safetensors gives exact control over which 48 tensors are touched, keeps the other 112 byte-identical, and lets each required check be an explicit abort. peft/torch-state-bridge are irrelevant here (no adapters, no key renaming).
- **Approximate time spent, if you can tell:** ~5 minutes.

## Checks enforced by the run

All raise and abort the run (non-zero exit) if violated:

1. `ft1` and `ft2` key sets equal the base key set.
2. All 48 expected MLP names exist in the base.
3. Every tensor outside those 48 is bit-identical, and same shape/dtype, in all three checkpoints.
4. The 48 MLP tensors match in shape/dtype across the three checkpoints.
5. Exactly 48 tensors were merged.
6. The in-memory output has exactly 160 tensors.
7. After writing, the file is re-read: 160 tensors, key set matches the base, and no non-MLP tensor was modified.
