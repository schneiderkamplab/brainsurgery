# T2 self-report (condition F, GPT-2 124M)

- **Final artifact path:** `out/T2/solution.py` (output: `out/T2/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none; the first execution produced the output.
- **Pitfalls or surprises you hit:**
  - `F-allowed.md` names `transformers` `prune_heads` as the plausible route for T2, but transformers 5.12.1 has removed that API: neither `GPT2Model.prune_heads` nor `GPT2Attention.prune_heads` exists (both checked). The documented route is dead in the pinned environment.
  - Even if it existed, the HF route could not satisfy this task: the checkpoint has flat keys (`h.0.attn.c_attn.weight`, no `transformer.` prefix) and includes the 12 `h.<i>.attn.bias` causal-mask buffers. `from_pretrained` loads only 148 of the 160 tensors — the mask buffers are non-persistent in v5 — so a `save_pretrained` round-trip would drop 12 tensors and re-prefix the rest, breaking "exactly 160 tensors, names unchanged".
  - Conv1D `[in, out]` layout is the real trap: `c_attn` heads are *column* blocks (slice dim 1) while `c_proj` heads are *row* blocks (slice dim 0). The two head-bearing tensors of the same module are sliced on opposite axes.
  - `c_attn` is fused `[q | k | v]`: the 64-column hole must be punched three times, at offsets 0, 768 and 1536, and the three segments must stay in q, k, v order — building one 704-index pattern and repeating it per segment keeps that ordering by construction.
  - `attn.c_proj.bias [768]` and `attn.bias [1,1,1024,1024]` sit under the same `h.<i>.attn.` prefix as the tensors that must change; matching on the prefix alone would have corrupted them. I matched on exact suffixes and asserted the input shape before every slice.
- **Anything in the task text or documentation that was unclear:** nothing in TASK.md — the explicit keep-ranges made the spec unambiguous, and I used them as an independent oracle. Only `F-allowed.md` was misleading, in pointing at an API the pinned `transformers` no longer ships.
- **Tools used (condition F):**
  - `safetensors` 0.5.3 — `safe_open` / `save_file` for exact tensor-level read and write, preserving the key set, dtypes and file metadata verbatim.
  - `torch` 2.14.0 — `index_select` for the gather-based slicing and `torch.equal` for bit-exact verification.
  - `transformers` 5.12.1 — used only to *test* the advertised `prune_heads` route; rejected for the reasons above.
  - No higher-level tool fit. mergekit slices layers, not head blocks inside a fused projection; torch-state-bridge rewrites keys, and here the keys are unchanged and only values are. The task is a pure intra-tensor slice, which is below the granularity every allowed toolkit operates at.
- **Verification performed:** independently of the solution, I rebuilt the keep-index from the literal ranges in TASK.md (`0..319, 384..767, 768..1087, 1152..1535, 1536..1855, 1920..2303` and `0..319, 384..767`), and checked that the output has the same 160 keys, identical dtypes, the three required shapes on *every* layer (not just layer 0), the expected values bit-exactly on all 36 modified tensors, and bit-exact equality with the input on all 124 untouched tensors.
- **Approximate time spent:** about 5 minutes.
