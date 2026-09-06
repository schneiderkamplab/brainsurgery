# T2 self-report (condition F, GPT-2 124M)

- Final artifact path: `out/T2/solution.py` (writes `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - `transformers` 5.12.1 no longer has `prune_heads` / `prune_conv1d_layer`, so the route suggested in `F-allowed.md` for T2 does not exist in this environment.
  - `GPT2Attention` in transformers 5.12.1 derives head size as `n_embd / n_head`, so an 11-head model with 704-wide projections cannot even be instantiated for a functional check; I verified the layout with a small hand-written attention forward instead (pruned 11-head output equals the 12-head output with head 5 zeroed, max abs diff 0.0 in all 12 layers).
  - The mask buffer `attn.bias` shares the `.attn.` prefix with the real biases, so tensor selection has to key on `c_attn.bias`, not `attn.bias`.
- Anything in the task text or documentation that was unclear: nothing; the explicit column/row ranges made the layout unambiguous. `record-template.md` was referenced by the prompt but the template only exists at the sandbox root, which was fine.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: load/save the checkpoint (bit-exact, no shared-tensor issues since no tensor was aliased).
  - `torch` 2.14.0: `index_select` on the head axis with a keep-index built from `arange` / head-id mask, plus `torch.equal` checks.
  - `transformers` 5.12.1: attempted for an independent `prune_heads` cross-check only; not usable (see pitfalls), not part of the solution.
  - Not used: mergekit, peft, torch-state-bridge (no key rewriting or merging needed).
- Approximate time spent, if you can tell: about 5 minutes.

## Checks enforced by `solution.py` (run fails with `CHECK FAILED: ...` before writing)

- input has 160 tensors, destination does not already exist;
- every `h.<i>.attn.c_attn.weight` is `[768, 2304]` in and `[768, 2112]` out; bias `[2304]` -> `[2112]`; `c_proj.weight` `[768, 768]` -> `[704, 768]`; exactly 36 tensors touched, dtypes unchanged;
- required checks: `h.0.attn.c_attn.weight` `[768, 2112]`, `h.0.attn.c_attn.bias` `[2112]`, `h.0.attn.c_proj.weight` `[704, 768]`, output has exactly 160 tensors, key set identical;
- values of `h.0` sliced tensors equal the concatenation of the exact ranges listed in TASK.md; all non-head tensors bit-equal to the input;
- after writing, the file is reloaded and every tensor compared bit-exactly.
