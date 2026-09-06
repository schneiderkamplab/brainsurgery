# T2 report — condition F, GPT-2 (124M)

## Participant self-report

- **Final artifact path:** `out/T2/solution.py` (invoked via `out/T2/run.sh`), output at `out/T2/model.safetensors`.
- **Number of times you executed the script or plan:** 1.
- **Which executions failed, and why (one line each):** none; the single execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - `transformers` 5.12.1's `GPT2Attention`/`GPT2PreTrainedModel` has no `prune_heads`/`_prune_heads` implementation, so the toolkit route suggested in `F-allowed.md` (`prune_heads`) doesn't actually exist for GPT-2 in this version — checked directly rather than discovering it via a failed run.
  - Easy to conflate `attn.c_proj.bias` (untouched, output-dim, not per-head) with `attn.c_proj.weight` (row-pruned); and `attn.bias` (the causal-mask buffer, untouched) with `attn.c_attn.bias`/`attn.c_proj` — kept the string matching on exact `.endswith(...)` suffixes rather than a substring/regex to avoid overmatching.
  - Conv1D `[in, out]` layout means "heads are column blocks" for `c_attn.weight` and "row blocks" for `c_proj.weight" — different axis per tensor, easy to swap by habit if used to `nn.Linear`'s `[out, in]`.
- **Anything in the task text or documentation that was unclear:** no — the required column/row ranges were given explicitly in TASK.md and matched a from-scratch derivation from head index, head_dim and hidden size.
- **Tools used (condition F): name, version, and why:**
  - `torch` 2.14.0 and `safetensors` 0.5.3 only, via a plain script. Considered `transformers.PreTrainedModel.prune_heads` (not implemented for GPT-2 in this version — verified before use), `mergekit` (operates on whole tensors/layers, not sub-tensor column/row slices inside a fused QKV block), and `torch-state-bridge` (key rewriting/renaming, not intra-tensor slicing) — none fit the required per-head slice-and-splice inside `c_attn`/`c_proj`, so a direct index-select script gave the most control and was easiest to verify against the spec.
- **Approximate time spent, if you can tell:** single short session; no retries needed.

## Verification performed beyond the required checks

In addition to the four required shape/count checks enforced inline in
`solution.py` (which abort before writing if violated), independently
recomputed the exact keep-column/keep-row index lists from TASK.md's spec in
a separate check script and confirmed the output is bit-exact against that
reconstruction for all 12 layers, and that every non-attention tensor
(`ln_*`, `mlp.*`, `attn.c_proj.bias`, `attn.bias` mask buffer, and any
top-level non-`h.*` tensors) is byte-identical to the input.
