# T2 participant self-report (Condition F)

- **Final artifact path:** `out/T2/solution.py` (invoked via `out/T2/run.sh`), output at `out/T2/model.safetensors`.
- **Number of times you executed the script or plan:** 1.
- **Which executions failed, and why:** none; the single execution succeeded.
- **Pitfalls or surprises you hit:**
  - None specific to this task — the head boundary (`128 * head_index` for a 128-dim head) and row-vs-column block convention (row blocks for q/k/v since heads are stacked in the output dimension of `[out, in]`, column blocks for o_proj since heads are stacked in its input dimension) were given explicitly in TASK.md, so no discovery was needed.
  - Loaded all tensors from both shards via `model.safetensors.index.json` up front rather than trying to reason about which shard held which tensor, since the task only cares about the merged key set.
- **Anything in the task text or documentation that was unclear:** no, the row/column ranges to keep (`0..639`, `768..2047`) were spelled out per tensor, which made this closer to a transcription-and-check task than a discovery task.
- **Tools used (condition F):** `torch` 2.14.0 and `safetensors` 0.5.3 only, as a plain script — no `transformers.prune_heads`, `mergekit`, `peft`, or `torch-state-bridge`. Chose plain tensor slicing over `transformers.prune_heads()` because the required output is a raw safetensors file with unchanged tensor names, not a re-saved HF model export, and I wanted the row/column boundaries and the merged 114-tensor set to be visibly asserted in the script (per "Required checks") rather than delegated to library internals I'd have to trust and re-verify anyway. `torch.cat` slicing is also simplest to verify bit-exactly against the source tensors, which I did with an explicit spot-check assertion before writing.
- **Approximate time spent:** a few minutes (single write-and-run pass, no debugging needed).
