# Participant self-report — T2-F-1

- **Final artifact path:** `out/T2/solution.py` (invoked by `out/T2/run.sh`), producing `out/T2/model.safetensors`.
- **Number of times you executed the script or plan:** 1.
- **Which executions failed, and why:** None; the single execution succeeded.
- **Pitfalls or surprises you hit:**
  - The query_key_value layout is per-head interleaved (768-row blocks each holding q/k/v for one head), not a global `[q|k|v]` split across all heads — slicing has to operate on the 768-row block for the pruned head, not on three separate 2048-row q/k/v regions.
  - `dense.weight` heads are column blocks (256 wide), the opposite axis from `query_key_value`'s row blocks, since it's the consumer of the head outputs rather than the producer.
- **Anything in the task text or documentation that was unclear:** No — the row/column ranges to keep were given explicitly in "Required result", which made the slicing unambiguous.
- **Tools used (condition F):** `safetensors==0.5.3` (load/save) and `torch==2.14.0` (tensor slicing/concatenation) via a plain script. Considered `transformers.prune_heads`, but GPT-NeoX's fused, interleaved qkv projection isn't representable by that generic API (it assumes separable per-head-dimension weight matrices, not a fused block layout), so a direct slice against the documented layout was the more reliable and auditable route. The script asserts input/output tensor counts and every per-layer output shape before writing, and I additionally verified bit-exactness of the pruned tensors and identity of all untouched tensors against the input file.
- **Approximate time spent:** ~10 minutes (mostly re-deriving and double-checking the row/column arithmetic from the spec).
