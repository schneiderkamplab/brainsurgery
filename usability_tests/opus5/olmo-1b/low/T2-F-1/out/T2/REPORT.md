# T2 — Participant self-report

- **Final artifact path:** `out/T2/solution.py` (output: `out/T2/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none; the single execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - The layout is the whole task: q/k/v are `[out, in]` so heads are row blocks, while `o_proj` consumes the head dimension on its input axis, so the same head is a *column* block there — mixing those up would still produce loadable, wrong tensors.
  - `index_select` returns a fresh tensor, but I called `.contiguous()` anyway so `save_file` never sees a view/shared-storage tensor.
  - The input is sharded across two files, so the state dict had to be merged (with a duplicate-key assert) before slicing and re-saved as a single file.
  - I checked `prune_heads` from transformers (the route the condition notes suggest) and rejected it: it zero-ablates/reshapes via the model class and rewrites `config` head counts and `pruned_heads` bookkeeping, which does not give the bit-exact, name-preserving 114-tensor checkpoint this task grades on.
- **Anything in the task text or documentation that was unclear:** nothing. The kept row/column ranges were given explicitly, which removed all ambiguity about head ordering.
- **Tools used (condition F):**
  - `safetensors` 0.5.3 — load the two input shards and write the single output file; the task is expressed directly in tensor terms.
  - `torch` 2.14.0 — `index_select` for the keep-index slicing and `torch.equal` for a bit-exact spot check of the kept blocks.
  - No merging/adapter toolkit was used: mergekit and peft operate on models/adapters, not on a single-tensor-axis structural edit, and would have added a config-rewriting layer this task does not want.
- **Approximate time spent:** ~5 minutes.

## Checks enforced by the run

Assertions fire before anything is written: input tensor count is 114; every
matched q/k/v and o_proj tensor is `[2048, 2048]` on input; exactly 48 q/k/v
and 16 o_proj tensors match (guards against under- and over-matching);
layer-0 q/k/v are `[1920, 2048]` and layer-0 o_proj is `[2048, 1920]`; output
has exactly 114 tensors with an unchanged key set; and the kept row/column
blocks of layer 0 are bit-identical to the source.
