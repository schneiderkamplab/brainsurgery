# T2 — Participant self-report (condition P)

- **Final artifact path:** `out/T2/solution.py` (output: `out/T2/model.safetensors`)

- **Number of times you executed the script or plan:** 1

- **Which executions failed, and why (one line each):**
  - None. The single execution succeeded.

- **Pitfalls or surprises you hit (one line each):**
  - The obvious trap is reading `query_key_value` as `[q | k | v]` segments; GPT-NeoX interleaves per head, so head 5 is one contiguous 768-row block (rows 3840..4607) rather than three 256-row strips at 1280/3328/5376 — I derived the indices from the head geometry and then asserted them against the literal ranges in TASK.md so a wrong reading could not reach the output.
  - `attention.dense` is `[out, in]`, so its heads are on the **column** axis; slicing dim 0 there would have produced a file with plausible-looking shapes for the qkv tensors and a silently wrong output projection.
  - I initially wrote a blanket "every tensor must be float16" check and caught it before running: GPT-NeoX checkpoints carry non-fp16 attention buffers (`attention.bias`, `masked_bias`, `rotary_emb.inv_freq`), so that check would have false-failed. Replaced it with a per-tensor dtype-preservation check against the input, plus an fp16 assertion restricted to the 48 edited projections.
  - `safe_open(...).metadata()` returns `{'format': 'pt'}` on this checkpoint; `safetensors.torch.save_file` drops metadata unless you pass it explicitly, so I carried it over rather than silently changing the header.
  - `index_select` returns a fresh tensor, which sidesteps the shared-storage rejection `save_file` raises for views; I called `.contiguous()` anyway to be explicit.

- **Anything in the task text or documentation that was unclear:**
  - The task specifies key set, shapes, dtypes and values for grading but says nothing about the safetensors `__metadata__` header; I preserved the input's verbatim, which seemed the safer of the two readings.
  - Nothing else. The explicit statement of the interleaved layout plus the literal keep-ranges removed the one genuine ambiguity in the task; without that sentence the qkv layout would have needed to be inferred.

- **Tools used (condition F):** n/a — condition P (torch 2.14.0+cu130, safetensors 0.5.3, stdlib only).

- **Approximate time spent, if you can tell:** ~5 minutes, most of it reading the layout spec and reviewing the script before running it.

## What the script checks before writing

- input has 244 tensors and all 48 head-bearing tensors are present;
- derived keep-indices equal rows `0..3839 + 4608..6143` and columns `0..1279 + 1536..2047`;
- every source tensor has its documented pre-prune shape (`[6144, 2048]`, `[6144]`, `[2048, 2048]`);
- exactly the 48 expected tensors were edited, no more;
- the three required layer-0 shapes, and the same three shapes on all 16 layers;
- `attention.dense.bias` still `[2048]`;
- no tensor changed dtype;
- output has exactly 244 tensors, re-verified by reopening the written file.
