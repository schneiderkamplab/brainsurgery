# T3 participant self-report

- **Final artifact path:** `out/T3/solution.py` (invoked via `out/T3/run.sh`), output written to `out/T3/` (9 shards + `model.safetensors.index.json`).
- **Number of times you executed the script or plan:** 1.
- **Which executions failed, and why:** none; the first execution succeeded.
- **Pitfalls or surprises you hit:**
  - The task text describes `embed_in.weight`/`embed_out.weight` as "206 MB each" — that's their *input* float16 size. Once upcast to float32 (required, since embeddings must end up float32) they're ~412 MB / ~393 MiB each, which is what actually pushes them over the 256 MiB shard budget and forces each into its own shard. Worth computing shard sizes from the *output* dtype, not the input one.
  - `masked_bias` is a scalar (`[]` shape) stored as F16, not U8 like `attention.bias`; the three deleted buffers per layer don't share a dtype, so I matched by exact key name rather than by dtype/shape heuristics.
- **Anything in the task text or documentation that was unclear:** the "206 MB each" figure for the embedding tensors could be misread as the *output* size; it's the input size. No other ambiguity — the per-layer key lists and required checks were exact enough to write directly against.
- **Tools used (condition F):** `torch` 2.14.0 and `safetensors` 0.5.3 only, via a plain script. No merge toolkit or HF sharded-save helper was used: `mergekit`'s task-arithmetic/passthrough configs don't express "these named tensors -> bf16, everything else -> fp32, drop these buffers" as a single declarative op, and `transformers`' sharded `save_pretrained` doesn't offer a per-tensor dtype policy. A ~150-line script against `safetensors.torch.load_file`/`save_file` gave direct, auditable control over dtype casting, buffer deletion, the required pre-write checks, and the greedy byte-budget sharding, and was verified bit-exact against the source afterward.
- **Approximate time spent:** ~10 minutes (inspecting the input header, writing the script, running it once, and verifying dtype/shape/bit-exactness against the source).
