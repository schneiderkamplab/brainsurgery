# T3 (OLMo-1B-0724-hf, condition B) — participant self-report

- Final artifact path: `out/T3/plan.yaml` (output checkpoint in `out/T3/`: `model-0000{1..10}-of-00010.safetensors` + `model.safetensors.index.json`; executed-transform summary in `out/T3/summary.yaml`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - There is no assert operator that counts tensors *by dtype*, so "exactly 112 tensors are bfloat16" had to be expressed as: the 112-match projection pattern is bfloat16 AND its negative-lookahead complement (2 tensors) is float32 AND the total is 114.
  - Tensor references are full-match regexes, so dots had to be escaped and the projection set spelled out (`self_attn.[qkvo]_proj` / `mlp.(gate|up|down)_proj`) instead of a broad `.*weight`, which would also have hit `embed_tokens` and `lm_head`.
  - Shard budget units are binary (`256MB` = 268,435,456 bytes), which matched the task's 256 MiB requirement exactly; the two 412 MB embedding matrices were placed alone in their own shards automatically.
- Anything in the task text or documentation that was unclear:
  - Whether `assert: dtype` checks *every* matched tensor of a multi-match pattern or only the first; the help text says "the tensor". I paired it with a `count` on the same pattern, and the run confirmed 112 sites were cast, so it behaved as all-match.
  - The README/help do not state whether an input path may be a sharded directory with an index; the log confirmed it is supported (`Detected safetensors index`).
- Tools used (condition F): n/a (condition B)
- Approximate time spent, if you can tell: about 5 minutes (reading the doc pack, one plan run of ~6 s, independent verification of the output).
