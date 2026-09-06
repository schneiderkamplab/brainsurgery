# T3 self-report

- Final artifact path: `out/T3/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The obvious hazard is regex overreach: `.*weight` would also hit
    `model.embed_tokens.weight`, `lm_head.weight` and the norms, so the pattern is
    anchored to `model.layers.<i>.{self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj}.weight`
    and the count is asserted to be exactly 112.
  - The two 412 MB embedding/lm_head tensors exceed the 256 MiB shard budget, so the
    budget assertion has to exempt single-tensor shards; `split_torch_state_dict_into_shards`
    already isolates them.
- Anything in the task text or documentation that was unclear:
  - The task does not say whether `config.json`/tokenizer files should be copied into
    `out/T3`; I copied them since they make the output loadable and grading is described
    as comparing tensors/sharding.
  - The exact shard-ordering convention is not specified; I used the HF helper with the
    input index key order, which yields 10 shards (2 single-tensor + 8 x 14 tensors).
- Tools used (condition F):
  - `torch` 2.14.0 — dtype cast (`.to(torch.bfloat16)`, round-to-nearest-even) and value checks.
  - `safetensors` 0.5.3 — shard load/save.
  - `huggingface_hub` (pinned) — `split_torch_state_dict_into_shards`, the same helper
    `transformers.save_pretrained` uses, so the shard layout and index format are canonical.
    I did not route through `transformers.save_pretrained` because it applies one dtype to
    the whole model; this task needs a per-tensor mix, which is simpler to express directly.
- Approximate time spent, if you can tell: about 5 minutes.
