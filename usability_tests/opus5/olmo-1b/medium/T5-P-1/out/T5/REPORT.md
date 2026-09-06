# T5 self-report (condition P)

- **Final artifact path:** `out/T5/solution.py`; output checkpoint in `out/T5/`
  (10 shard files + `model.safetensors.index.json`).
- **Number of times you executed the script or plan:** 2
- **Which executions failed, and why (one line each):**
  - execution 1 — `crash` (`TypeError: empty(): argument 'dtype' must be
    torch.dtype, not str`): `safe_open(...).get_slice(name).get_dtype()` returns
    the safetensors dtype *name* (`"F32"`), not a `torch.dtype`, so my byte-size
    helper could not build a probe tensor from it.
- **Pitfalls or surprises you hit (one line each):**
  - The adapter prefix is `base_model.model.` + the base name (i.e.
    `base_model.model.model.layers.…`), so the mapping is a prefix strip, not a
    regex over `model.`.
  - `get_dtype()` on a slice returns a string; sizing shards from header
    metadata needs an explicit dtype→itemsize table.
  - Sharding had to be planned from the headers (shapes/dtypes) before loading
    any tensor, otherwise the whole 5.1 GB float32 state dict sits in RAM at
    once; loading per shard keeps the peak near one shard.
  - `save_file` needs contiguous, non-aliased tensors, so each tensor is
    `.contiguous().clone()`d before writing.
- **Anything in the task text or documentation that was unclear:**
  - The sharding rule and its example disagree. `model.embed_tokens.weight` and
    `lm_head.weight` are 412,090,368 bytes each, which is **below** the
    536,870,912-byte budget, so the "a single tensor larger than that is stored
    alone in its own shard" clause never fires for them. I implemented the
    stated budget rule with in-order greedy packing over sorted tensor names:
    `lm_head.weight` ends up alone in shard 1 only because the next tensor would
    overflow, and `model.embed_tokens.weight` shares shard 2 with one 64 MB MLP
    weight (479,199,232 bytes total, within budget). If the hidden reference
    really does isolate both big tensors, this shard assignment will differ,
    though every shard still respects the byte budget.
  - The task does not say whether `config.json` / tokenizer files should be
    copied into the output. I copied `config.json` and `generation_config.json`
    only; the graded key set is unaffected.
- **Tools used (condition F):** n/a (condition P: torch 2.14.0, safetensors 0.5.3).
- **Approximate time spent, if you can tell:** ~10 minutes, of which the script
  run itself is ~9 seconds.
