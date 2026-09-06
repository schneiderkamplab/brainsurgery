# T4 — Participant self-report (condition P)

- **Final artifact path:** `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none — the single execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - The ordering hazard is only a hazard if you merge in place; writing `out[X]`
    into a fresh dict and reading `base[X]`, `ft1[X]`, `ft2[X]` from the
    read-only source handles makes both task vectors trivially relative to the
    unmodified base, so I never mutated `base`.
  - Bit-exact comparison of the 180 shared tensors needs to be NaN-safe:
    `torch.equal` on float16 would call two NaNs unequal, so I compared the raw
    `uint8` views of the storage instead.
  - `inputs/` also contains a `lora/` directory that this task does not use;
    I keyed off the three checkpoints named in TASK.md only.
  - I constructed the 64 MLP names explicitly (layers 0..15 × 4 tensors) rather
    than regex-matching `mlp`, to avoid any chance of over- or under-matching,
    and asserted all 64 exist in the checkpoints.
  - Memory: streaming per-key via `safe_open.get_tensor` keeps only the ~2 GB
    output dict resident instead of three full 2 GB state dicts.
- **Anything in the task text or documentation that was unclear:**
  - "identical" in step 1 was not qualified — I read it as bit-exact and also
    checked shape and dtype agreement, which is strictly stronger than a
    tolerance-based reading.
  - Whether to write safetensors metadata was unspecified; I wrote the
    conventional `{"format": "pt"}` header, which grading should ignore.
  - Only `model.safetensors` was requested, so I did not copy `config.json` or
    the tokenizer files into `out/T4/`.
- **Tools used (condition F):** n/a — condition P (torch 2.14.0, safetensors 0.5.3).
- **Approximate time spent, if you can tell:** ~5 minutes total; the script runs in ~9 s wall clock.
