# T4 participant self-report (condition P)

- **Final artifact path:** `out/T4/solution.py` (output: `out/T4/model.safetensors`,
  5119161112 bytes, 114 tensors)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none; the single
  execution passed every check.
- **Pitfalls or surprises you hit (one line each):**
  - The base is sharded (two files plus an index) while the fine-tunes are
    single files, so the reader had to resolve names through
    `model.safetensors.index.json` rather than opening one file per checkpoint.
  - The ordering hazard the task warns about: I formed both task vectors from
    the same freshly loaded base tensor in one expression, so the first merge
    can never contaminate the second — no in-place `+=` on a base state dict.
  - "Identical" for the 66 shared tensors is graded bit-exactly, so I compared
    raw bytes (`.numpy().view(np.uint8)`) instead of `==` / `torch.equal`,
    which would treat `-0.0` and `0.0` as equal and any `NaN` as unequal.
  - Three float32 copies of a 5 GB checkpoint do not need to be resident at
    once; `safe_open` lets each tensor be pulled and dropped one at a time,
    keeping peak memory near the ~5 GB output dict.
  - The MLP set was derived by regex over the actual key set and then checked
    against the constructed gate/up/down × layer set, so a stray match (or a
    missed layer) fails loudly rather than silently changing the merged count.
- **Anything in the task text or documentation that was unclear:**
  - Step 1 says "the same tensor names"; it does not explicitly require equal
    shapes/dtypes across the three checkpoints, so I checked those too as part
    of the abort condition.
  - The task does not say whether the output should be accompanied by the
    config/tokenizer files. I wrote only `model.safetensors`, since the
    "Required result" names exactly that single file.
- **Tools used (condition F):** n/a — condition P; standard library plus
  `torch` 2.14.0, `safetensors` 0.5.3, `numpy` 2.5.2.
- **Approximate time spent, if you can tell:** ~10 minutes, of which the script
  run itself was 17 s wall clock.
