# T5 — Participant self-report

- Final artifact path: `out/T5/solution.py` (invoked via `out/T5/run.sh`), output
  written to `out/T5/` (5 shards + `model.safetensors.index.json`).
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none — the single
  execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - None of real consequence; the task text already spells out the Conv1D
    `[in, out]` vs `nn.Linear` `[out, in]` layout and the required transpose,
    so no discovery was needed there.
  - Had to bin-pack shards manually to respect the "one tensor over budget
    gets its own shard" rule, since a naive fixed-tensor-count-per-shard
    split would not guarantee the 100 MiB bound.
- Anything in the task text or documentation that was unclear: no.
- Tools used (condition F): name, version, and why:
  - `torch` 2.14.0 — float32 tensor ops (`B @ A`, transpose, add).
  - `safetensors` 0.5.3 — `load_file`/`save_file` for reading the base and
    adapter checkpoints and writing shards.
  I wrote a plain script directly on the checkpoint tensors instead of
  `peft.merge_and_unload`, since that API expects a live `nn.Module` (it would
  require instantiating the whole GPT-2 model just to do a matmul + add that
  TASK.md already specifies precisely per-tensor); a direct script is smaller,
  has no hidden state, and is easy to add the required pre-write checks to.
  `mergekit` and `transformers` were not needed for the same reason.
- Approximate time spent, if you can tell: a few minutes (single pass, no
  retries).
