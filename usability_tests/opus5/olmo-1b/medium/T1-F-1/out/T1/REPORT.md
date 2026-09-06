# T1 self-report (condition F, OLMo-1B-0724-hf)

- **Final artifact path:** `out/T1/solution.py` (entry point `out/T1/run.sh`); output `out/T1/model.safetensors`
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none — the first execution produced the output.
- **Pitfalls or surprises you hit:**
  - The renumbering collision hazard is real but it is *not* caught by
    `torch-state-bridge`'s `detect_collision`: rules are applied sequentially,
    each to the output of the previous, so a naive rule list
    (`layers.3.`→`layers.2.`, `layers.4.`→`layers.3.`, …) cascades within a
    single key and silently produces no collision. I verified this on a toy
    state dict. I therefore rewrote in two passes through a marker namespace
    (`model.layers.NEW<i>.`), which makes both cascade and collision
    impossible; collision detection stays on as a belt-and-braces check.
  - Rule sources must carry the trailing dot, otherwise `model.layers.1.`
    would also match `model.layers.11.`.
  - Input is sharded, so a shard-overlap check and an index/key-set
    cross-check are needed before counting anything.
  - `safetensors` refuses shared/non-contiguous storage, so the output tensors
    are saved `.contiguous()`; writing goes to a temp file and is `os.replace`d
    only after every check passes, so a failed run leaves no output.
- **Anything unclear in the task text or documentation:** nothing material. The
  task says "a single file `out/T1/model.safetensors`", so I did not emit an
  index, a 12-layer `config.json`, or tokenizer copies; grading is on the
  tensors only.
- **Tools used (condition F):**
  - `torch-state-bridge` 0.1.0 — rule-based key rewriting, the tool in the
    allowed list aimed at exactly this bulk-rename job; used for the
    renumbering, plus its collision detection.
  - `safetensors` 0.5.3 — reading the two input shards and writing the output.
  - `torch` 2.14.0 — tensor identity/shape/dtype checks.
  - `numpy`/`transformers`/`peft`/`mergekit` not used. I considered mergekit
    `passthrough` layer slicing, but this task drops interior blocks
    (2, 6, 10, 14), which needs several slices stitched together, and mergekit
    writes a whole model directory with a rewritten config rather than the
    single file the task asks for — more moving parts for no benefit.
- **Approximate time spent:** ~10 minutes.

## Checks enforced by the run (all fail loudly, before anything is written)

Input side: 114 tensors, key set equals `model.safetensors.index.json`, no key
in two shards, blocks exactly 0..15, 7 tensors per block, exactly 2 non-block
tensors. Output side: no tensor of blocks ≥ 12 (covers 12, 13, 14, 15); block
indices exactly 0..11; exactly 12 tensors matching
`model.layers.<i>.self_attn.q_proj.weight`; exactly 86 tensors; every output
tensor is the identical object from its expected source with unchanged shape
and dtype; both non-block tensors unchanged.

Independently re-verified after the run: the written file reloads to 86 keys
that match the expected mapping exactly, bit-exact values, all float32.
