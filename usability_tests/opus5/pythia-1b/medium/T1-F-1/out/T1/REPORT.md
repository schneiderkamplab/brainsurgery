# T1 participant self-report (condition F)

- **Final artifact path:** `out/T1/solution.py` (driver: `out/T1/run.sh`); output `out/T1/model.safetensors`
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none — the first execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - Renumbering collision hazard: solved by emitting rename rules in increasing old-index order (every target < its source, so no rule re-fires on another rule's output) and keeping `detect_collision=True`.
  - `torch-state-bridge` rules are unanchored substring matches with `{n}` = `\d+`, so a naive `layers.{n}.` rule would chain; I used explicit literal per-block rules with the trailing dot to bound the index (`layers.3.` cannot match inside `layers.13.`).
  - Each block owns 3 non-parameter buffers (`attention.bias` uint8, `attention.masked_bias`, `rotary_emb.inv_freq`); counting 15 tensors per block, not 12, matters for the 184 total.
  - Wrote to `model.safetensors.tmp` and `os.replace`d it only after re-reading and comparing the written file, so a failure can never leave partial output.
- **Anything in the task text or documentation that was unclear:** nothing material; the explicit old→new index table removed all ambiguity. The task does not say whether the 12-layer `config.json` should also be emitted — I read it as "single file", and wrote only `model.safetensors`.
- **Tools used (condition F):**
  - `safetensors` 0.5.3 — load/save the checkpoint; the only format involved, so no conversion round-trip.
  - `torch-state-bridge` 0.1.0 (`state_bridge`, `detect_collision=True`) — rule-based block renumbering; it is the purpose-built route for this rename and gives collision detection for free, which is exactly the task's stated hazard.
  - `torch` 2.14.0 — `torch.equal` for the bit-exactness checks.
  - Considered and rejected: `mergekit` passthrough layer slicing. It expresses "keep these layer ranges" and renumbers, but it goes through `transformers` model construction and re-serialization, which risks dtype/key/sharding drift (and rewrites buffers such as `attention.bias`); for a pure key-space edit that must be bit-exact, direct safetensors I/O is both simpler and safer to verify.
- **Checks enforced by the run (non-zero exit, nothing written, on failure):** no tensor of blocks 12..15 remains; block indices are exactly 0..11; exactly 12 `attention.query_key_value.weight` tensors; exactly 15 tensors per block; exactly 60 tensors dropped; exactly 184 output tensors; every output tensor bit-identical (value, shape, dtype) to its source under the mapping; post-write re-read round-trip check before the atomic move.
- **Approximate time spent:** ~10 minutes, of which ~5 s of compute.
