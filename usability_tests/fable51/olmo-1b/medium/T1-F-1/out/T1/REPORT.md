# T1 report (condition F, OLMo-1B-0724-hf)

- Final artifact path: `out/T1/solution.py` (produces `out/T1/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - None during the run. The known hazard (renumbering collisions when shifting blocks in place) was avoided by design: the script builds a fresh output dict from an explicit old->new block map and fails on any duplicate destination key, so no in-place rename order matters.
- Anything in the task text or documentation that was unclear:
  - The "Required checks" list says "no tensor of blocks 12, 13, 14, 15 remains", which is phrased in terms of new indices (>= 12), while the removal list is in old indices (2, 6, 10, 14). Both readings are enforced: the script counts exactly 28 dropped tensors from the old blocks and asserts no output index >= 12.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: shard loading and single-file save; the only I/O needed.
  - `torch` 2.14.0: tensor equality checks (`torch.equal`) for value fidelity.
  - Plain Python `re`/`json`: key parsing and reading the shard index.
  - Not used: mergekit (layer slicing would work but pulls in model loading and produces a sharded HF directory rather than a single file, and its output naming/dtype handling would need extra verification) and torch-state-bridge (regex rewriting is close to what is needed, but a 20-line explicit map with collision and count checks was simpler and easier to prove correct).
- Approximate time spent, if you can tell: about 2 minutes, one script write and one run.

## Checks enforced by the script (non-zero exit, nothing written, on failure)

- Output file must not pre-exist; input must have 114 tensors with no duplicate keys across shards.
- Exactly 28 tensors dropped (4 blocks x 7 tensors); destination-key collision check on every insert.
- No output tensor with block index >= 12; exactly 12 `self_attn.q_proj.weight` tensors; block indices exactly 0..11; every block has 7 tensors; exactly 86 tensors in total.
- Every surviving block tensor and both non-block tensors compared bit-exactly (shape, dtype, values) against the source under the mapping.
- Written to a temp file, re-opened and its key set re-verified, then atomically renamed into place.
