# T1 participant self-report (GPT-2 124M, condition F)

- Final artifact path: `out/T1/solution.py` (output: `out/T1/model.safetensors`, 121 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Renumbering collision hazard was avoided by construction: the old->new map is computed from the sorted survivor list and every tensor is written into a fresh dict under its final name, with an explicit collision check, so shift order never matters.
  - The regex is anchored (`^h\.(\d+)\.(.+)$`) so `h.1.` cannot match `h.10.`/`h.11.`; the causal-mask buffer `attn.bias` is kept as part of each block (13 tensors), as the task requires.
  - transformers 5.x no longer registers `attn.bias` as a state-dict buffer, so a strict `load_state_dict` reports them as unexpected keys; this is expected and does not affect the required artifact.
- Anything in the task text or documentation that was unclear: the "Required checks" say no tensor of blocks 9, 10, 11 may remain, which for a renumbered checkpoint means indices >= 9; I enforced that reading (all block indices must be < 9) plus contiguity 0..8.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3 (`safetensors.torch.load_file` / `save_file`): direct, lossless load and save of the single-file checkpoint with `{"format": "pt"}` metadata preserved.
  - `torch` 2.14.0: tensor equality checks (`Tensor.equal`) and `.contiguous()` before saving.
  - Python `re` for anchored block-name parsing.
  - Not used: mergekit passthrough (would go through HF model loading, rewrite config and possibly drop the `attn.bias` buffers or change file layout; bit-exact key set is easier to guarantee with a script); torch-state-bridge (unfamiliar API, no persistence, and the plain script already gives collision detection and previews via explicit checks).
  - Verification after the run (not part of the artifact): rebuilt the expected key/value map independently and compared bit-exactly; loaded the result into a `GPT2LMHeadModel` with `n_layer=9` via transformers 5.12.1 (no missing keys).
- Approximate time spent, if you can tell: about 3 minutes.
