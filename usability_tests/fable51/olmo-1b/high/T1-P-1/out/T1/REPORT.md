# Participant self-report: T1 (OLMo-1B-0724-hf, condition P)

- Final artifact path: `out/T1/solution.py` (script) and `out/T1/model.safetensors` (checkpoint, 86 tensors, 12 blocks, float32)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Renumbering collision hazard avoided by construction: the output is a fresh dict keyed by new names built from an injective old->new map, never an in-place rename, with an explicit collision assert.
  - The task's "no tensor of blocks 12, 13, 14, 15 remains" check is stated in terms of output indices, so the script checks the output key set for any block index >= 12 rather than the removed input blocks (2, 6, 10, 14).
  - Output is written to a temp file, re-read and verified, then renamed, so a failed check leaves no `model.safetensors` behind.
- Anything in the task text or documentation that was unclear: nothing material. The "blocks 12..15" wording in the required checks reads differently from the removed blocks 2, 6, 10, 14 in the required result; I read it as post-renumbering indices, and both interpretations are covered by the checks.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 2 minutes, of which ~8 s was the script run.
