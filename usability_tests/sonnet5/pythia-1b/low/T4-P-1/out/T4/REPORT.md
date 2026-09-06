# Participant self-report

- **Task**: T4 — task-vector merge of two Pythia-1B fine-tunes
- **Condition**: P (Python / PyTorch baseline)
- **Approach**: Load `base`, `ft1`, `ft2` with `safetensors.torch.load_file`.
  Verify all three share the same tensor name set, and that every tensor
  outside the 64 known MLP tensor names is shape/dtype/value-identical
  across all three checkpoints (bit-exact via `torch.equal`), aborting with
  a loud error otherwise. For each of the 64 MLP tensors, compute both task
  vectors (`ft1 - base` and `ft2 - base`) independently in float32 against
  the unmodified base, scale each by `lambda = 0.4`, add both to the base,
  then cast back to float16. Copy all other tensors from base unchanged.
  Assert exactly 64 tensors were merged and the output has exactly 244
  tensors before writing `out/T4/model.safetensors`.
- **Executions**: 1 (script ran and wrote output on the first run)
- **Failed executions**: none
- **Tools/packages used**: `torch`, `safetensors.torch` (load_file/save_file)
  only, as specified in `requirements-P.txt`. No `brainsurgery` package or CLI.
- **Pitfalls / things to note**: none encountered — the tensor name pattern
  for the MLP layers was given explicitly in TASK.md, so no discovery was
  needed there. The main care point was computing each fine-tune's delta
  against the unmodified `base` tensor (not against a partially-merged
  result), which the code does by building both deltas before combining.
- **Notes on grading tolerance**: merged tensors are computed in float32 and
  cast to float16 at the end, matching the spec's stated tolerance
  (bit-exact for the 180 unchanged tensors, ≤1e-3 relative Frobenius error
  for the 64 merged tensors).
