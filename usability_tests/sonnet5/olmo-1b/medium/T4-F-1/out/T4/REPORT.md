## Participant self-report

- Final artifact path: `out/T4/solution.py` (invoked via `out/T4/run.sh`), producing `out/T4/model.safetensors`.
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The base checkpoint is sharded (`model.safetensors.index.json` + two shard files) while `ft1`/`ft2` are single-file, so loading had to branch on that rather than assume one layout.
  - `safe_open` tensors from disk are already contiguous, but tensors reconstructed via `float32` arithmetic (`b + lambda*(f1-b) + lambda*(f2-b)`) need an explicit `.contiguous()` before `save_file`, otherwise safetensors can reject non-contiguous storage.
- Anything in the task text or documentation that was unclear: no, the spec (verify-then-merge-then-copy-rest, 48 MLP names, formula, output shape) was unambiguous.
- Tools used (condition F): name, version, and why:
  - `torch` 2.14.0 — tensor arithmetic in float32 and dtype casting for the merge.
  - `safetensors` 0.5.3 (`safe_open`, `save_file`) — reading the sharded base and single-file fine-tunes, and writing a single 114-tensor output file.
  - No merge toolkit (`mergekit`) was used: the task is a single two-input task-vector add over a fixed, known tensor set, which a ~150-line script expresses more directly and auditably (explicit key-set/bit-identity checks, explicit tensor-count assertions) than authoring and debugging a mergekit YAML config for the same one-off computation. `peft`/`transformers` weren't needed since there's no LoRA adapter or head-pruning here, and `torch-state-bridge` wasn't needed since no key renaming happens.
- Approximate time spent, if you can tell: ~10 minutes (single write-and-run pass, no debugging required).
