# T1 self-report (Pythia-1B, condition F)

- Final artifact path: `out/T1/solution.py` (output: `out/T1/model.safetensors`, 184 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - None during the run. The collision hazard was avoided by building a full `old -> new` map up front and writing into a fresh dict (no in-place renames), with an explicit collision check on every insert.
  - The three non-parameter buffers per block (`attention.bias`, `masked_bias`, `rotary_emb.inv_freq`) are matched by the same `gpt_neox.layers.<i>.` prefix, so they move with their block automatically; no special handling needed.
- Anything in the task text or documentation that was unclear:
  - The "Required checks" say "no tensor of blocks 12, 13, 14, 15 remains", which is about the *output* index space (old block 15 becomes 11), not the removed blocks 2/6/10/14. The script checks both: no output index >= 12, indices exactly 0..11, and the removed blocks are skipped by construction.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: load/save of the single checkpoint file; bit-exact, no framework model needed.
  - `torch` 2.14.0: tensor equality checks (`torch.equal`) for the value-level verification.
  - `transformers` 5.12.1: only for a post-hoc sanity check outside the script (a 12-layer `GPTNeoXForCausalLM` loads the output with no missing/unexpected parameter keys).
  - I did not use mergekit or torch-state-bridge: a 16-to-12 slice with a non-contiguous drop set (2, 6, 10, 14) needs 5 passthrough slices in mergekit and produces a sharded HF directory rather than a single file, and the rename itself is a 12-entry map that is easier to make collision-proof and verifiable in ~60 lines of plain Python than through a rule engine.
- Approximate time spent, if you can tell: about 2 minutes (one script write, one run, one external verification).

## Checks enforced by `solution.py` (non-zero exit, no output written on failure)

- destination must not already exist;
- input has 244 tensors;
- no rename collision while building the output dict;
- no output block index >= 12; block indices are exactly 0..11;
- exactly 12 `gpt_neox.layers.<i>.attention.query_key_value.weight` tensors;
- exactly 184 output tensors;
- spot bit-exact comparison of each kept block against its source block;
- write to a temp file then `os.replace`, then re-open and compare the key set (removes the file on mismatch).
