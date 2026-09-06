# T1 report (condition F)

- Final artifact path: `out/T1/solution.py` (invoked via `out/T1/run.sh`), output at `out/T1/model.safetensors`.
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The renumbering map is not a simple "subtract number of prior drops" shift once you also have to guard against writing over a not-yet-renamed survivor; building the full old->new dict first and writing into a fresh dict (never mutating in place) avoids the collision hazard entirely, so this was a non-issue in practice but worth calling out as the trap the task is designed around.
  - `attention.bias`, `attention.masked_bias` and `attention.rotary_emb.inv_freq` are per-block buffers, not global ones, so they had to be renumbered along with the parameter tensors rather than treated as the 4 "non-block" tensors.
- Anything in the task text or documentation that was unclear: no, the explicit old->new index list in the task text removed any ambiguity about traversal order.
- Tools used (condition F): `safetensors` 0.5.3 (`safetensors.torch.load_file`/`save_file`) and `torch` 2.14.0 for tensor equality checks, used directly rather than through mergekit or torch-state-bridge. The rename is a single fully-specified index map with no merging/arithmetic and no need for regex-capture rewriting, so a ~90-line plain script over `safetensors` was the smallest correct tool; it also let the required checks (forbidden indices, exact tensor counts, collision detection, non-block tensors byte-identical) be asserted directly in the same script that produces the output, so a bad run fails loudly before anything is written.
- Approximate time spent, if you can tell: a few minutes of scripting plus verification.
