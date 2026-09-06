# T1 self-report

- **Final artifact path:** `out/T1/solution.py` (output: `out/T1/model.safetensors`)

- **Number of times you executed the script or plan:** 2

- **Which executions failed, and why (one line each):**
  1. `KeyError: Key collisions detected: h.7.* -> h.5.*` — `torch_state_bridge` applies *every* rule in sequence to *every* key, so the destination of one rename was re-matched by another rule's source (`h.10.x` -> `h.7.x` -> `h.5.x`); the collision check fired on that chained result.
  2. (succeeded)

- **Pitfalls or surprises you hit (one line each):**
  - `torch-state-bridge` rewriting is chained, not simultaneous: with overlapping source/destination index spaces a single rule set silently double-renames, so the rename had to go through a temporary namespace (`__NEW__<i>__<rest>`) that no rule can match.
  - Its `{name}` placeholders compile to `\d+` only (numeric), not a general wildcard — `h.3.{rest}, h.2.{rest}` matched nothing and returned the dict unchanged, i.e. a silent no-op rather than an error. I switched to one explicit literal rule per key.
  - Its rule regexes are unanchored `re.sub` patterns, so a "temporary" prefix like `tmp_h.5.x` would still contain the substring `h.5.x`; the temp namespace had to avoid any `h.<digit>.` substring.
  - The causal-mask buffer `h.<i>.attn.bias` is part of each block's 13 tensors and must be renamed like the rest (13*9+4 = 121). transformers 5.12 reports these as `unexpected` on load because the mask buffer is non-persistent there — harmless, and not a reason to drop them.
  - Renumbering collisions were a non-issue in the end because the surviving blocks are written into a fresh dict keyed by the new name, but the drop-then-renumber order still matters and is asserted.

- **Anything in the task text or documentation that was unclear:** Nothing. The explicit old->new index table and the 121-tensor count made the target unambiguous; it was clear that `attn.bias` counts as one of the 13 block tensors.

- **Tools used (condition F): name, version, and why:**
  - `safetensors` 0.5.3 — load the input state dict and write the output; direct, dtype- and bit-preserving.
  - `torch-state-bridge` 0.1.0 — the key rewriting itself, chosen for its explicit rule list and built-in collision detection, which is exactly the hazard this task is about. Its rules are generated from the actual key set (one literal rule per surviving tensor), so no pattern can overreach onto e.g. `mlp.c_proj`.
  - `torch` 2.14.0 — tensor comparison (`Tensor.equal`) in the post-conditions.
  - `transformers` 5.12.1 — only for an out-of-band sanity load of the result into a `n_layer=9` `GPT2Config`; not part of the solution script.
  - I did not use mergekit: its passthrough layer slicing is built around contiguous `layer_range` slices of a HF model and would have meant several slices plus a full model re-save, which is more machinery and less direct control over the exact key set than a rule-based rename.

- **Approximate time spent, if you can tell:** ~10 minutes.

## Checks enforced by `out/T1/solution.py` (all before anything is written)

Every failure calls `sys.exit(1)` and no output file is created (the file is written to a temp path and `os.replace`d only after all checks pass):

- input has blocks 0..11 and the three blocks to drop are present;
- exactly 3*13 = 39 tensors dropped;
- the derived renumbering map equals the one the task specifies;
- `state_bridge(..., detect_collision=True)` on both rename phases;
- no tensor of blocks 9, 10, 11 remains;
- exactly 9 tensors match `h.<i>.attn.c_attn.weight`, and output block indices are contiguous 0..8;
- output has exactly 121 tensors;
- every output tensor is bit-equal, same shape and same dtype as its source tensor under the inverse map;
- the 4 non-block tensors are exactly `wte.weight`, `wpe.weight`, `ln_f.weight`, `ln_f.bias`.
