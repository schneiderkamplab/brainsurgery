# T1 self-report (condition F)

- **Final artifact path:** `out/T1/model.safetensors` (121 tensors).
  Authored files: `out/T1/solution.py`, `out/T1/rename_rules.txt`
  (torch-state-bridge rule set), `out/T1/run.sh` (runner).

- **Number of times you executed the script or plan:** 2

- **Which executions failed, and why (one line each):**
  1. `failed_assertion` — my own input sanity check `sorted(DROP) + sorted(KEEP) == sorted(src_blocks)`
     compared a concatenation of two separately-sorted lists against a sorted list, so it
     fired on valid input; fixed to `sorted([*DROP, *KEEP]) == src_blocks`. Nothing was written.
  2. Succeeded.

- **Pitfalls or surprises you hit (one line each):**
  - Renumbering collisions are a real hazard with `torch-state-bridge`: its `RuleEngine`
    applies *every* rule in order to *every* key, so a rule set written in descending source
    order cascades (`h.9.`→`h.6.`→`h.4.`→`h.3.`→`h.2.`). I verified this: the descending
    variant produces 26 colliding keys. Listing rules in ascending source order makes every
    rule's target smaller than any later rule's source, so no key is rewritten twice.
  - The checkpoint keys are flat (`h.<i>.…`), with no `transformer.` prefix, so an
    unanchored substring rule is fine here, but `h.1.` vs `h.11.` needed a moment's thought
    (the library escapes literal dots, so `h\.1\.` cannot match inside `h.11.`).
  - The causal-mask buffer `h.<i>.attn.bias` is part of the 13 tensors per block and must be
    carried along, but `transformers` 5.12 no longer registers it: a 9-layer `GPT2Model`
    state dict has 112 keys, not 121. My load check therefore requires the 112 model keys to
    be present and allows exactly the 9 `attn.bias` buffers as extras.
  - I did not use mergekit. Its passthrough layer-slicing route goes through
    `transformers`/`save_pretrained`, which would rename keys to `transformer.h.*` and drop
    the `attn.bias` buffers — i.e. it cannot produce the exact 121-key set this task grades.

- **Anything in the task text or documentation that was unclear:** nothing material. The task
  spells out the old→new index mapping and the tensor inventory, which removed all ambiguity.
  Minor: "a single file `out/T1/model.safetensors`" sits next to the rule that authored files
  go under `out/<task>/`; I read it as "one checkpoint file" and put the script, rules and this
  report alongside it.

- **Tools used (condition F):**
  - `torch-state-bridge` 0.1.0 — the actual renumbering, via a declarative rule file
    (`rename_rules.txt`). Chosen because it is the one allowed package aimed exactly at
    key rewriting, and its `state_bridge_preview` / `detect_collision=True` give a
    purpose-built guard against the collision hazard the task calls out.
  - `safetensors` 0.5.3 — `load_file` / `save_file`; the only I/O needed, and it preserves
    dtypes bit-exactly.
  - `torch` 2.14.0 — `torch.equal` for the bit-exactness checks.
  - `transformers` 5.12.1 — `AutoConfig` from `inputs/base` plus a meta-device `GPT2Model`
    with `n_layer=9`, to prove the result really loads into a 9-layer configuration
    (structural check only, no weights materialised).
  - Deletion of blocks 2/5/8 and all assertions are plain Python: `torch-state-bridge`
    rewrites keys but cannot drop them, and none of the allowed tools enforce task-level
    invariants.

- **Checks enforced by the run (all fatal, output published only after they pass):** input is
  160 tensors / blocks 0..11; dropping 2/5/8 removes exactly 39 tensors; the rename mapping is
  previewed and each key's rename is required to change *only* the block index; `state_bridge`
  runs with `detect_collision=True`; no tensor of blocks 9/10/11 remains; exactly 9 keys match
  `h.<i>.attn.c_attn.weight` and the indices are contiguous 0..8 with 13 tensors each; the
  output has exactly 121 tensors; every output tensor is bit-exact, same shape and same dtype
  as its source key; the 4 non-block tensors are exactly `wte/wpe/ln_f.{weight,bias}`; a
  9-layer `GPT2Model` would have no missing keys. The file is written to a `.tmp` path, read
  back and re-verified, and only then `os.replace`d into place, so a failure leaves no output.

- **Approximate time spent:** ~5 minutes.
