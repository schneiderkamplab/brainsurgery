# T1 participant self-report (condition F, Pythia-1B)

- **Final artifact path:** `out/T1/solution.py` (run via `out/T1/run.sh`);
  output at `out/T1/model.safetensors`.

- **Number of times you executed the script or plan:** 1 execution that was
  meant to produce the output, and it succeeded. (Three further executions
  were deliberate negative tests on copies in `/tmp`, run *after* the output
  existed, to prove the guards fail loudly; they were not attempts at the
  output.)

- **Which executions failed, and why:** none.

- **Pitfalls or surprises you hit:**
  - The renumbering cascade: `torch_state_bridge.RuleEngine` applies *every*
    rule in sequence to *every* key, so a key already rewritten to
    `layers.2.` can be re-matched by a later rule. Emitting the rules in
    ascending source order makes every destination index strictly lower than
    every later source index, which avoids it; I verified the descending
    order does collide (`state_bridge` raised `KeyError: Key collisions
    detected: layers.4.w->layers.2.w, layers.5.w->layers.2.w, ...`).
  - Rules must carry the trailing dot (`gpt_neox.layers.1.`), otherwise
    `layers.1` also matches `layers.11`/`layers.15`. torch-state-bridge
    `re.escape`s the literal parts, so the unescaped-dot hazard does not
    apply here.
  - I did not route this through `mergekit` passthrough slicing or
    `transformers.save_pretrained`, although F-allowed.md suggests mergekit
    for T1: both go through a `transformers` model, and transformers 5.12
    no longer registers the three per-block buffers
    (`attention.bias`, `attention.masked_bias`,
    `attention.rotary_emb.inv_freq`). A model round-trip would have dropped
    36 of the required 184 tensors. Grading is on the exact key set, so the
    "recommended" tool route would have failed this task. Confirmed
    empirically: loading the output into a 12-layer config gives 0 missing
    keys and 36 *unexpected* keys, i.e. those buffers exist only in the
    checkpoint, not in the model.
  - `safetensors` refuses shared/non-contiguous storage on save; I call
    `.contiguous()` on write, which was not actually needed here.

- **Anything in the task text or documentation that was unclear:** nothing
  material. The task is fully specified, including the explicit old->new
  index table, so I did not have to infer the drop policy. One minor point:
  the task says the result "must load into a 12-layer configuration of the
  same architecture", which is true only in the `strict=False` sense given
  the three non-parameter buffers per block that current transformers does
  not register; the required 184-tensor key set and a strict load are not
  simultaneously satisfiable with transformers 5.12.

- **Tools used (condition F):**
  - `safetensors` 0.5.3 — load/save. Chosen because grading is bit-exact on
    values, shapes and dtypes; it moves tensors without touching any of them.
  - `torch-state-bridge` 0.1.0 — the block renumbering, via `parse_rules`
    literal rules, `state_bridge_preview` and `state_bridge(...,
    detect_collision=True)`. Chosen because its collision detection is
    exactly the hazard this task is about, so the rename has a guard that is
    independent of my own reasoning about rule order.
  - `torch` 2.14.0 — tensor comparison in the checks.
  - `transformers` 5.12.1 — only for the post-hoc confirmation that the
    result loads into a 12-layer GPTNeoX config; not used to produce the
    output, for the buffer reason above.
  - `mergekit` 0.1.4 — considered and rejected, see pitfalls.

- **Approximate time spent:** about 5 minutes: ~2 minutes reading the input
  key layout and the torch-state-bridge rule engine (to find the sequential
  rule-application behaviour), ~2 minutes writing the script, ~1 minute
  running it and the negative tests.

## What the run enforces

All required checks run *before* anything is written, and the output is
written to a `.tmp` file that is only renamed into place after it is read
back and compared tensor by tensor:

- input has exactly blocks 0..15 (guards against a surprising input);
- exactly 60 tensors dropped (4 blocks x 15);
- no renumbering collision (`state_bridge_preview` + `detect_collision`);
- the resulting key set equals the mapping the task specifies, key for key;
- no tensor of block index 12..15 remains;
- exactly 12 `gpt_neox.layers.<i>.attention.query_key_value.weight` tensors;
- surviving block indices are exactly 0..11, each with 15 tensors;
- the 4 non-block tensors are present and pass through by identity;
- output has exactly 184 tensors;
- every tensor's shape and dtype matches its source tensor;
- the file on disk re-reads to 184 tensors, identical keys and bit-identical
  values, or it is deleted and the run exits non-zero.

Verified negatively: with `DROP = [2, 6, 10]` the run exits 1 and writes no
output file.
