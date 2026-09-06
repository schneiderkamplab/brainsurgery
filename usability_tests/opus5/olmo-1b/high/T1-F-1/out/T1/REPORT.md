# T1 participant self-report

- **Final artifact path:** `out/T1/solution.py` (output: `out/T1/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none — the single execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - `torch-state-bridge` placeholders (`{n}`) only ever compile to `(?P<n>\d+)`, so a
    "rest of the name" wildcard like `model.layers.3.{r}` silently matches nothing; I
    checked this on a toy dict before touching the real checkpoint and switched to
    plain prefix rules (`model.layers.3.` -> `model.layers.2.`).
  - `RuleEngine` applies *all* rules in sequence to each key rather than stopping at the
    first match, so rule order is load-bearing: ascending old-index order is safe
    (a key renamed by rule `k` can never match a later rule for `k+1`), descending
    order collides survivors onto each other. Verified the collision guard actually
    fires on the descending variant before relying on it.
  - Trailing dots in the prefix rules are what stops `model.layers.1.` from matching
    inside `model.layers.11.`; without them the renumbering quietly overreaches.
  - The input is sharded 113 + 1 tensors, so the shard split is uneven and the second
    shard holds only `lm_head.weight`; the loader has to follow the index, not assume
    a split.
- **Anything in the task text or documentation that was unclear:**
  - Whether `out/T1/` should also carry a `config.json` with `num_hidden_layers: 12`.
    The "Required result" says a single file `model.safetensors` and grading compares
    tensors only, so I wrote just that file; the checkpoint is nonetheless ready to load
    into a 12-layer config of the same architecture.
- **Tools used (condition F):**
  - `safetensors` 0.5.3 — load the two input shards and write the single output file.
    Chosen for tensor I/O because it round-trips values bit-exactly with no dtype or
    layout handling in the way, which is what grading checks.
  - `torch-state-bridge` 0.1.0 — the block renumbering, expressed as a declarative
    ordered rule list, with `state_bridge_preview` for the mapping and
    `detect_collision=True` for the exact hazard the task calls out (a shifted block
    overwriting a surviving one). This is the part of the job the package exists for.
  - `torch` 2.14.0 — only as the tensor type behind safetensors, plus `Tensor.equal`
    in the post-write read-back check.
  - Considered and rejected: `mergekit` passthrough layer slicing. It expresses the
    layer selection well, but it goes through `transformers` model construction and
    writes a sharded HF directory with its own dtype and lm_head-tying decisions —
    more machinery between the input bits and the output bits than a task graded on
    bit-exactness wants, and it does not produce the single required file directly.
- **Where the required checks live:** all in `out/T1/solution.py`, all on the in-memory
  result *before* any write, each calling `fail()` (stderr + exit 1, no output file):
  no tensor of blocks >= 12 remains; exactly 12 `self_attn.q_proj.weight` matches and
  indices contiguous 0..11; exactly 86 output tensors. Plus, beyond what was required:
  input tensor count, the drop count, an independent reconstruction of the expected key
  set from the spec's mapping compared against the rule engine's output, per-tensor
  identity/shape/dtype preservation, the two non-block tensors carried through
  untouched, and a read-back of the written file (unlinked if it disagrees).
- **Approximate time spent:** ~10 minutes, most of it reading the `torch-state-bridge`
  source to pin down its placeholder and rule-ordering semantics before running anything.
