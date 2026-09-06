# Participant self-report

- Final artifact path: `out/T2/plan.yaml` (executed to produce `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single run succeeded.
- Pitfalls or surprises you hit (one line each):
  - `concat`'s `from` list does not support regex-capture broadcasting across layers
    (unlike `copy`/`move`/`assign`), so each of the 12 layers needed its own three
    `concat` blocks rather than one broadcasted transform.
  - `concat`/`move` destinations must not already exist, so each pruned tensor was
    built under a temporary `_pruned` name, the original was `delete`d, then the
    temporary was `move`d back onto the original name to keep tensor names unchanged.
  - Had to keep straight that GPT-2's `c_attn.weight` packs `[q | k | v]` as three
    768-wide column segments, each of which is itself split into 12 head-sized
    64-column blocks; removing head 5 means dropping columns 320:384 inside each
    of the three segments, not one single contiguous span.
- Anything in the task text or documentation that was unclear: none; the required
  column/row ranges for the kept segments were given explicitly in TASK.md, which
  made verifying the plan's slice bounds straightforward.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: a few minutes to read the layout,
  author the plan (with a short script to generate the repetitive per-layer
  blocks), and verify shapes/values against the original checkpoint before
  finishing.
