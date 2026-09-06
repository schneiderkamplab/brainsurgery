# Participant self-report

- Final artifact path: `out/T2/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The fused `c_attn` tensor packs q, k and v as three separate 768-wide
    column segments, each independently divided into 64-wide head blocks, so
    the head-5 columns to drop are at a different offset in each segment
    (320-383, 1088-1151, 1856-1919) rather than one contiguous run.
  - `c_proj.weight` prunes by *rows* (input side, since GPT-2's Conv1D layout
    is `[in, out]`), while `c_attn.weight` prunes by *columns* (output side) —
    easy to swap by reflex if you're used to `nn.Linear`'s `[out, in]`.
- Anything in the task text or documentation that was unclear: none; the
  explicit kept-column/row ranges in "Required result" made it easy to
  cross-check the script's output independently of its own slicing logic.
- Tools used (condition F): n/a (condition P).
- Approximate time spent, if you can tell: ~10 minutes.
