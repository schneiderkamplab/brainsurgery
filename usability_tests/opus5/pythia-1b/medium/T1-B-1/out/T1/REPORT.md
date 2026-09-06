# T1 self-report (Condition B: BrainSurgery plan)

- **Final artifact path:** `out/T1/plan.yaml` (output at `out/T1/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none; the single execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - The renumbering collision hazard is real: `move` refuses an existing destination, so the
    move order matters. Deleting blocks 2/6/10/14 first and then moving in ascending
    destination order (3->2, 4->3, 5->4, 7->5, ...) makes every target slot free at the time
    it is written; any descending or arbitrary order would have aborted on `dest_exists`.
  - Dots must be escaped in the `from`/`target` regexes (`gpt_neox\.layers\.3\.`), otherwise
    `.` matches any character; the escaped-and-anchored form also keeps `layers.1.` from
    matching `layers.11.`, `layers.12.` etc., which was the other overmatch trap here.
  - The `to` side of `move` is a rewrite template, not a regex, so it uses plain dots plus a
    `\1` backreference for the captured tail; this asymmetry is documented but easy to miss.
  - `count` with a negative lookahead (`(?!gpt_neox\.layers\.).*`) worked for asserting that
    exactly the 4 non-block tensors survive, since references are full-match regexes.
- **Anything in the task text or documentation that was unclear:**
  - The task text is precise; nothing blocked me. Minor: the README describes `output` behavior
    for directory paths and sharding, but does not spell out that a `.safetensors` file path
    produces a single unsharded file — I relied on the "file or directory" wording and it
    behaved as expected (one 184-tensor file).
  - `help` lists the assert operators with required keys but gives no worked example for
    `count` with a pattern; I inferred it from the `equal` examples.
- **Tools used (condition F):** n/a (condition B: only the `brainsurgery` CLI and its plan format).
- **Approximate time spent, if you can tell:** roughly 5 minutes, most of it reading
  `docpack/README.md` and the `move`/`delete`/`assert.count` entries in `docpack/help.txt`.
