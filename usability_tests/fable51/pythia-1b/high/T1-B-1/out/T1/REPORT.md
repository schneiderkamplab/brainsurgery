# Participant self-report: T1 (Pythia-1B, condition B)

- Final artifact path: `out/T1/plan.yaml` (output checkpoint `out/T1/model.safetensors`, 184 tensors)
- Number of times you executed the script or plan: 1 execution of `out/T1/plan.yaml`
  (succeeded on the first run and wrote the output). Separately, I ran a
  read-only verification plan `out/T1/verify.yaml` once (two inputs, no `output`,
  `assert equal` of every output tensor against its mapped source tensor) and
  one throwaway negative-control plan to confirm the equality assert actually
  fails on a mismatched block. Neither of those writes anything.
- Which executions failed, and why (one line each): none of the solution plan
  executions failed. The negative-control plan failed by design
  (`equal failed: new::gpt_neox.layers.2.* != base::gpt_neox.layers.2.*`).
- Pitfalls or surprises you hit (one line each):
  - Renumbering collision hazard: solved by moving one surviving block per
    `move` transform in ascending original order; each destination index is
    strictly lower and already vacated, and `move` refuses existing
    destinations, so a wrong order would fail loudly instead of overwriting.
  - Regex references are full-match, so `gpt_neox\.layers\.(2|6|10|14)\..*`
    is needed (a bare `layers.1` prefix would not match, but an unanchored
    `1\.` style pattern could catch 10..15 with other tools; the full-match
    semantics avoid this).
  - `move` with a regex `from` and a `\1` back-reference in `to` works for
    whole blocks (15 tensors per move); the docs describe this for `copy`/`equal`
    and I inferred it for `move` from "like 'to' in copy/move".
- Anything in the task text or documentation that was unclear:
  - The README's `move` entry says only "destination must not exist"; it
    would help to state explicitly that regex capture groups (`\1`) are
    supported in `to`, as the `equal` docs do.
  - The task's required check "no tensor of blocks 12, 13, 14, 15 remains"
    reads oddly since old blocks 12, 13 and 15 survive (renumbered to 9, 10, 11);
    I interpreted it as "no name with index 12..15 remains after renumbering".
- Tools used (condition F): n/a (condition B, `brainsurgery` CLI only)
- Approximate time spent, if you can tell: about 3 minutes
