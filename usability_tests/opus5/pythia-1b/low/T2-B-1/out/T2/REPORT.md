# T2 self-report (condition B)

- Final artifact path: `out/T2/plan.yaml` (output `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1 execution of the final plan
  (plus one small throwaway probe plan under `tmp/` to confirm ref/slice syntax,
  and two read-only verification plans on the output).
- Which executions failed, and why (one line each): none; the plan passed on its
  first execution.
- Pitfalls or surprises you hit (one line each):
  - `concat` requires each source to resolve to exactly one tensor and a single
    destination, so it cannot be pattern-batched; the 48 rebuild steps had to be
    written out per layer while the slicing copies could use one regex each.
  - Destinations must not already exist, so the originals had to be deleted
    before the pruned tensors could be concatenated back under the same names.
  - Regex refs are full-match, so dots need escaping (`layers\.(\d+)\.`) to avoid
    matching neighbouring names.
- Anything in the task text or documentation that was unclear: nothing blocking;
  the interleaved GPT-NeoX qkv layout was spelled out explicitly, which is the
  part that would otherwise be easy to get wrong.
- Tools used (condition F): n/a.
- Approximate time spent, if you can tell: about 5 minutes.
