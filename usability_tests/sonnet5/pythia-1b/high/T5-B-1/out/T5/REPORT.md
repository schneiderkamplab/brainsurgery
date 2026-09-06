# Participant self-report

- Final artifact path: `out/T5/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution of `out/T5/plan.yaml` succeeded.
- Pitfalls or surprises you hit (one line each):
  - `matmul`/`add_`/`scale_`/`cast` with regex references only document that `from_a` (or the equivalent primary source) may include capture groups; it isn't spelled out that `from_b`/`to` are resolved as a *rewrite* of that same match (like `copy`'s `to`) rather than independently matched and paired up. I verified this by running a small synthetic two-layer checkpoint through an isolated `matmul` before trusting it on the real inputs.
  - The two embedding tensors (`gpt_neox.embed_in.weight`, `embed_out.weight`, ~196 MiB each) are well under the 512 MiB shard budget, so with this tool's default in-order first-fit packing they land in the same shard as several smaller tensors rather than each getting an exclusive shard; I did not add anything special for this since the "at most 512 MiB per shard" rule and the "oversized tensor alone" rule (which doesn't trigger here) were both satisfied.
- Anything in the task text or documentation that was unclear: the "Required result" section states the two embedding tensors are "stored alone in its own shard", but at ~196 MiB each (well under the 512 MiB budget) that isn't a consequence of the documented shard-packing rule (in-order first-fit up to budget, singleton only if a tensor alone exceeds the budget); I left the plan using the tool's documented default packing rather than adding ad hoc logic to force them into singleton shards.
- Tools used (condition F): n/a (condition B, brainsurgery plan only).
- Approximate time spent, if you can tell: about 20 minutes, most of it spent reading `docpack/help.txt`/`README.md` and validating the `matmul`/`from_b` rewrite semantics on a synthetic checkpoint before writing the real plan.
