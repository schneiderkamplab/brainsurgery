# T2 report

## Participant self-report

- Final artifact path: `out/T2/solution.py` (invoked via `out/T2/run.sh`), output at `out/T2/model.safetensors`.
- Number of times you executed the script or plan: 1.
- Which executions failed, and why (one line each): none.
- Pitfalls or surprises you hit (one line each):
  - The condition-F allowed list suggested `transformers` `prune_heads` as the
    route for this task, but in the pinned version (5.12.1) `GPT2Attention`
    has no `prune_heads` method and `transformers.pytorch_utils` has no
    `prune_conv1d_layer` / `find_pruneable_heads_and_indices` either, so that
    route is not actually available; switched to a plain slicing script on
    `torch`/`safetensors` instead.
  - The checkpoint's key names (`h.<i>...`, `wte.weight`, `wpe.weight`,
    `ln_f.*`, no `transformer.` prefix) are `GPT2Model`'s own state-dict
    naming, not `GPT2LMHeadModel`'s — irrelevant once slicing directly, but
    would matter for any `from_pretrained`-based route.
- Anything in the task text or documentation that was unclear: none; the
  column/row index ranges for head 5 were spelled out exactly in TASK.md and
  matched a plain per-64-column/row-block derivation.
- Tools used (condition F): `torch` 2.14.0 and `safetensors` 0.5.3 — plain
  tensor slicing on the loaded state dict. Did not use `transformers`
  `prune_heads` (unavailable in this version, see pitfalls), `mergekit`
  (built for merging/layer-slicing across models, not intra-tensor head
  slicing), `peft` (no adapters involved here), or `torch-state-bridge`
  (this task needs slicing along tensor axes, not key rewriting).
- Approximate time spent, if you can tell: a few minutes (one script, one
  successful run, verified bit-exact against manual index slices before
  reporting done).
