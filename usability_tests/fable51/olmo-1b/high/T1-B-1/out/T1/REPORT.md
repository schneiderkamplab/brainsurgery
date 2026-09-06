# T1 participant self-report (condition B, OLMo-1B-0724-hf)

- Final artifact path: `out/T1/plan.yaml` (output checkpoint: `out/T1/model.safetensors`, 86 tensors, single file)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution passed all 21 procedures and wrote the output.
- Pitfalls or surprises you hit (one line each):
  - `move` refuses existing destinations, so renumbering had to be ordered by ascending old index (delete first, then 3->2, 4->3, ...); with that order no destination ever exists at move time, which is exactly the collision guard the task warns about.
  - Regex references are full-match, so a block pattern needs an explicit tail such as `model\.layers\.3\.(.+)`; full-match also means `layers\.1\.` can never bleed into `layers.11`, which removes the classic prefix-overmatch hazard of this task.
  - `not` takes a bare assert expression (`not: { exists: ... }`), and `exists` takes the reference directly rather than an `of:` key, unlike `count`/`shape`; found this from the README's `all` example and `docpack/examples/validation.yaml`.
  - The single-file output path with a `.safetensors` suffix is written unsharded, so no `shard` setting was needed for a 4 GB file.
- Anything in the task text or documentation that was unclear:
  - The README plan format shows only file inputs; that a sharded HF directory with `model.safetensors.index.json` is accepted as an input is only visible from the example plan and the interfaces reference.
  - `help.txt` for `move` does not mention that regex capture groups in `from` can be used in `to` (`\1`); that is documented under `assert.equal` and in the interfaces reference instead.
- Tools used (condition F): not applicable (condition B).
- Approximate time spent, if you can tell: about 5 minutes wall clock, most of it reading the doc pack; the plan ran in roughly 8 seconds.
