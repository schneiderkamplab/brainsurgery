# T1 self-report (Condition B: BrainSurgery plan)

- Final artifact path: `out/T1/model.safetensors` (plan: `out/T1/plan.yaml`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single run
  succeeded and wrote the output.
- Pitfalls or surprises you hit (one line each):
  - The collision hazard is entirely handled by ordering: renumbering in
    ascending destination order (3->2, 4->3, 6->4, 7->5, 9->6, 10->7, 11->8)
    means each destination slot has already been vacated by the delete or by
    the previous move, so `move`'s "destination must not exist" rule never
    trips. Doing it in descending order would have failed loudly instead of
    silently, which is a good property of the tool.
  - `move` rewrites `to` as a regex substitution over each `from` match, so a
    whole block is one line: `from: 'h\.3\.(.*)', to: 'h.2.\1'`. No need to
    enumerate the 13 tensors per block.
  - `assert: count` with `of: '.*'` is a cheap total-tensor check; for "nothing
    of blocks 9-11 remains" I used `not: { exists: ... }` rather than
    `count: is: 0`, since `count` is documented as an exact-match count and I
    did not want to depend on it accepting a zero-match reference.
  - Regexes must be single-quoted in YAML so `\.` and `\1` survive unescaped.
- Anything in the task text or documentation that was unclear:
  - `assert.dtype` documents `of` as a single tensor ("the tensor has the given
    dtype") but accepted a multi-match pattern (`of: '.*'`); whether it checks
    every match or just one is not stated in the docs. It passed, so the
    ambiguity did not matter here, but a plan relying on it as a global check
    is relying on undocumented behaviour.
  - The README lists `count`'s semantics but not whether a reference matching
    zero tensors is an error or a legal count of 0.
  - Otherwise the task text was fully specified: the exact old->new index map
    and the expected tensor totals removed all guesswork.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: ~10 minutes, most of it reading the
  doc pack (`README.md`, `help.txt` for `move`/`delete`/`assert.count`) before
  writing the plan.

## What the plan does

1. Pre-check: input has 160 tensors and 12 blocks.
2. `delete: { target: 'h\.(2|5|8)\..*' }` removes the 3 blocks (39 tensors);
   post-check 121 tensors and none of blocks 2/5/8 remain.
3. Seven `move` transforms renumber survivors in ascending destination order.
4. Required checks as `assert` transforms, all inside one `all` block: no
   `h.9/10/11` tensor exists, exactly 9 `h.<i>.attn.c_attn.weight`, exactly 121
   tensors total. Additional checks: 117 block tensors, all with indices 0..8;
   the 4 non-block tensors present with correct shapes; 9 `attn.bias` and 9
   `mlp.c_fc.weight` (blocks are complete, 13 each); dtype float32 throughout.
5. `output.path: out/T1/model.safetensors`, safetensors, unsharded.

Asserts run as transforms before the output is written, so any failure aborts
the run non-zero with no file produced.

## Verification done outside the plan

No Python was written. I parsed the safetensors JSON header of input and output
with shell tools (`od`, `head`, `grep`) and byte-compared tensor payloads with
`dd`/`cmp`: the output has 121 tensors, blocks 0..8 with exactly 13 tensors
each, plus the 4 non-block tensors; and `h.2.*`/`h.5.*`/`h.6.*`/`h.8.*` are
bit-identical to input `h.3.*`/`h.7.*`/`h.9.*`/`h.11.*` respectively, with
`ln_f.weight` and `wpe.weight` unchanged.
