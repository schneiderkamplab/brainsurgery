# T1 Participant self-report

- **Final artifact path:** `out/T1/model.safetensors` (plan: `out/T1/plan.yaml`)

- **Number of times you executed the script or plan:** 1
  (`brainsurgery out/T1/plan.yaml`, exit 0, first attempt.)
  Two additional runs were *negative controls* on separate throwaway plan files,
  not on `out/T1/plan.yaml`: one with the moves reordered descending and one with
  a deliberately impossible `count` assert, both writing to a scratch output path.
  Both exited 1 and wrote no output, confirming the guards fire. Their files were
  deleted afterwards.

- **Which executions failed, and why (one line each):** none — the plan passed on
  its first execution.

- **Pitfalls or surprises you hit (one line each):**
  - The renumbering collision is the whole task: moves must run in *ascending*
    source order so every destination slot is already free (deleted, or vacated by
    an earlier move). `move` has a MUST_NOT_EXIST destination policy, so the wrong
    order fails with `MoveTransformError: move destination already exists` rather
    than silently clobbering a block — verified with negative control A.
  - Unescaped dots in patterns: `gpt_neox\.layers\.3\.(.*)` must escape the dots,
    otherwise `.` matches any character. Patterns are full-match, which is what
    keeps `layers\.1\.` from also hitting `layers.10.`/`layers.11.` and
    `layers\.(2|6|10|14)\.` from hitting `layers.12.`.
  - A block is 15 tensors, not just the 12 parameters: `attention.bias` (uint8
    `[1,1,2048,2048]`), `attention.masked_bias` and `attention.rotary_emb.inv_freq`
    are buffers that must move with the block. Matching whole blocks with
    `<block>\.(.*)` rather than enumerating parameter names handles them for free.
  - Counting alone cannot detect a collision — a clobbered block still yields 12
    blocks and 184 tensors. I loaded the checkpoint a second time under an `orig::`
    alias purely as an assert reference and checked every surviving block bit-exactly
    against its *original* index. Asserts do not write, so the output alias stayed
    unambiguously `model` and the second input did not confuse output inference.
  - Shell-side, not tool-side: I generated the repetitive move/assert lines with a
    loop and the first attempt produced garbage because zsh does not word-split
    unquoted parameters. Caught by reading the generated YAML before running it.

- **Anything in the task text or documentation that was unclear:**
  - The README documents backreference rewriting (`\1`) for `assert.equal`'s `right`
    and says it works "exactly like `to` in `copy`/`move`", but the `move` help entry
    itself shows only literal one-to-one examples, so pattern-based renaming with
    capture groups has to be inferred from the `assert.equal`/`assign` docs or the
    example plan. A capture-group example directly under `move` would help.
  - "Every execution of the plan counts as an attempt" does not say whether
    exploratory runs of a *different* plan file count; I kept everything off
    `out/T1/plan.yaml` and reported the extra runs above to be safe.
  - The `output` alias-inference rule is documented clearly, but it is only obvious
    after reading it that a read-only reference alias is safe because asserts do not
    count as writes.

- **Tools used (condition F):** n/a (condition B).

- **Approximate time spent, if you can tell:** roughly 10 minutes end to end —
  most of it reading `help.txt`/README and verifying the result, ~7 s of plan runtime.

## Verification performed after the run

Independent of the plan's own asserts, I parsed the safetensors headers of input and
output directly and compared raw byte ranges:

- output has exactly 184 tensors: 12 blocks x 15 tensors + 4 non-block tensors;
- block indices present are exactly 0..11, 15 tensors each, no gaps;
- all 180 block tensors are byte-identical to their source block under the mapping
  0<-0, 1<-1, 2<-3, 3<-4, 4<-5, 5<-7, 6<-8, 7<-9, 8<-11, 9<-12, 10<-13, 11<-15;
- the 4 non-block tensors are byte-identical to the input;
- dtypes/shapes preserved, including the uint8 `attention.bias` masks.
