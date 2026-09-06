# T1 (Pythia-1B, condition P) — participant self-report

- **Final artifact path:** `out/T1/model.safetensors` (184 tensors, 1 671 050 816 bytes);
  script at `out/T1/solution.py`.

- **Number of times you executed the script or plan:** 1

- **Which executions failed, and why (one line each):** none — the single
  execution succeeded.

- **Pitfalls or surprises you hit (one line each):**
  - The renumbering collision hazard the task warns about is only real if you
    rename in place; I sidestepped it by building a fresh output dict keyed by
    the new names and asserting on every insertion that the key is not already
    present, so an in-place-style bug would fail loudly instead of silently
    overwriting a survivor.
  - The three non-parameter buffers per block (`attention.bias` uint8
    `[1,1,2048,2048]`, `attention.masked_bias`, `attention.rotary_emb.inv_freq`)
    have to move with the block; a naming pattern aimed only at `weight`/`bias`
    of the projections would leave them behind and land on 12*12+4 tensors
    instead of 184. Matching on the whole `gpt_neox.layers.<i>.` prefix avoids
    this and keeps the mixed dtypes (F16 + U8) intact.
  - `attention.bias` also means the block prefix must be anchored: a loose
    `layers.1` style match would catch 1, 10..15. I used an anchored regex with
    escaped dots and an integer group, `^gpt_neox\.layers\.(\d+)\.(.+)$`.
  - safetensors rejects tensors that share storage, so I wrote contiguous
    clones. It was not needed here (each `get_tensor` allocates independently)
    but it is the cheap defence and RAM was ample.
  - "Fail loudly with no output written" pushed the ordering: every check runs
    against the in-memory dict before `save_file`, and the write goes to a
    `.tmp` path that is `os.replace`d into place, so a mid-write failure cannot
    leave a partial `model.safetensors`.

- **Anything in the task text or documentation that was unclear:**
  - Nothing blocking. The explicit old→new index table removed all ambiguity
    about the renumbering; I derived the same map programmatically from the
    drop set and checked it against the table printed at run time.
  - The task specifies "a single file `out/T1/model.safetensors`" and does not
    mention the HF `config.json` (whose `num_hidden_layers` is still 16) or the
    tokenizer files, so I copied nothing else into `out/T1`. If the graded
    result is meant to be loadable as a 12-layer model end to end, a patched
    `config.json` would be needed, but the "Required result" and the stated
    grading (key set, shapes, dtypes, values) cover the checkpoint only.
  - The detailed description of the GPT-NeoX interleaved `query_key_value`
    row layout is not needed for this task — no tensor is sliced or reordered
    here, only renamed. I assume it is shared boilerplate across the task set.

- **Tools used (condition F):** n/a — condition P. Only `torch` 2.14.0+cu130,
  `safetensors` 0.5.3 and the standard library (`os`, `re`, `sys`).

- **Approximate time spent, if you can tell:** ~6 minutes wall clock, most of
  it reading the task and writing the checks; the run itself took well under a
  minute.

## Checks implemented in the script

Required by the task:
- no tensor of blocks 12–15 remains (generalised: no surviving index >= 12);
- exactly 12 blocks remain, and exactly 12 tensors match
  `gpt_neox.layers.<i>.attention.query_key_value.weight`;
- the output has exactly 184 tensors.

Added as belt and braces:
- input is exactly blocks 0..15 plus the 4 named non-block tensors, with every
  key recognised and every block carrying the same 15-name tensor set;
- surviving indices are contiguous 0..11, not merely 12 in count;
- no name collision on any insertion into the output dict;
- no dropped block's tensor survives anywhere in the output (identity check);
- every kept tensor is bit-equal to its source with unchanged shape and dtype,
  and the 4 non-block tensors are unchanged.

After the run I additionally re-read the written file (read-only, not a solve
attempt) and confirmed: 184 tensors, blocks 0..11 with 15 tensors each, 12
`query_key_value.weight`, dtypes {F16, U8}, `format: pt` metadata preserved,
and spot-checked bit-exactness of renamed pairs including 15->11 and 13->10.
