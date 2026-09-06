# Condition F: free choice of tooling

You are given a checkpoint-editing task (TASK.md below). Solve it with
whatever approach you judge best, using only the repositories and Python
packages listed in `F-allowed.md` (they are pre-installed or checked out in
your environment). Existing tools are fair game: model-merging toolkits,
adapter libraries, checkpoint utilities, or plain scripts on top of them. Do
not use the `brainsurgery` package or CLI, and do not install anything else.

Your environment: this sandbox directory is your working directory and your
Python environment is private to this run (see "Environment" in TASK.md).
Inputs are under `inputs/`; the output must be written exactly where TASK.md
says, under `out/`. Where an input directory contains HuggingFace config and
tokenizer files, you may use them.

Rules:

- Put everything you author under `out/<task>/` (for example
  `out/<task>/solution.py`, a config file for a tool, or a shell script
  `out/<task>/run.sh` that invokes a tool). Every execution of a script, tool
  command or plan that is meant to produce the output counts as an attempt.
- Whatever you use, the "Required checks" in TASK.md must be enforced by
  something you run, so that the run fails if they do not hold.
- When you are done, write `out/<task>/REPORT.md` with the fields in
  `record-template.md`, section "Participant self-report", and state which
  tools you used and why.

---

# T1: Depth pruning with layer renumbering (Pythia-1B)

## Objective

Produce a 12-layer Pythia-1B checkpoint by removing 4 transformer
blocks from the 16-layer model and renumbering the remaining blocks so
that layer indices are contiguous again. This is the checkpoint side of depth
pruning: the result must load into a 12-layer configuration of the same
architecture.

## Why it is meaningful

Depth pruning is a standard way to shrink a model for cheaper inference or to
build a student for distillation: drop whole transformer blocks, then keep
training. The checkpoint side is a bulk rename with a collision hazard: if
blocks are shifted in the wrong order, a block overwrites a surviving one, and
the result still loads and runs, silently wrong. A correct solution has to
target whole blocks by pattern, delete them, renumber the rest without
collisions, and prove afterwards that exactly 12 blocks remain.

## Environment

This task runs in its own sandbox: a fresh working directory and a fresh
Python environment that contains only the packages of your condition. Nothing
from other tasks, other conditions or earlier runs is available, and nothing
you do here is visible to them. Inputs are under `inputs/` (read-only). Write
only under `out/`. Do not leave the sandbox directory.

## Input

- `inputs/base/model.safetensors`: Pythia-1B: 244 tensors, float16.
  Each transformer block `i` in 0..15 owns 15 tensors named
  `gpt_neox.layers.<i>.<rest>`: layer norms `input_layernorm`/`post_attention_layernorm` (weight and bias), the fused attention projection `attention.query_key_value` (weight `[6144, 2048]`, bias `[6144]`), `attention.dense` (weight `[2048, 2048]`, bias `[2048]`), the MLP `mlp.dense_h_to_4h` (weight `[8192, 2048]`, bias `[8192]`) and `mlp.dense_4h_to_h` (weight `[2048, 8192]`, bias `[2048]`), and three non-parameter buffers: `attention.bias` (`[1, 1, 2048, 2048]`, uint8 causal mask), `attention.masked_bias` (scalar) and `attention.rotary_emb.inv_freq` (`[32]`). The
  remaining 4 tensors are `gpt_neox.embed_in.weight` (`[50304, 2048]`), `embed_out.weight` (`[50304, 2048]`), `gpt_neox.final_layer_norm.weight` (`[2048]`), `gpt_neox.final_layer_norm.bias` (`[2048]`). All projection matrices use the `nn.Linear` layout `[out, in]`. The checkpoint is stored in float16. In the fused `query_key_value` projection the 6144 rows are ordered per head: head `h` owns rows `768*h .. 768*h+767`, and inside that block the first 256 rows are its query, the next 256 its key and the last 256 its value (GPT-NeoX interleaved layout, not `[q | k | v]` segments). The
  directory also holds the HuggingFace config and tokenizer files of the
  16-layer model.

## Required result

1. Remove every tensor of blocks 2, 6, 10, 14.
2. Renumber the surviving blocks in their original order so that indices run
   0..11 without gaps: old 0 stays 0, old 1 stays 1, old 3 becomes 2, old 4 becomes 3, old 5 becomes 4, old 7 becomes 5, old 8 becomes 6, old 9 becomes 7, old 11 becomes 8, old 12 becomes 9, old 13 becomes 10, old 15 becomes 11. Only the block index in the
   name changes; the rest of each name, and all values, shapes and dtypes,
   stay the same.
3. The 4 non-block tensors are unchanged.
4. Output: a single file `out/T1/model.safetensors` with exactly 184
   tensors.

## Required checks

Your solution must fail loudly (non-zero exit, no output written) if any of
these does not hold:

- no tensor of blocks 12, 13, 14, 15 remains;
- exactly 12 blocks remain (for example, exactly 12 tensors match
  `gpt_neox.layers.<i>.attention.query_key_value.weight`);
- the output has exactly 184 tensors.

## Grading

`grade.py T1 --target pythia-1b` compares `out/T1` with a hidden reference:
exact key set, shapes, dtypes and bit-exact values.
