# Condition B: BrainSurgery plan

You are given a checkpoint-editing task (TASK.md below). Solve it with a
BrainSurgery plan: a YAML file with `inputs`, `transforms` and `output`,
executed with the `brainsurgery` command-line tool. Do not write Python; the
whole edit must be expressed in the plan. Reading the tool's documentation
and running `brainsurgery` with `help` transforms is allowed and encouraged.

Documentation available to you (the doc pack):

- `docpack/README.md` (BrainSurgery README: plan format, tensor references,
  transform list, assert operators, output behavior)
- `docpack/interfaces-reference.md`
- `docpack/help.txt` (the built-in `help` output for every transform and
  assert expression)
- `docpack/examples/` (worked example plans unrelated to the tasks)

Your environment: this sandbox directory is your working directory and your
Python environment is private to this run (see "Environment" in TASK.md).
Only `brainsurgery` and its dependencies are installed. Inputs are under
`inputs/`; the output must be written exactly where TASK.md says, under
`out/`.

Rules:

- Write your plan to `out/<task>/plan.yaml` and run it with
  `brainsurgery out/<task>/plan.yaml` (CLI options such as `--provider` are
  allowed). Every execution of the plan counts as an attempt.
- Your plan must implement the "Required checks" in TASK.md as `assert`
  transforms, so the run fails if they do not hold.
- When you are done, write `out/<task>/REPORT.md` with the fields in
  `record-template.md`, section "Participant self-report".

---

# T1: Depth pruning with layer renumbering (OLMo-1B-0724-hf)

## Objective

Produce a 12-layer OLMo-1B-0724-hf checkpoint by removing 4 transformer
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

- `inputs/base/`: OLMo-1B-0724-hf as a sharded safetensors directory (two shard files plus `model.safetensors.index.json`): 114 tensors, float32.
  Each transformer block `i` in 0..15 owns 7 tensors named
  `model.layers.<i>.<rest>`: separate attention projections `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`, `self_attn.o_proj` (all `[2048, 2048]`) and the MLP `mlp.gate_proj` (`[8192, 2048]`), `mlp.up_proj` (`[8192, 2048]`), `mlp.down_proj` (`[2048, 8192]`). There are no biases and no layer-norm parameters (the model uses non-parametric layer norm). The
  remaining 2 tensors are `model.embed_tokens.weight` (`[50304, 2048]`), `lm_head.weight` (`[50304, 2048]`). All projection matrices use the `nn.Linear` layout `[out, in]`. The
  directory also holds the HuggingFace config and tokenizer files of the
  16-layer model.

## Required result

1. Remove every tensor of blocks 2, 6, 10, 14.
2. Renumber the surviving blocks in their original order so that indices run
   0..11 without gaps: old 0 stays 0, old 1 stays 1, old 3 becomes 2, old 4 becomes 3, old 5 becomes 4, old 7 becomes 5, old 8 becomes 6, old 9 becomes 7, old 11 becomes 8, old 12 becomes 9, old 13 becomes 10, old 15 becomes 11. Only the block index in the
   name changes; the rest of each name, and all values, shapes and dtypes,
   stay the same.
3. The 2 non-block tensors are unchanged.
4. Output: a single file `out/T1/model.safetensors` with exactly 86
   tensors.

## Required checks

Your solution must fail loudly (non-zero exit, no output written) if any of
these does not hold:

- no tensor of blocks 12, 13, 14, 15 remains;
- exactly 12 blocks remain (for example, exactly 12 tensors match
  `model.layers.<i>.self_attn.q_proj.weight`);
- the output has exactly 86 tensors.

## Grading

`grade.py T1 --target olmo-1b` compares `out/T1` with a hidden reference:
exact key set, shapes, dtypes and bit-exact values.
