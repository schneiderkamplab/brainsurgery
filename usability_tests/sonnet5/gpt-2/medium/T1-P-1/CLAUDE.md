# Condition P: Python / PyTorch baseline

You are given a checkpoint-editing task (TASK.md below). Solve it with a
standalone Python script using PyTorch and the `safetensors` package, the way
a practitioner writes a one-off state-dict script. Only the packages listed
in `requirements-P.txt` are installed in your environment. Do not use the
`brainsurgery` package or CLI, and do not install anything.

Your environment: this sandbox directory is your working directory and your
Python environment is private to this run (see "Environment" in TASK.md).
Inputs are under `inputs/`; the output must be written exactly where TASK.md
says, under `out/`.

Rules:

- Write your script to `out/<task>/solution.py` and run it with
  `python out/<task>/solution.py`. Every execution of the script counts as an
  attempt; do not test ideas in a REPL or with partial snippets.
- Your script must implement the "Required checks" in TASK.md and fail
  loudly if they do not hold.
- When you are done, write `out/<task>/REPORT.md` with the fields in
  `record-template.md`, section "Participant self-report".

---

# T1: Depth pruning with layer renumbering (GPT-2 (124M))

## Objective

Produce a 9-layer GPT-2 (124M) checkpoint by removing 3 transformer
blocks from the 12-layer model and renumbering the remaining blocks so
that layer indices are contiguous again. This is the checkpoint side of depth
pruning: the result must load into a 9-layer configuration of the same
architecture.

## Why it is meaningful

Depth pruning is a standard way to shrink a model for cheaper inference or to
build a student for distillation: drop whole transformer blocks, then keep
training. The checkpoint side is a bulk rename with a collision hazard: if
blocks are shifted in the wrong order, a block overwrites a surviving one, and
the result still loads and runs, silently wrong. A correct solution has to
target whole blocks by pattern, delete them, renumber the rest without
collisions, and prove afterwards that exactly 9 blocks remain.

## Environment

This task runs in its own sandbox: a fresh working directory and a fresh
Python environment that contains only the packages of your condition. Nothing
from other tasks, other conditions or earlier runs is available, and nothing
you do here is visible to them. Inputs are under `inputs/` (read-only). Write
only under `out/`. Do not leave the sandbox directory.

## Input

- `inputs/base/model.safetensors`: GPT-2 (124M): 160 tensors, float32.
  Each transformer block `i` in 0..11 owns 13 tensors named
  `h.<i>.<rest>`: layer norms `ln_1`/`ln_2` (weight and bias), the fused attention projection `attn.c_attn` (weight `[768, 2304]`, bias `[2304]`), `attn.c_proj` (weight `[768, 768]`, bias `[768]`), the causal-mask buffer `attn.bias` (`[1, 1, 1024, 1024]`), and the MLP `mlp.c_fc` (weight `[768, 3072]`, bias `[3072]`) and `mlp.c_proj` (weight `[3072, 768]`, bias `[768]`). The
  remaining 4 tensors are `wte.weight` (`[50257, 768]`), `wpe.weight` (`[1024, 768]`), `ln_f.weight` (`[768]`), `ln_f.bias` (`[768]`). GPT-2 stores projection matrices as `[in, out]` (Conv1D layout), the transpose of `nn.Linear`. The
  directory also holds the HuggingFace config and tokenizer files of the
  12-layer model.

## Required result

1. Remove every tensor of blocks 2, 5, 8.
2. Renumber the surviving blocks in their original order so that indices run
   0..8 without gaps: old 0 stays 0, old 1 stays 1, old 3 becomes 2, old 4 becomes 3, old 6 becomes 4, old 7 becomes 5, old 9 becomes 6, old 10 becomes 7, old 11 becomes 8. Only the block index in the
   name changes; the rest of each name, and all values, shapes and dtypes,
   stay the same.
3. The 4 non-block tensors are unchanged.
4. Output: a single file `out/T1/model.safetensors` with exactly 121
   tensors.

## Required checks

Your solution must fail loudly (non-zero exit, no output written) if any of
these does not hold:

- no tensor of blocks 9, 10, 11 remains;
- exactly 9 blocks remain (for example, exactly 9 tensors match
  `h.<i>.attn.c_attn.weight`);
- the output has exactly 121 tensors.

## Grading

`grade.py T1 --target gpt-2` compares `out/T1` with a hidden reference:
exact key set, shapes, dtypes and bit-exact values.
