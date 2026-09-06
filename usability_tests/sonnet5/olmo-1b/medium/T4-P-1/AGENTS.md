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

# T4: Task-vector merge of two fine-tunes (OLMo-1B-0724-hf)

## Objective

Merge two fine-tunes of the same base model by task arithmetic: add a scaled
copy of each fine-tune's change (its task vector) to the base. Both fine-tunes
were trained with a frozen backbone, so only their MLP tensors differ from
the base; the merge must verify that assumption before touching anything.

## Why it is meaningful

Task arithmetic (adding scaled task vectors of several fine-tunes to a base)
is a widely used way to combine skills without retraining. The precondition
matters: it only makes sense if the fine-tunes share the base everywhere
except the tensors that were trained, so a careful merge verifies that before
touching anything. The arithmetic hazard is ordering: each task vector must be
taken against the unmodified base, not against a base that the first merge
already changed. A correct solution has to check three checkpoints against
each other, compute the merge in the right order, and leave everything else
untouched.

## Environment

This task runs in its own sandbox: a fresh working directory and a fresh
Python environment that contains only the packages of your condition. Nothing
from other tasks, other conditions or earlier runs is available, and nothing
you do here is visible to them. Inputs are under `inputs/` (read-only). Write
only under `out/`. Do not leave the sandbox directory.

## Inputs

- `inputs/base/`: OLMo-1B-0724-hf as a sharded safetensors directory (two shard files plus `model.safetensors.index.json`): the base, 114 tensors, float32.
- `inputs/ft1/model.safetensors`: fine-tune 1, same 114 names, shapes and dtypes.
- `inputs/ft2/model.safetensors`: fine-tune 2, same layout.

The MLP tensors are, per layer `i` in 0..15:
- `model.layers.<i>.mlp.gate_proj.weight` (`[8192, 2048]`)
- `model.layers.<i>.mlp.up_proj.weight` (`[8192, 2048]`)
- `model.layers.<i>.mlp.down_proj.weight` (`[2048, 8192]`)
48 tensors in total.

## Required result

1. Before doing anything else, verify that the three checkpoints have the
   same tensor names and that every tensor outside the 48 MLP tensors is
   identical in all three. Abort with an error if not.
2. For each of the 48 MLP tensors `X`, with `lambda = 0.4`:

       out[X] = base[X] + lambda * (ft1[X] - base[X]) + lambda * (ft2[X] - base[X])

   computed in float32.
3. Every other tensor is taken from the base unchanged. Tensor names do not
   change.
4. Output: a single file `out/T4/model.safetensors` with exactly 114 tensors.

## Required checks

Your solution must fail loudly if any of these does not hold:

- the shared-tensor verification in step 1;
- exactly 48 tensors were merged;
- the output has exactly 114 tensors.

## Grading

`grade.py T4 --target olmo-1b` compares `out/T4` with a hidden reference:
exact key set, shapes, dtypes, bit-exact values for the 66 unchanged
tensors, and for the 48 merged tensors a relative Frobenius error of at
most 1e-5 (so a different order of additions is fine).
