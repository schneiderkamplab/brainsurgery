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

# T4: Task-vector merge of two fine-tunes (GPT-2 (124M))

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

- `inputs/base/model.safetensors`: GPT-2 (124M): the base, 160 tensors, float32.
- `inputs/ft1/model.safetensors`: fine-tune 1, same 160 names, shapes and dtypes.
- `inputs/ft2/model.safetensors`: fine-tune 2, same layout.

The MLP tensors are, per layer `i` in 0..11:
- `h.<i>.mlp.c_fc.weight` (`[768, 3072]`)
- `h.<i>.mlp.c_fc.bias` (`[3072]`)
- `h.<i>.mlp.c_proj.weight` (`[3072, 768]`)
- `h.<i>.mlp.c_proj.bias` (`[768]`)
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
4. Output: a single file `out/T4/model.safetensors` with exactly 160 tensors.

## Required checks

Your solution must fail loudly if any of these does not hold:

- the shared-tensor verification in step 1;
- exactly 48 tensors were merged;
- the output has exactly 160 tensors.

## Grading

`grade.py T4 --target gpt-2` compares `out/T4` with a hidden reference:
exact key set, shapes, dtypes, bit-exact values for the 112 unchanged
tensors, and for the 48 merged tensors a relative Frobenius error of at
most 1e-5 (so a different order of additions is fine).
