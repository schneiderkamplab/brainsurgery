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

# T2: Structured attention-head pruning (GPT-2 (124M))

## Objective

Remove one attention head from every layer of GPT-2 (124M) at the checkpoint
level. Pruning a head means removing its slice from every head-bearing
projection tensor: the input-side projections that produce the head's query,
key and value, and the output projection that consumes it. The result must be
loadable as the same architecture with 11 heads per layer.

## Why it is meaningful

Structured head pruning (removing whole attention heads found to be
redundant) is a well-studied way to speed up transformers, and it is done at
the checkpoint level so the pruned model loads with a smaller head count. The
work is in the layout: which axis of which tensor holds the heads, whether
query, key and value are fused, and in what order. Getting a block boundary
wrong produces a checkpoint that loads and runs with garbage attention. A
correct solution has to slice and reassemble tensors, keep the piece order
right, and check the resulting shapes on every projection.

## Environment

This task runs in its own sandbox: a fresh working directory and a fresh
Python environment that contains only the packages of your condition. Nothing
from other tasks, other conditions or earlier runs is available, and nothing
you do here is visible to them. Inputs are under `inputs/` (read-only). Write
only under `out/`. Do not leave the sandbox directory.

## Input

- `inputs/base/model.safetensors`: GPT-2 (124M): 160 tensors, float32, 12 layers,
  12 heads of 64 dimensions each, hidden size 768.
  GPT-2 stores projection matrices as `[in, out]` (Conv1D layout), the transpose of `nn.Linear`.

Per layer `i` in 0..11 the head-bearing tensors are:

- `h.<i>.attn.c_attn.weight`, shape `[768, 2304]`: fused `[q | k | v]` projection; heads are column blocks inside each 768-wide segment.
- `h.<i>.attn.c_attn.bias`, shape `[2304]`: same layout as the columns of `attn.c_attn.weight`.
- `h.<i>.attn.c_proj.weight`, shape `[768, 768]`: output projection; heads are row blocks.

Not per head, and therefore untouched: attn.c_proj.bias (`[768]`), attn.bias (`[1, 1, 1024, 1024]`, the mask buffer).

## Required result

For every layer `i`, remove head 5 (heads are numbered from 0):

1. `h.<i>.attn.c_attn.weight` becomes `[768, 2112]`: keep columns `0..319`, `384..767`, `768..1087`, `1152..1535`, `1536..1855`, `1920..2303`, in that order.
2. `h.<i>.attn.c_attn.bias` becomes `[2112]`: keep rows `0..319`, `384..767`, `768..1087`, `1152..1535`, `1536..1855`, `1920..2303`, in that order.
3. `h.<i>.attn.c_proj.weight` becomes `[704, 768]`: keep rows `0..319`, `384..767`, in that order.
4. Every other tensor is unchanged. Tensor names do not change.
5. Output: a single file `out/T2/model.safetensors` with exactly 160 tensors.

## Required checks

Your solution must fail loudly if any of these does not hold before writing:

- `h.0.attn.c_attn.weight` has shape `[768, 2112]`;
- `h.0.attn.c_attn.bias` has shape `[2112]`;
- `h.0.attn.c_proj.weight` has shape `[704, 768]`;
- the output has exactly 160 tensors.

## Grading

`grade.py T2 --target gpt-2` compares `out/T2` with a hidden reference:
exact key set, shapes, dtypes and bit-exact values.
