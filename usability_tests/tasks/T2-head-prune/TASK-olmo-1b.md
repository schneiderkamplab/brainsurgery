# T2: Structured attention-head pruning (OLMo-1B-0724-hf)

## Objective

Remove one attention head from every layer of OLMo-1B-0724-hf at the checkpoint
level. Pruning a head means removing its slice from every head-bearing
projection tensor: the input-side projections that produce the head's query,
key and value, and the output projection that consumes it. The result must be
loadable as the same architecture with 15 heads per layer.

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

- `inputs/base/`: OLMo-1B-0724-hf as a sharded safetensors directory (two shard files plus `model.safetensors.index.json`): 114 tensors, float32, 16 layers,
  16 heads of 128 dimensions each, hidden size 2048.
  All projection matrices use the `nn.Linear` layout `[out, in]`.

Per layer `i` in 0..15 the head-bearing tensors are:

- `model.layers.<i>.self_attn.q_proj.weight`, shape `[2048, 2048]`: query projection; heads are row blocks.
- `model.layers.<i>.self_attn.k_proj.weight`, shape `[2048, 2048]`: key projection; heads are row blocks.
- `model.layers.<i>.self_attn.v_proj.weight`, shape `[2048, 2048]`: value projection; heads are row blocks.
- `model.layers.<i>.self_attn.o_proj.weight`, shape `[2048, 2048]`: output projection; heads are column blocks.

Not per head, and therefore untouched: the three MLP matrices.

## Required result

For every layer `i`, remove head 5 (heads are numbered from 0):

1. `model.layers.<i>.self_attn.q_proj.weight` becomes `[1920, 2048]`: keep rows `0..639`, `768..2047`, in that order.
2. `model.layers.<i>.self_attn.k_proj.weight` becomes `[1920, 2048]`: keep rows `0..639`, `768..2047`, in that order.
3. `model.layers.<i>.self_attn.v_proj.weight` becomes `[1920, 2048]`: keep rows `0..639`, `768..2047`, in that order.
4. `model.layers.<i>.self_attn.o_proj.weight` becomes `[2048, 1920]`: keep columns `0..639`, `768..2047`, in that order.
5. Every other tensor is unchanged. Tensor names do not change.
6. Output: a single file `out/T2/model.safetensors` with exactly 114 tensors.

## Required checks

Your solution must fail loudly if any of these does not hold before writing:

- `model.layers.0.self_attn.q_proj.weight` has shape `[1920, 2048]`;
- `model.layers.0.self_attn.k_proj.weight` has shape `[1920, 2048]`;
- `model.layers.0.self_attn.v_proj.weight` has shape `[1920, 2048]`;
- `model.layers.0.self_attn.o_proj.weight` has shape `[2048, 1920]`;
- the output has exactly 114 tensors.

## Grading

`grade.py T2 --target olmo-1b` compares `out/T2` with a hidden reference:
exact key set, shapes, dtypes and bit-exact values.
