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
