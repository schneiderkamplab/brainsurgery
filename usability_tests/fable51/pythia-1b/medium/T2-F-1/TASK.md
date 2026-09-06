# T2: Structured attention-head pruning (Pythia-1B)

## Objective

Remove one attention head from every layer of Pythia-1B at the checkpoint
level. Pruning a head means removing its slice from every head-bearing
projection tensor: the input-side projections that produce the head's query,
key and value, and the output projection that consumes it. The result must be
loadable as the same architecture with 7 heads per layer.

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

- `inputs/base/model.safetensors`: Pythia-1B: 244 tensors, float16, 16 layers,
  8 heads of 256 dimensions each, hidden size 2048.
  All projection matrices use the `nn.Linear` layout `[out, in]`. The checkpoint is stored in float16. In the fused `query_key_value` projection the 6144 rows are ordered per head: head `h` owns rows `768*h .. 768*h+767`, and inside that block the first 256 rows are its query, the next 256 its key and the last 256 its value (GPT-NeoX interleaved layout, not `[q | k | v]` segments).

Per layer `i` in 0..15 the head-bearing tensors are:

- `gpt_neox.layers.<i>.attention.query_key_value.weight`, shape `[6144, 2048]`: fused projection, interleaved per head; a head is one 768-row block holding its q, k and v.
- `gpt_neox.layers.<i>.attention.query_key_value.bias`, shape `[6144]`: same layout as the rows of `attention.query_key_value.weight`.
- `gpt_neox.layers.<i>.attention.dense.weight`, shape `[2048, 2048]`: output projection; heads are 256-wide column blocks.

Not per head, and therefore untouched: attention.dense.bias (`[2048]`), the three attention buffers, the MLP tensors.

## Required result

For every layer `i`, remove head 5 (heads are numbered from 0):

1. `gpt_neox.layers.<i>.attention.query_key_value.weight` becomes `[5376, 2048]`: keep rows `0..3839`, `4608..6143`, in that order.
2. `gpt_neox.layers.<i>.attention.query_key_value.bias` becomes `[5376]`: keep rows `0..3839`, `4608..6143`, in that order.
3. `gpt_neox.layers.<i>.attention.dense.weight` becomes `[2048, 1792]`: keep columns `0..1279`, `1536..2047`, in that order.
4. Every other tensor is unchanged. Tensor names do not change.
5. Output: a single file `out/T2/model.safetensors` with exactly 244 tensors.

## Required checks

Your solution must fail loudly if any of these does not hold before writing:

- `gpt_neox.layers.0.attention.query_key_value.weight` has shape `[5376, 2048]`;
- `gpt_neox.layers.0.attention.query_key_value.bias` has shape `[5376]`;
- `gpt_neox.layers.0.attention.dense.weight` has shape `[2048, 1792]`;
- the output has exactly 244 tensors.

## Grading

`grade.py T2 --target pythia-1b` compares `out/T2` with a hidden reference:
exact key set, shapes, dtypes and bit-exact values.
