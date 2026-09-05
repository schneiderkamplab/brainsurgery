# T5: LoRA adapter merge with sharded export (Pythia-1B)

## Objective

Fold a LoRA adapter into the base weights so that the result is a plain dense
checkpoint with no adapter tensors, then write it sharded. This is what
adapter frameworks call "merge and unload", done directly on the checkpoint
files.

## Why it is meaningful

Merging a LoRA adapter into the base weights ("merge and unload") is the last
step before deploying an adapted model without an adapter runtime. Doing it
directly on the checkpoint files avoids instantiating the model. Two details
decide correctness: the adapter's scaling factor alpha over r, and the
relation between the adapter's factor layout and the base weight layout,
which PEFT records as `fan_in_fan_out`. A correct solution has to map adapter
names to base names, multiply, scale, add, and leave no
adapter or intermediate tensor in a sharded output.

## Environment

This task runs in its own sandbox: a fresh working directory and a fresh
Python environment that contains only the packages of your condition. Nothing
from other tasks, other conditions or earlier runs is available, and nothing
you do here is visible to them. Inputs are under `inputs/` (read-only). Write
only under `out/`. Do not leave the sandbox directory.

## Inputs

- `inputs/base/model.safetensors`: Pythia-1B: the base, 244 tensors, float16.
- `inputs/lora/adapter_model.safetensors`: a PEFT-style adapter with
  32 tensors, float32. For each layer `i` in 0..15 and each
  adapted module in `attention.query_key_value`:
  - `base_model.model.gpt_neox.layers.<i>.<module>.lora_A.weight`, shape `[16, 2048]`
  - `base_model.model.gpt_neox.layers.<i>.<module>.lora_B.weight`, shape `[6144, 16]`
- `inputs/lora/adapter_config.json`: `r = 16`, `lora_alpha = 32`,
  `target_modules = ["attention.query_key_value"]`, `fan_in_fan_out = false`.

The adapted base tensors are `gpt_neox.layers.<i>.<module>.weight`, shape
`[6144, 2048]`. They use the `nn.Linear` layout `[out, in]`, the same convention as the adapter factors, so `B @ A` (shape `[out, in]`) is added without transposition (`fan_in_fan_out = false`).

## Required result

For every layer `i` and every adapted module, with `A = lora_A.weight`,
`B = lora_B.weight` and `scale = lora_alpha / r = 2`:

1. `gpt_neox.layers.<i>.<module>.weight += scale * B @ A`, computed in
   float32, then cast back to float16 (the base dtype); the tolerance below absorbs the rounding. The result keeps its name, shape `[6144, 2048]` and
   dtype float16.
2. No adapter tensor and no intermediate tensor appears in the output.
3. Every other base tensor is unchanged.
4. Output: a sharded safetensors checkpoint in the directory `out/T5/`:
   - shard files plus an index file `model.safetensors.index.json` whose
     `weight_map` maps every tensor name to the shard file that holds it;
   - the tensors in one shard total at most 512 MiB (536,870,912 bytes) of tensor data, not
     counting file headers. A single tensor larger than that (here
     `gpt_neox.embed_in.weight` and `embed_out.weight`, 206 MB each) is stored alone in its own shard.
   - Expected total: 244 tensors, the same names as the base.

## Required checks

Your solution must fail loudly if any of these does not hold before writing:

- exactly 16 adapter pairs were found and merged;
- no tensor name containing `lora_` is in the output;
- `gpt_neox.layers.0.attention.query_key_value.weight` still has shape `[6144, 2048]`;
- the output has exactly 244 tensors.

## Grading

`grade.py T5 --target pythia-1b` compares `out/T5` with a hidden reference:
sharding rules, exact key set, shapes, dtypes, bit-exact values for the
228 unchanged tensors, and for the 16 merged weights a relative
Frobenius error of at most 1e-3.
