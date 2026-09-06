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

# T5: LoRA adapter merge with sharded export (GPT-2 (124M))

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
names to base names, multiply, scale, transpose, add, and leave no
adapter or intermediate tensor in a sharded output.

## Environment

This task runs in its own sandbox: a fresh working directory and a fresh
Python environment that contains only the packages of your condition. Nothing
from other tasks, other conditions or earlier runs is available, and nothing
you do here is visible to them. Inputs are under `inputs/` (read-only). Write
only under `out/`. Do not leave the sandbox directory.

## Inputs

- `inputs/base/model.safetensors`: GPT-2 (124M): the base, 160 tensors, float32.
- `inputs/lora/adapter_model.safetensors`: a PEFT-style adapter with
  24 tensors, float32. For each layer `i` in 0..11 and each
  adapted module in `attn.c_attn`:
  - `base_model.model.h.<i>.<module>.lora_A.weight`, shape `[16, 768]`
  - `base_model.model.h.<i>.<module>.lora_B.weight`, shape `[2304, 16]`
- `inputs/lora/adapter_config.json`: `r = 16`, `lora_alpha = 32`,
  `target_modules = ["attn.c_attn"]`, `fan_in_fan_out = true`.

The adapted base tensors are `h.<i>.<module>.weight`, shape
`[768, 2304]`. They use the Conv1D layout `[in, out]`, while the adapter factors follow the `nn.Linear` convention, which is what `fan_in_fan_out = true` signals: the low-rank product `B @ A` (shape `[out, in]`) must be transposed before it is added.

## Required result

For every layer `i` and every adapted module, with `A = lora_A.weight`,
`B = lora_B.weight` and `scale = lora_alpha / r = 2`:

1. `h.<i>.<module>.weight += scale * (B @ A).T`, computed in
   float32. The result keeps its name, shape `[768, 2304]` and
   dtype float32.
2. No adapter tensor and no intermediate tensor appears in the output.
3. Every other base tensor is unchanged.
4. Output: a sharded safetensors checkpoint in the directory `out/T5/`:
   - shard files plus an index file `model.safetensors.index.json` whose
     `weight_map` maps every tensor name to the shard file that holds it;
   - the tensors in one shard total at most 100 MiB (104,857,600 bytes) of tensor data, not
     counting file headers. A single tensor larger than that (here
     `wte.weight`, 154 MB) is stored alone in its own shard.
   - Expected total: 160 tensors, the same names as the base.

## Required checks

Your solution must fail loudly if any of these does not hold before writing:

- exactly 12 adapter pairs were found and merged;
- no tensor name containing `lora_` is in the output;
- `h.0.attn.c_attn.weight` still has shape `[768, 2304]`;
- the output has exactly 160 tensors.

## Grading

`grade.py T5 --target gpt-2` compares `out/T5` with a hidden reference:
sharding rules, exact key set, shapes, dtypes, bit-exact values for the
148 unchanged tensors, and for the 12 merged weights a relative
Frobenius error of at most 1e-5.
