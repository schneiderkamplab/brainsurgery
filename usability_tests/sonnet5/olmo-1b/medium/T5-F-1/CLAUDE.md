# Condition F: free choice of tooling

You are given a checkpoint-editing task (TASK.md below). Solve it with
whatever approach you judge best, using only the repositories and Python
packages listed in `F-allowed.md` (they are pre-installed or checked out in
your environment). Existing tools are fair game: model-merging toolkits,
adapter libraries, checkpoint utilities, or plain scripts on top of them. Do
not use the `brainsurgery` package or CLI, and do not install anything else.

Your environment: this sandbox directory is your working directory and your
Python environment is private to this run (see "Environment" in TASK.md).
Inputs are under `inputs/`; the output must be written exactly where TASK.md
says, under `out/`. Where an input directory contains HuggingFace config and
tokenizer files, you may use them.

Rules:

- Put everything you author under `out/<task>/` (for example
  `out/<task>/solution.py`, a config file for a tool, or a shell script
  `out/<task>/run.sh` that invokes a tool). Every execution of a script, tool
  command or plan that is meant to produce the output counts as an attempt.
- Whatever you use, the "Required checks" in TASK.md must be enforced by
  something you run, so that the run fails if they do not hold.
- When you are done, write `out/<task>/REPORT.md` with the fields in
  `record-template.md`, section "Participant self-report", and state which
  tools you used and why.

---

# T5: LoRA adapter merge with sharded export (OLMo-1B-0724-hf)

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

- `inputs/base/`: OLMo-1B-0724-hf as a sharded safetensors directory (two shard files plus `model.safetensors.index.json`): the base, 114 tensors, float32.
- `inputs/lora/adapter_model.safetensors`: a PEFT-style adapter with
  64 tensors, float32. For each layer `i` in 0..15 and each
  adapted module in `self_attn.q_proj`, `self_attn.v_proj`:
  - `base_model.model.model.layers.<i>.<module>.lora_A.weight`, shape `[16, 2048]`
  - `base_model.model.model.layers.<i>.<module>.lora_B.weight`, shape `[2048, 16]`
- `inputs/lora/adapter_config.json`: `r = 16`, `lora_alpha = 32`,
  `target_modules = ["self_attn.q_proj", "self_attn.v_proj"]`, `fan_in_fan_out = false`.

The adapted base tensors are `model.layers.<i>.<module>.weight`, shape
`[2048, 2048]`. They use the `nn.Linear` layout `[out, in]`, the same convention as the adapter factors, so `B @ A` (shape `[out, in]`) is added without transposition (`fan_in_fan_out = false`).

## Required result

For every layer `i` and every adapted module, with `A = lora_A.weight`,
`B = lora_B.weight` and `scale = lora_alpha / r = 2`:

1. `model.layers.<i>.<module>.weight += scale * B @ A`, computed in
   float32. The result keeps its name, shape `[2048, 2048]` and
   dtype float32.
2. No adapter tensor and no intermediate tensor appears in the output.
3. Every other base tensor is unchanged.
4. Output: a sharded safetensors checkpoint in the directory `out/T5/`:
   - shard files plus an index file `model.safetensors.index.json` whose
     `weight_map` maps every tensor name to the shard file that holds it;
   - the tensors in one shard total at most 512 MiB (536,870,912 bytes) of tensor data, not
     counting file headers. A single tensor larger than that (here
     `model.embed_tokens.weight` and `lm_head.weight`, 412 MB each) is stored alone in its own shard.
   - Expected total: 114 tensors, the same names as the base.

## Required checks

Your solution must fail loudly if any of these does not hold before writing:

- exactly 32 adapter pairs were found and merged;
- no tensor name containing `lora_` is in the output;
- `model.layers.0.self_attn.q_proj.weight` still has shape `[2048, 2048]`;
- the output has exactly 114 tensors.

## Grading

`grade.py T5 --target olmo-1b` compares `out/T5` with a hidden reference:
sharding rules, exact key set, shapes, dtypes, bit-exact values for the
82 unchanged tensors, and for the 32 merged weights a relative
Frobenius error of at most 1e-5.
