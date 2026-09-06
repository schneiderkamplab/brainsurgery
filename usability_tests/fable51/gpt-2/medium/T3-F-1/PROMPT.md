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

# T3: Mixed-precision export with sharding (GPT-2 (124M))

## Objective

Prepare GPT-2 (124M) for a memory-constrained deployment: store the large
projection matrices in bfloat16, keep everything numerically sensitive
(embeddings, norms, biases) in float32, drop non-parameter buffers, and write
the result as a sharded checkpoint with an index file.

## Why it is meaningful

Exporting a checkpoint for deployment routinely mixes precisions: large
projection matrices in bfloat16 to halve the size, while embeddings, layer
norms and biases stay in float32 because they are small and numerically
sensitive. Sharding with an index file is what serving stacks expect. The
hazard is over-broad targeting: a pattern like `.*weight` also hits
embeddings and norms, and buffers such as causal masks are not parameters. A
correct solution has to cast exactly the intended matrices, drop the buffers,
upcast what must be float32, and produce a valid sharded layout.

## Environment

This task runs in its own sandbox: a fresh working directory and a fresh
Python environment that contains only the packages of your condition. Nothing
from other tasks, other conditions or earlier runs is available, and nothing
you do here is visible to them. Inputs are under `inputs/` (read-only). Write
only under `out/`. Do not leave the sandbox directory.

## Input

- `inputs/base/model.safetensors`: GPT-2 (124M): 160 tensors, float32, 12 layers.
  Per layer `i` in 0..11 the projection matrices are
  - `h.<i>.attn.c_attn.weight` (`[768, 2304]`)
  - `h.<i>.attn.c_proj.weight` (`[768, 768]`)
  - `h.<i>.mlp.c_fc.weight` (`[768, 3072]`)
  - `h.<i>.mlp.c_proj.weight` (`[3072, 768]`)
  Each layer also holds 1 non-parameter buffer(s): `h.<i>.attn.bias` (`[1, 1, 1024, 1024]`, the causal-mask buffer).

## Required result

1. Cast exactly the 48 projection matrices listed above to bfloat16
   (round-to-nearest-even, as `tensor.to(torch.bfloat16)` does).
2. Every other tensor is float32 in the output with unchanged values. This includes
   `wte.weight`, `wpe.weight`, all layer-norm weights and biases, and all projection biases.
3. Delete the 12 buffers `h.<i>.attn.bias`. Do not delete any parameter.
4. Tensor names do not change.
5. Output: a sharded safetensors checkpoint in the directory `out/T3/`:
   - shard files plus an index file `model.safetensors.index.json` whose
     `weight_map` maps every tensor name to the shard file that holds it;
   - the tensors in one shard total at most 64 MiB (67,108,864 bytes) of tensor data, not
     counting file headers. A single tensor larger than that (here
     `wte.weight`, 154 MB) is stored alone in its own shard.
   - Expected total: 148 tensors.

## Required checks

Your solution must fail loudly if any of these does not hold before writing:

- exactly 48 tensors are bfloat16;
- `h.0.attn.c_attn.weight` is bfloat16;
- `wte.weight` is float32;
- the output has exactly 148 tensors.

## Grading

`grade.py T3 --target gpt-2` compares `out/T3` with a hidden reference:
sharding rules, exact key set, shapes, dtypes and bit-exact values.
