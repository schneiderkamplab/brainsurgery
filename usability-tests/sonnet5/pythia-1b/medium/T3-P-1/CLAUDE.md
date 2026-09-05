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

# T3: Mixed-precision export with sharding (Pythia-1B)

## Objective

Prepare Pythia-1B for a memory-constrained deployment: store the large
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

- `inputs/base/model.safetensors`: Pythia-1B: 244 tensors, float16, 16 layers.
  Per layer `i` in 0..15 the projection matrices are
  - `gpt_neox.layers.<i>.attention.query_key_value.weight` (`[6144, 2048]`)
  - `gpt_neox.layers.<i>.attention.dense.weight` (`[2048, 2048]`)
  - `gpt_neox.layers.<i>.mlp.dense_h_to_4h.weight` (`[8192, 2048]`)
  - `gpt_neox.layers.<i>.mlp.dense_4h_to_h.weight` (`[2048, 8192]`)
  Each layer also holds 3 non-parameter buffer(s): `gpt_neox.layers.<i>.attention.bias` (`[1, 1, 2048, 2048]`, uint8 causal-mask buffer), `gpt_neox.layers.<i>.attention.masked_bias` (`[]`, scalar buffer), `gpt_neox.layers.<i>.attention.rotary_emb.inv_freq` (`[32]`, rotary frequency buffer).

## Required result

1. Cast exactly the 64 projection matrices listed above to bfloat16
   (round-to-nearest-even, as `tensor.to(torch.bfloat16)` does; the input is float16, so upcast to float32 first or cast directly, both give the same result).
2. Every other tensor is float32 in the output (upcast from float16 where necessary; values are unchanged since float16 is exact in float32). This includes
   `gpt_neox.embed_in.weight`, `embed_out.weight`, all layer-norm weights and biases, and all projection biases (these are float16 in the input and must be upcast to float32).
3. Delete the 48 buffers `gpt_neox.layers.<i>.attention.bias`, `gpt_neox.layers.<i>.attention.masked_bias`, `gpt_neox.layers.<i>.attention.rotary_emb.inv_freq`. Do not delete any parameter.
4. Tensor names do not change.
5. Output: a sharded safetensors checkpoint in the directory `out/T3/`:
   - shard files plus an index file `model.safetensors.index.json` whose
     `weight_map` maps every tensor name to the shard file that holds it;
   - the tensors in one shard total at most 256 MiB (268,435,456 bytes) of tensor data, not
     counting file headers. A single tensor larger than that (here
     `gpt_neox.embed_in.weight` and `embed_out.weight`, 206 MB each) is stored alone in its own shard.
   - Expected total: 196 tensors.

## Required checks

Your solution must fail loudly if any of these does not hold before writing:

- exactly 64 tensors are bfloat16;
- `gpt_neox.layers.0.attention.query_key_value.weight` is bfloat16;
- `gpt_neox.embed_in.weight` is float32;
- the output has exactly 196 tensors.

## Grading

`grade.py T3 --target pythia-1b` compares `out/T3` with a hidden reference:
sharding rules, exact key set, shapes, dtypes and bit-exact values.
