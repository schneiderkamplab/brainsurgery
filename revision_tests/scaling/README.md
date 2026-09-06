# Scaling and systems measurements

## Matrix

Measure the same operation and storage layout on progressively larger models:

1. GPT-2 124M;
2. Pythia 1B;
3. OLMo 1B, including sharded storage;
4. at least one 7B sharded checkpoint.

Compare Python/PyTorch, BrainSurgery in-memory execution, and BrainSurgery arena
execution where the semantics are genuinely equivalent.

## Metrics

- wall-clock time;
- peak resident CPU memory;
- peak GPU memory when a GPU path is used;
- bytes read and written and effective throughput;
- temporary disk usage;
- output size and shard count;
- validation result and output hash/manifest.

## Comparability gate

Use identical inputs, storage, warm/cold-cache policy, operation, and validation
on one machine for each comparison. Report checkpoint bytes separately from
parameter count, and do not place Mac and Linux timings on the same performance
curve unless hardware is explicitly modeled.
