# Robustness and failure semantics

## Questions under test

- Does an invalid plan fail before publishing an output?
- Can a failed or interrupted save leave visible partial output?
- Is a pre-existing destination preserved?
- Are errors specific enough to diagnose the failing plan step?

## Planned cases

- invalid YAML and invalid top-level structure;
- unknown transformation and invalid arguments;
- invalid regex, zero matches, and unintended multiple matches;
- missing aliases, files, tensors, and shards;
- failed assertions;
- corrupt or truncated safetensors and shard indexes;
- injected save exception and process interruption;
- pre-existing destination and insufficient disk space.

## Recorded outcome

For every case record the exit code, error class, relevant diagnostic, source
hash, destination visibility and loadability, changes to pre-existing output,
and leftover temporary or partial files. The initial study characterizes the
current behavior; it does not silently change publication semantics after the
usability-study freeze.
