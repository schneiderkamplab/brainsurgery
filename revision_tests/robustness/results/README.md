# Robustness and failure-semantics results

The frozen `eacl2027_robustness_v1` protocol was executed through the public
BrainSurgery CLI on macOS. It covers 19 deterministic cases and keeps harness
success separate from system safety.

## Results

| Scenario | Expected process observed | Source unchanged | Destination safe |
|---|---:|---:|---:|
| Failure before publication | 15/15 | 15/15 | 15/15 |
| Mid-save exception or interruption, fresh destination | 2/2 | 2/2 | 0/2 |
| Mid-save exception, pre-existing sharded destination | 1/1 | 1/1 | 0/1 |
| Successful publication control | 1/1 | 1/1 | 1/1 |
| **Total** | **19/19** | **19/19** | **16/19** |

All 18 expected non-success outcomes were detected, and all 17 cases for which
a Python diagnostic was expected matched a case-relevant diagnostic pattern.
The independent artifact-auditor controls passed.

Failures before publication include malformed and non-mapping YAML, an unknown
transformation, a missing required argument, an invalid regex, zero and
excessive tensor matches, an unknown model alias, a missing input, a failed
assertion, corrupt and truncated safetensors, a missing shard, and an
unavailable output parent. None changed an input. Fourteen left no destination;
the case with a pre-existing destination left it byte-identical.

## Negative publication finding

BrainSurgery does not currently provide atomic sharded publication:

- an injected exception after the first complete shard left that shard visible
  without an index;
- `SIGTERM` after the first complete shard produced the same partial directory;
- an injected exception against a valid pre-existing sharded destination left
  the old index and shard plus an unindexed new shard, changing the directory.

These three cases passed the *evaluation* because the frozen harness detected
the expected unsafe behavior. They are not safety passes. The paper must not
claim rollback, transactional save, or crash-safe publication.

Until publication is made atomic, users should write to a fresh destination,
accept it only after successful process exit and independent index/loading
validation, and avoid treating the mere existence of an output directory as
success.

## Claim supported

> Across the 15 tested failures before publication, BrainSurgery rejected the
> operation, left every input byte-identical, and either exposed no output or
> preserved the pre-existing destination exactly.

This claim is limited to the enumerated cases, in-memory provider,
safetensors, and recorded macOS filesystem. Mid-save failure and interruption
are explicitly excluded from that safety claim.

## Evidence

- [`macos_fd20fa27/`](macos_fd20fa27/paper_table.md): compact record for run
  `2026_09_06_robustness_macos_fd20fa27` at commit
  `fd20fa2726f0cde5d39a2870242c83b26b8f46b9`.
- [`paper_table.tex`](paper_table.tex): compact LaTeX table for the paper.
- [`paper_text.md`](paper_text.md): conservative result and limitation text.

Raw plans, process logs, fixtures, and output remnants remain under the ignored
`log/revision_tests/2026_09_06_robustness_macos_fd20fa27/` directory. Repeat the
filesystem-sensitive publication cases on Linux. A true out-of-space case
requires a bounded disposable filesystem and is not part of this macOS result.
