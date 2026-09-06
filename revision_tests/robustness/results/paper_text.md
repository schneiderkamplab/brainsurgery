# Suggested paper language

## Method

We evaluated failure behavior with 19 deterministic cases executed through the
public CLI. The cases covered plan parsing and transform validation, tensor
selection cardinality, references, missing and damaged inputs, assertions,
destination preservation, save failure, process interruption, and a successful
control. An independent auditor hashed all source and destination files and
loaded safetensors directly. For publication faults, an evaluation-only wrapper
raised or paused immediately after one complete shard write; the interruption
case then received `SIGTERM`. The wrapper did not modify BrainSurgery or affect
the normal CLI cases. We distinguished correct detection by the evaluation
harness from safe application behavior.

## Result

All 18 expected non-success outcomes were observed, all 17 applicable
diagnostic patterns matched, and all 19 source inputs remained byte-identical.
For the 15 failures occurring before publication, 14 exposed no destination
and one preserved a pre-existing file exactly. Mid-save behavior was not
atomic: an injected exception and process interruption each left one visible
shard without an index. When the destination already contained a valid sharded
checkpoint, the injected exception retained its old index and shard but added
an unindexed new shard. Thus, 16/19 cases met the operational destination-safety
criterion; the remaining three are explicit negative findings.

## Claim

These results demonstrate source preservation, relevant diagnostics, and safe
failure before publication for the enumerated malformed-plan, invalid-operation,
input-integrity, and assertion cases. They do not establish transactional or
crash-safe checkpoint publication.

## Limitation and operational guidance

BrainSurgery currently writes output shards directly into the final directory
and writes the index after the shards. A save exception, termination, power
loss, or storage failure can therefore expose a partial or mixed directory.
Until staging and atomic publication are implemented, outputs should be written
to a fresh destination and accepted only after a successful exit and independent
index/loading validation. The reported filesystem behavior was observed on
macOS and should be confirmed on Linux; reproducible out-of-space testing
requires a bounded disposable filesystem and was not included.
