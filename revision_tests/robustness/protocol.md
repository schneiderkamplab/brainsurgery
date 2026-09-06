# Robustness and failure-semantics protocol

Protocol identifier: `eacl2027_robustness_v1`

Status: frozen before the first reported run

## Scope

This evaluation covers BrainSurgery's public checkpoint-transformation CLI,
using the in-memory provider and safetensors inputs and outputs. Axon, Synapse,
model-quality effects, and performance are out of scope.

The study asks whether malformed plans, invalid transformations, missing or
damaged inputs, failed assertions, save exceptions, and process interruption:

1. produce a non-zero process outcome;
2. provide a case-relevant diagnostic when a Python exception can be emitted;
3. leave every source input byte-identical;
4. avoid publishing a new output for failures before save;
5. preserve a pre-existing destination for a failure before save; and
6. leave visible, loadable, or partial output artifacts during publication
   failure or interruption.

The case matrix and diagnostic patterns are serialized in `cases.yaml`. A case
passes the *evaluation* only when its process outcome, diagnostic expectation,
artifact state, source immutability, and expected safety classification all
match. This is distinct from whether the observed system behavior is safe.

## Fixtures and independence

The runner constructs deterministic safetensors fixtures from literal values
and arithmetic sequences. It also independently constructs corrupt, truncated,
and missing-shard inputs. BrainSurgery is not used to create these inputs.

Normal cases invoke the installed `brainsurgery` executable through its public
CLI in fresh subprocesses. The save-exception and interruption cases invoke
`fault_injector.py`, which monkeypatches only the private shard-write call in
that subprocess. It either raises after one complete shard or blocks after one
complete shard until the parent sends `SIGTERM`. The wrapper does not alter
repository source or the behavior of any normal case.

The artifact auditor is independent of BrainSurgery. It hashes all visible
files, parses a shard index with the Python JSON library, and loads safetensors
directly. Its negative controls must distinguish an absent destination, a
valid complete destination, a pre-existing unchanged destination, and a
partial directory without an index.

## Safety definition

For this protocol, `observed_safe = true` requires source-byte preservation and
one of the following destination states:

- no destination exists;
- a pre-existing destination remains byte-identical; or
- a complete valid output is expected from a successful control.

A newly visible incomplete output is classified as unsafe even if every file
that is present is individually valid. This operational definition concerns
publication atomicity, not arbitrary filesystem or adversarial guarantees.

The three injected mid-save cases are frozen to expect an unsafe partial or
mixed output under the current implementation. One starts with a valid
pre-existing sharded destination. Their evaluation should pass only if those
negative findings are detected. A future atomic-save implementation should
change the protocol version and expectation rather than silently changing the
reported v1 result.

## Reproducibility record

Every run records the protocol and case-file checksums, repository commit and
relevant-path status, exact command, machine and package versions, fixture
manifests, generated-plan checksums, CLI return code or signal, elapsed time,
diagnostic match, source hashes, destination hashes and loadability, and any
leftover files.

Raw artifacts live under `log/revision_tests/<run_id>/robustness/`. Compact
summaries may be committed below `revision_tests/robustness/results/`. Run
directories are immutable; reruns require new identifiers.

## Reporting and claim boundary

Report counts for evaluated cases, expected failures detected, source inputs
unchanged, diagnostics matched, failures without a newly exposed output,
pre-existing destinations preserved, and unsafe partial-output findings. Name
the failure classes represented by the cases. Do not claim transactional or
crash-safe publication unless every relevant publication case is observed
safe.

The macOS run is suitable for semantic and filesystem-behavior evidence on the
tested local filesystem. Repeat the protocol on the Linux/CUDA machine to
check platform-sensitive filesystem behavior; CUDA is not itself required.
True out-of-space behavior is intentionally excluded from the laptop protocol
because safely and reproducibly exhausting a real volume requires a controlled
filesystem or quota. It should be added as a separately versioned Linux case
using a bounded disposable filesystem.
