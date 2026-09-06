# Submission claim boundaries

Use this checklist when integrating revision-test results into the EACL 2027
demo paper, appendix, demo narration, and anonymous repository. It records what
the current protocols can establish and prevents ordinary safetensors sharding
from being presented as distributed checkpoint evaluation.

## Claims supported by completed macOS evidence

- For the enumerated lossless correctness cases, tensors outside the declared
  write-set are preserved exactly and changed tensors match an independent
  oracle.
- The real-checkpoint preservation cases cover the pinned GPT-2, Pythia 1B,
  and sharded OLMo 1B checkpoints.
- The 19 robustness cases characterize malformed plans, selection/reference
  failures, corrupt inputs, missing shards, failed assertions, save failures,
  interruption, and destination handling on the tested macOS filesystem.
- Mid-save failure or interruption can expose partial or mixed sharded output.
  BrainSurgery does not currently provide atomic checkpoint publication.

These claims remain limited to the named protocols, versions, fixtures, and
checkpoints. Do not generalize them to every transformation or storage backend.

## Claims conditional on successful Linux runs

- Runtime, peak RSS, process I/O, logical throughput, temporary arena disk,
  and output-shard counts may be reported only after every scaling reporting
  gate passes on one clean Linux checkout and one filesystem.
- Comparisons with MergeKit and `torch-state-bridge` may cover only R01, M01,
  and M02, with correctness as the primary endpoint. Do not infer an overall
  tool ranking.
- Usability, completion, error, token, cost, or time claims require the closed
  and manually audited official cohorts. Prepared tasks or pilots are not
  evidence for those claims.
- Behavioral claims require the frozen CUDA inference runs on both the
  reference and transformed checkpoints under identical settings.

## Claims excluded from this revision

Unless a new protocol is frozen and executed before manuscript integration,
do not claim empirical support for:

- distributed checkpoint formats or distributed execution;
- multi-rank save/load;
- resharding between world sizes;
- rank-local model or optimizer state;
- optimizer-state transformation or restoration;
- multi-node behavior or fault tolerance;
- GPU acceleration of the scaling rewrite;
- atomic, transactional, crash-safe, or rollback-safe publication;
- universal information preservation, ease of use, efficiency, or superiority;
- MoE construction/upcycling or downstream-quality preservation.

## Drop-in scope wording

The following wording is safe before the Linux results are known:

> We evaluate single-process transformations of single-file and indexed,
> sharded safetensors checkpoints. Ordinary safetensors sharding should not be
> confused with distributed checkpointing: this evaluation does not cover
> rank-local or optimizer state, multi-rank execution, distributed checkpoint
> formats, or resharding across world sizes.

For publication behavior:

> The source checkpoint remained unchanged in all evaluated failure cases, but
> interrupted or failed sharded saves could leave partial or mixed destination
> files. BrainSurgery therefore does not currently guarantee atomic output
> publication; callers should publish into a fresh temporary destination and
> promote it only after validation.

After a successful scaling run, replace placeholders only with values copied
from its audited generated table:

> On [machine and storage], the frozen CPU/I/O rewrite was validated on ten
> pinned checkpoints from [smallest checkpoint bytes] through [largest
> checkpoint bytes]. We report end-to-end wall time, peak process-tree RSS,
> process I/O, temporary arena storage, and output sharding. These results do
> not measure GPU, training, inference, or distributed execution performance.

## Final manuscript audit

Search the full submission sources, appendix, captions, and demo script for:

```text
distributed
reshard
rank
optimizer state
atomic
transaction
rollback
lossless
no information loss
scalable
efficient
out-of-core
memory-mapped
faster
easier
superior
```

For each occurrence, attach a named result/protocol or narrow the sentence.
Record the final decision in the evidence map; absence of a result is a scope
limitation, not a negative score for another tool.
