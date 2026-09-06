# EACL 2027 revision plan

Deadline: 2026-09-22

Operational order: [`execution_plan.md`](execution_plan.md)

This document maps the reviews to possible evidence. It is not a promise to
implement every suggestion. The revision should prioritize a small number of
strong, reproducible evaluations and narrow claims that cannot be supported
within the available time.

## Reviewer concerns

| # | Concern | Detail | Raised by |
|---:|---|---|---|
| 1 | Evaluation is too narrow to establish practical utility | Current evidence is dominated by semantic equivalence and compactness/readability comparisons. This is necessary but does not establish whether BrainSurgery is practically better than scripting. | nHbn, Tuk7, 4zYf; meta-reviewer |
| 2 | Usability, auditability, accessibility, productivity, and error-reduction claims lack user evidence | There is no user study, time-to-completion or comprehension experiment, error-rate comparison, practitioner feedback, or adoption evidence. | nHbn, Tuk7, 4zYf; meta-reviewer |
| 3 | Correctness uses non-independent references | BrainSurgery and equivalent PyTorch implementations were both written by the authors and may share the same mistaken interpretation. | nHbn, 4zYf |
| 4 | No downstream NLP/task-quality evaluation | PHLoRA, MoE upcycling, and low-rank rewriting are not evaluated for task accuracy, generation quality, training stability, or deployment outcomes. | Tuk7, 4zYf; meta-reviewer |
| 5 | Scale, efficiency, and systems claims are insufficiently tested | Existing experiments stop near 1–2B parameters despite scaling, sharding, batch, and memory-mapped-arena claims. Wall time, peak memory, I/O, realistic sharding, and 7B+ evidence are missing. | nHbn, 4zYf; meta-reviewer |
| 6 | Comparison with overlapping systems is incomplete | Orbax and `torch-state-bridge` are omitted, and MergeKit's overlapping YAML, slicing, out-of-core, LoRA, MoE, and workflow capabilities are understated. | Tuk7, 4zYf; meta-reviewer |
| 7 | Inference evaluation is small and insufficiently documented | The 50 prompts lack documented source, sampling, task coverage, and language coverage. | 4zYf |
| 8 | Robustness and failure semantics are unclear | Malformed plans, failed assertions, regex collisions, missing matches, corrupt inputs, interruption, rollback, and partial output are not evaluated. | nHbn, 4zYf |
| 9 | Distributed-checkpoint coverage is not demonstrated | There is no evidence for optimizer states, resharding, multi-rank execution, or distributed checkpoint formats. | 4zYf |
| 10 | Demonstration video is hard to follow | The video lacks narration and contextual guidance through a realistic end-to-end workflow. | nHbn; meta-reviewer |
| 11 | Methodological novelty and NLP contribution appear limited | The work is perceived primarily as software engineering, with limited empirical evidence of an NLP-methodological contribution. | nHbn, Tuk7 |

## Priority decision

The revision is organized into four workstreams:

1. coding-agent usability and competing approaches;
2. correctness, preservation, robustness, and failure semantics;
3. a modest but honest systems/size and behavioral evaluation;
4. claims, positioning, reproducibility, and the narrated demo.

Broad distributed-training support, an ambitious RYS-style circuit-duplication
case, and wide downstream evaluation are stretch work. If they cannot be done
convincingly, narrow the corresponding claims rather than adding weak evidence.

## Work items

Priority uses `P0` for submission-critical, `P1` for high value, `P2` for
optional if the stronger evidence is already complete, and `defer` for work
that should be handled through scope/claim changes in this revision.

| # | Area | Proposed response | Priority | Evidence location |
|---:|---|---|---|---|
| 1 | Coding-agent evaluation | Give agents equivalent checkpoint-editing tasks under Python/PyTorch, allowed-package, and BrainSurgery conditions; compare success, retries, failed executions, tokens/cost, time, and defect detection. | P0 | `usability_tests/` |
| 2 | Correctness | Add hand-verifiable fixtures, independent oracles, write-set hashing, and metamorphic tests. | P0 | `revision_tests/correctness/` |
| 3 | Robustness | Test malformed plans, bad regexes and matches, failed assertions, corrupt inputs, interrupted execution, and save failures. | P0 | `revision_tests/robustness/` |
| 4 | Failure semantics | Define and report output-publication behavior, including whether partial output can remain after failure. | P0 | `revision_tests/robustness/` |
| 5 | Claims and positioning | Frame BrainSurgery as checkpoint-editing systems/tooling; replace universal “easier” or “efficient” claims with claims directly supported by results. | P0 | `revision_tests/plans/claim_boundaries.md` and final evidence map |
| 6 | Reproducibility | Preserve protocols, commands, manifests, machine fingerprints, compact summaries, and raw-result locations. | P0 | this directory and `log/revision_tests/` |
| 7 | Competing tools | Benchmark a small number of genuinely shared operations against MergeKit and `torch-state-bridge`; improve Orbax positioning and distinguish direct comparisons from adjacent feature coverage. | P1 | `revision_tests/competing_tools/` |
| 8 | Scaling | Add a controlled four-point Pythia curve through 12B plus paired GPT-2, OLMo, and Qwen2.5 architecture/storage checks, with time, memory, I/O, dtype, and sharding measurements. Protocol and harness are frozen/Mac-preflighted; real measurements remain Linux work. | P1 | `revision_tests/scaling/` |
| 9 | Behavioral evaluation | Replace the undocumented prompt set with a versioned, sourced, categorized manifest and stated evaluation procedure. | P1 | `revision_tests/behavioral/` |
| 10 | Demo video | Create a narrated valid-plan → intentional failure → diagnosis → correction → validation/diff → export walkthrough. | P1 | demo script/storyboard, then submission asset |
| 11 | Downstream quality | Evaluate one defensible intentionally lossy transformation against an unchanged baseline if compute and methodology permit. | P2 | `revision_tests/downstream/` |
| 12 | RYS/circuit duplication | Consider a realistic block-duplication and reindexing case only if it can be completed and evaluated without displacing P0/P1 evidence. | P2 | future case-study protocol |
| 13 | Additional use cases | Add none merely for breadth; choose only cases answering a specific reviewer concern. | defer | claim narrowing |
| 14 | Distributed support | No distributed-format/resharding experiment is planned for this revision. Explicitly limit evaluation to single-process single-file/indexed-shard safetensors and exclude rank-local state, optimizer state, multi-rank execution, and resharding claims. | defer | `revision_tests/plans/claim_boundaries.md` |
| 15 | Community adoption | Do not imply adoption from the coding-agent study; report adoption evidence only if independently available. | defer | claims |

## Correctness claim

Avoid claiming that BrainSurgery never loses information. The testable claim is:

> For transformations defined as lossless, BrainSurgery changes only the
> declared write-set, preserves all other tensors exactly, and produces changed
> tensors matching an independent oracle.

For dtype conversion, low-rank approximation, pruning, merging, and other lossy
operations, state the intended information change and use explicit numerical or
downstream-quality criteria.

## Usability interpretation

Do not pre-register the universal hypothesis that BrainSurgery is easier than
Python. The five tasks plausibly favor different interfaces:

| Task type | Expected trade-off |
|---|---|
| Bulk layer deletion or renaming | BrainSurgery may become shorter after its targeting syntax is learned; Python has lower initial familiarity cost. |
| Architecture-specific head slicing | Python is likely easier with the current syntax. |
| Bulk dtype conversion and sharding | BrainSurgery should benefit from explicit declarative export. |
| Multi-checkpoint arithmetic with validation | BrainSurgery may benefit from aliases and assertions after onboarding. |
| LoRA merge with unusual layouts | BrainSurgery may be shorter, while Python may be easier to debug. |

Agents have extensive Python/PyTorch priors but essentially no BrainSurgery
prior. The learning cost is part of the result. BrainSurgery's own development
with supervised agents does not remove that onboarding cost for isolated study
participants.

A defensible hypothesis is:

> BrainSurgery trades initial learning and debugging cost for concise, explicit,
> reproducible, and machine-checkable checkpoint transformations, especially
> for bulk targeting, validation, arithmetic, and export.

Useful outcomes include final artifact size, silent-error rate, defect-detection
accuracy, reproducibility, and performance by transformation class. A result in
which Python is faster but BrainSurgery produces fewer silent errors or is
reviewed more accurately is still informative.

## Completion checklist

- [ ] The usability study has complete repeats and manual bookkeeping.
- [x] Correctness uses independent, hand-verifiable evidence. See
      `revision_tests/correctness/results/`.
- [x] Robustness and current failure-publication semantics are demonstrated.
      See `revision_tests/robustness/results/`; failures before publication are
      safe in all 15 enumerated cases, while three mid-save cases expose
      partial or mixed output and rule out an atomic-publication claim.
- [ ] Scaling claims match the largest tested checkpoint and controlled metrics.
- [x] Behavioral prompts have documented provenance and coverage. See
      `revision_tests/behavioral/`.
- [ ] Baseline comparisons use genuinely equivalent operations and one oracle.
- [x] Direct comparisons and adjacent related-system capabilities are separated
      in Markdown and LaTeX. See
      `revision_tests/competing_tools/feature_coverage.{md,tex}`.
- [ ] Every table is reproducible from a named commit, command, and run ID.
- [x] Distributed claim boundaries and safe manuscript wording are prepared.
      Applying them to the final paper remains part of manuscript integration.
- [ ] The narrated demo follows the final documented behavior.
- [ ] The abstract, introduction, conclusion, and limitations make no claim
      stronger than the completed evidence.
