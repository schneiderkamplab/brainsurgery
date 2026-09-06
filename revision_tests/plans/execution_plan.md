# Revision-test execution plan

Last checked: 2026-09-06

This plan is organized by backend requirement, not by physical machine. Run as
much as possible here on macOS, freeze the protocols and inputs, and then move
only the backend-dependent work to Linux/CUDA.

## Phase 0: freeze shared inputs and protocols on macOS

Complete this phase before starting a reported Linux run.

- [x] Select and record one revision-test Git commit.
- [x] Confirm that all model names and Hugging Face revisions are pinned.
- [x] Verify the usability-study data manifest reports `126/126`.
- [x] Record the SHA-256 of `usability_tests_data.tar`.
- [x] Freeze correctness fixtures, robustness cases, behavioral prompt
      manifest, baseline operations, metrics, tolerances, and stop conditions.
- [x] Ensure every runner writes raw output to
      `log/revision_tests/<run_id>/` and records the Git commit and environment.
- [x] Do not change a frozen protocol during a reported cohort. Fix centrally,
      choose a new commit, and restart the affected cohort if necessary.

## Phase 1: run here on macOS

These tasks do not require Linux or CUDA and should be completed locally first.

### 1. Existing usability-result audit

- [ ] Confirm every completed Claude cell has `run.json`, `harness.json`,
      `grade.json`, and `review.json`.
- [ ] Re-summarize saved transcripts where available.
- [ ] Fill the 23 currently blank failed-execution error classes.
- [ ] Manually confirm the 405 `review.json.detected` values.
- [ ] Produce a compact repeat-1 analysis snapshot and provenance note.
- [ ] Keep the two Astra macOS pilot cells excluded from official results.

This is bookkeeping and analysis. It does not require rerunning participant
agents or loading models on a GPU.

### 2. Correctness and no-unintended-change tests

- [x] Implement small, hand-verifiable fixtures for identity, rename and
      inverse rename, copy, move, delete, split/concatenate, dtype conversion,
      arithmetic, and shard/save/reload.
- [x] Declare the expected write-set for every case.
- [x] Hash every tensor outside the write-set before and after execution.
- [x] Compare changed tensors with an independent PyTorch oracle.
- [x] Use exact comparisons for lossless operations and declared rules for
      lossy arithmetic or dtype changes.
- [x] Check container metadata, tensor names, shapes, dtypes, source files, and
      shard indexes in addition to tensor values.
- [x] Confirm the same exact-preservation property across the pinned GPT-2,
      OLMo 1B sharded, and Pythia 1B base checkpoints.
- [x] Save compact summaries in Git and raw results under `log/revision_tests/`.

Evidence: `revision_tests/correctness/results/`. The primary tensor endpoints
all pass; custom safetensors header metadata is not preserved and is recorded as
a limitation outside the tensor-state claim.

### 3. Robustness and failure-semantics tests

- [x] Exercise invalid YAML and invalid top-level structures.
- [x] Exercise unknown transformations, invalid arguments, invalid regexes,
      zero matches, and unintended multiple matches.
- [x] Exercise missing aliases, files, tensors, and shards.
- [x] Exercise failed assertions and corrupted/truncated safetensors.
- [x] Inject save exceptions and interrupt a save process.
- [x] Test behavior when the destination already exists.
- [x] Record exit code, error class, diagnostic, source hash, output visibility,
      output loadability, changes to existing output, and leftover files.
- [x] Characterize current partial-publication behavior without changing
      BrainSurgery semantics during the frozen usability study.

Evidence: `revision_tests/robustness/results/`. All 19 evaluation cases and all
source-preservation checks pass. Three expected negative findings show that a
mid-save exception or interruption leaves a partial or mixed shard directory;
BrainSurgery therefore does not currently provide atomic publication.

Repeat only genuinely OS-dependent cases later on Linux; the malformed-plan and
semantic failure matrix can be established here.

### 4. Prepare behavioral and baseline protocols

- [x] Create a versioned behavioral prompt manifest with source, license,
      identifier, language, task category, split, and filtering procedure.
- [x] Freeze tokenizer, prompting, decoding, random seeds, metrics, and decision
      rules.
- [x] Define tool-neutral overlapping operations for MergeKit and
      `torch-state-bridge`; treat Orbax primarily as related-work positioning
      unless a genuinely equivalent executable case exists.
- [x] Create a paper-facing feature-coverage matrix that distinguishes direct
      tool comparisons from correctness, usability, systems, adjacent, and
      deferred evidence.
- [x] Build and unit-test analysis code against small synthetic data.
- [x] Validate Linux/CUDA launch commands without claiming local timings as
      final performance results.
- [x] Freeze the scaling operation, ten pinned model revisions, three
      operation-matched methods, independent exact oracle, resource metrics,
      cache policy, repetition schedule, and reporting gates.

Behavioral evidence: `revision_tests/behavioral/`. The committed 70-prompt
manifest contains 30 parallel Belebele prompts, 30 stratified MMLU prompts, and
10 HumanEval regression prompts. Its validator reports 70/70 and the analyzer's
five synthetic controls pass. A one-prompt GPT-2 CPU smoke run exercised both
model roles and the analyzer; it is stored only under `log/revision_tests/` and
is explicitly non-reportable. The full model comparison remains a Linux/CUDA
task in Phase 3.

Competing-tool protocol: `revision_tests/competing_tools/`. The frozen cases
compare regex key rewriting with `torch-state-bridge` and two checkpoint
arithmetic operations with MergeKit using an independent oracle. Actual-package
macOS preflights pass all six pairings on tiny and pinned GPT-2-derived inputs;
the runner labels those timings non-reportable. The controlled Linux run in
Phase 2 remains required for paper performance evidence.

Feature-coverage evidence:
`revision_tests/competing_tools/feature_coverage.{md,tex}`. It freezes the
reportable direct-comparison count at three operations (two MergeKit and one
`torch-state-bridge`) while keeping adjacent and deferred capabilities visible.

Scaling protocol: `revision_tests/scaling/`. Its synthetic Mac preflight
exercises the direct PyTorch baseline, BrainSurgery in-memory and arena paths,
sharded serialization, monitoring, and independent oracle. It suppresses Mac
performance values. All ten real checkpoint points and their systems
measurements remain a Linux run below.

### 5. Prepare transfer and handoff

- [x] Commit the frozen protocols, fixtures, runners, and plans.
- [x] Confirm the worktree diff contains no checkpoints, archives, credentials,
      environments, caches, or raw logs.
- [ ] Transfer `usability_tests_data.tar` separately from Git.
- [x] Write the exact Linux setup and verification commands into the applicable
      runner README.

Handoff evidence: `linux_handoff_manifest.json` records the 47 GB archive's
size and SHA-256, its archive/path audit, the pinned base-model revisions, and
the successful 126/126 local verification. `linux_handoff.md` contains the
ordered Linux commands, the first-cell transcript gate, and the official Codex
cohort namespace. Physical transfer remains unchecked until the archive is
copied to and verified on the receiving machine.

## Phase 2: run on Linux

These tasks require Linux because of the frozen study environment, participant
runner, package compatibility, or the need for comparable systems measurements.
CUDA is not necessarily required for every item in this phase.

### 6. Verify the Linux environment

```bash
git fetch origin
git checkout <REVISION_TEST_COMMIT>
git status --short --branch
free -h
df -h .
python3 --version
uv --version
codex --version
nvidia-smi
```

Before downloading models or starting a reported run, execute the frozen
protocol checks from the repository root:

```bash
test -z "$(git status --porcelain)"
.venv/bin/python -m pytest -q \
  revision_tests/scaling/test_scaling.py \
  revision_tests/competing_tools/test_competing_tools.py
.venv/bin/python revision_tests/scaling/validate_protocol.py
.venv/bin/python revision_tests/competing_tools/validate_protocol.py
```

All tests and both validators must pass. The checkout must remain clean until
the reported run is closed.

- [x] Record the output under a new `log/revision_tests/<run_id>/` directory.
- [x] Download base checkpoints at the revisions pinned in
      `usability_tests/targets.py`.
- [ ] Extract the transferred data bundle into `models/`.
- [ ] Run `usability_tests/setup.py` and
      `usability_tests/make_docpack.py`.
- [ ] Stop unless `usability_tests/make_manifest.py --verify` reports
      `126/126`.

### 7. Run the official Codex usability cohort

- [ ] Install condition F in a disposable environment as a preflight.
- [ ] Run one Codex cell with the frozen Codex runner.
- [ ] Compare its first `harness.json` field by field against the transcript.
- [ ] Continue only if the first-cell audit passes.
- [ ] Run all of repeat 1, then audit and complete its manual bookkeeping.
- [ ] Run repeat 2 only after repeat 1 is closed and audited.
- [ ] Do not pull or edit code during a repeat.

Use `usability_tests/run_full_codex.sh` and the exact pricing and model settings
chosen at freeze time. The official run is Linux-only; CUDA availability mainly
helps package compatibility and is not itself the variable being measured.

### 8. Run competing-tool comparisons

- [x] Install each pinned competing tool in an isolated environment.
- [x] Run the frozen common operations on identical checkpoint copies.
- [x] Validate every output with the same independent oracle.
- [x] Record unsupported operations and dependency failures explicitly.
- [x] Run all timed comparisons on the same Linux hardware and storage.

### 9. Confirm OS-dependent robustness cases

- [x] Repeat the frozen 19-case protocol with a unique Linux run ID:

      ```bash
      REVISION_COMMIT_SHORT="$(git rev-parse --short HEAD)"
      .venv/bin/python revision_tests/robustness/run.py \
        --run-id "eacl2027_robustness_linux_${REVISION_COMMIT_SHORT}"
      ```
- [x] Confirm partial-output visibility, rename behavior, permissions, and
      interruption behavior on the target Linux filesystem.
- [x] Keep these results separate from the macOS results when behavior differs.
- [ ] Treat true insufficient-disk behavior as optional: test it only in a
      bounded disposable filesystem/quota under a separately versioned
      protocol. Do not fill or risk the host filesystem.

## Phase 3: run CUDA inference and the remaining large-model systems work

Behavioral and downstream inference require the CUDA backend. The scaling
experiment runs on the same Linux host for operational convenience and
hardware consistency, but it is explicitly a CPU/I/O workload with CUDA
disabled.

### 10. Behavioral regression suite

- [x] Run `revision_tests/behavioral/run_cuda.sh` on the pinned GPT-2 reference
      and frozen byte-exact, sharded lossless transformation.
- [x] Confirm the independent pre-inference tensor gate reports 160/160 exact.
- [x] Run the frozen, versioned prompt suite on the unmodified reference model.
- [x] Run it on each transformed model under identical inference settings.
- [x] Record tensor-level validation separately from behavioral agreement.
- [x] Report task and language coverage, exclusions, failures, and uncertainty.

Evidence: the primary GPT-2 CUDA case passed 160/160 tensors and 70/70 prompt
comparisons. The supplementary ten-checkpoint matrix passed 3,243/3,243 tensors
and 700/700 prompt comparisons; see
`revision_tests/behavioral/results/linux_99693f2/`.

### 11. Downstream quality

- [ ] Select only transformations for which there is a defensible downstream
      hypothesis, such as PHLoRA, MoE upcycling, or low-rank rewriting.
- [ ] Compare with the unmodified checkpoint and any appropriate method
      baseline using frozen datasets, templates, seeds, and metrics.
- [ ] Record training or inference stability and resource use.
- [ ] If the evidence cannot be produced within budget, narrow the corresponding
      paper claim instead of substituting a weak proxy.

### 12. Scaling and systems measurements

- [x] Run the frozen four-point Pythia scaling curve plus two-point GPT-2,
      OLMo, and Qwen2.5 architecture/storage checks from
      `revision_tests/scaling/cases.yaml` (ten checkpoints total).
- [x] Run the equivalent direct Python/PyTorch, BrainSurgery in-memory, and
      BrainSurgery arena operation with `revision_tests/scaling/run.py`.
- [x] Confirm the runner records wall time, peak process-tree RSS, process I/O,
      effective logical throughput, temporary arena disk, output bytes/shards,
      and exact independent validation. GPU memory is intentionally N/A because
      this checkpoint rewrite is forced to CPU; do not present it as GPU work.
- [x] Confirm checkpoint bytes and measured parameter counts are reported
      separately, with checkpoint bytes as the primary systems-size axis.
- [x] Fit or connect the primary curve only within Pythia; show the GPT-2,
      OLMo, and Qwen2.5 pairs separately rather than pooling architectures.
- [x] Use the same hardware, filesystem, warm-cache policy, inputs, operation,
      and one-worker setting for every point in the reported comparison.
- [ ] Require all automated reportability gates plus human artifact/anonymity
      review before copying compact summaries into `scaling/results/`.

Small smoke cases may run on any CUDA-capable Linux host. Use the larger GPU
backend for 7B+ execution and downstream evaluation when the smaller GPU cannot
hold the required inference workload. CPU/I/O scaling should still be reported
as CPU/I/O work rather than attributed to the GPU.

## Phase 4: return results and integrate on macOS

- [ ] Copy or merge only completed, audited result summaries.
- [x] Keep raw artifacts under `log/revision_tests/<run_id>/` or in the external
      archival location; do not add large outputs to Git.
- [x] Re-run analysis from the imported raw records.
- [x] Link every paper table to its protocol, command, manifest, and run ID.
- [x] State the current failure-publication behavior exactly as observed.
- [x] Limit scale and downstream claims to the largest and strongest completed
      experiments.
- [ ] Update the revision-plan checkboxes and prepare the narrated demo only
      after the corresponding evidence is final.

## Submission gate

- [x] Every reported result has a commit, command, environment, manifest, and
      raw-record location.
- [ ] All usability failures and reviews have completed manual bookkeeping.
- [x] Correctness references are independent of the tested BrainSurgery path.
- [x] Lossless and lossy claims use appropriate comparison rules.
- [x] macOS and Linux/CUDA performance measurements are not silently mixed.
- [x] Negative results and unsupported features remain visible.
- [ ] Paper claims do not exceed the completed evidence.
