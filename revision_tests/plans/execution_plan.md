# Revision-test execution plan

Last checked: 2026-09-06

This plan is organized by backend requirement, not by physical machine. Run as
much as possible here on macOS, freeze the protocols and inputs, and then move
only the backend-dependent work to Linux/CUDA.

## Phase 0: freeze shared inputs and protocols on macOS

Complete this phase before starting a reported Linux run.

- [ ] Select and record one revision-test Git commit.
- [ ] Confirm that all model names and Hugging Face revisions are pinned.
- [ ] Verify the usability-study data manifest reports `126/126`.
- [ ] Record the SHA-256 of `usability_tests_data.tar`.
- [ ] Freeze correctness fixtures, robustness cases, behavioral prompt
      manifest, baseline operations, metrics, tolerances, and stop conditions.
- [ ] Ensure every runner writes raw output to
      `log/revision_tests/<run_id>/` and records the Git commit and environment.
- [ ] Do not change a frozen protocol during a reported cohort. Fix centrally,
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

- [ ] Create a versioned behavioral prompt manifest with source, license,
      identifier, language, task category, split, and filtering procedure.
- [ ] Freeze tokenizer, prompting, decoding, random seeds, metrics, and decision
      rules.
- [ ] Define tool-neutral overlapping operations for MergeKit and
      `torch-state-bridge`; treat Orbax primarily as related-work positioning
      unless a genuinely equivalent executable case exists.
- [ ] Build and unit-test analysis code against small synthetic data.
- [ ] Validate Linux/CUDA launch commands without claiming local timings as
      final performance results.

### 5. Prepare transfer and handoff

- [ ] Commit the frozen protocols, fixtures, runners, and plans.
- [ ] Confirm the worktree diff contains no checkpoints, archives, credentials,
      environments, caches, or raw logs.
- [ ] Transfer `usability_tests_data.tar` separately from Git.
- [ ] Write the exact Linux setup and verification commands into the applicable
      runner README.

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

- [ ] Record the output under a new `log/revision_tests/<run_id>/` directory.
- [ ] Download base checkpoints at the revisions pinned in
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

- [ ] Install each pinned competing tool in an isolated environment.
- [ ] Run the frozen common operations on identical checkpoint copies.
- [ ] Validate every output with the same independent oracle.
- [ ] Record unsupported operations and dependency failures explicitly.
- [ ] Run all timed comparisons on the same Linux hardware and storage.

### 9. Confirm OS-dependent robustness cases

- [ ] Repeat filesystem and interruption cases whose outcome may differ from
      macOS.
- [ ] Confirm partial-output visibility, rename behavior, permissions, and
      insufficient-disk handling on the target Linux filesystem.
- [ ] Keep these results separate from the macOS results when behavior differs.

## Phase 3: run with Linux and CUDA

These tasks load or execute models and therefore require the CUDA backend.

### 10. Behavioral regression suite

- [ ] Run the frozen, versioned prompt suite on the unmodified reference model.
- [ ] Run it on each transformed model under identical inference settings.
- [ ] Record tensor-level validation separately from behavioral agreement.
- [ ] Report task and language coverage, exclusions, failures, and uncertainty.

### 11. Downstream quality

- [ ] Select only transformations for which there is a defensible downstream
      hypothesis, such as PHLoRA, MoE upcycling, or low-rank rewriting.
- [ ] Compare with the unmodified checkpoint and any appropriate method
      baseline using frozen datasets, templates, seeds, and metrics.
- [ ] Record training or inference stability and resource use.
- [ ] If the evidence cannot be produced within budget, narrow the corresponding
      paper claim instead of substituting a weak proxy.

### 12. Scaling and systems measurements

- [ ] Measure GPT-2 124M, Pythia 1B, OLMo 1B/sharded, and at least one sharded
      7B checkpoint.
- [ ] Compare equivalent Python/PyTorch, BrainSurgery in-memory, and
      BrainSurgery arena operations.
- [ ] Record wall time, peak RSS, peak GPU memory, bytes read/written, effective
      I/O throughput, temporary disk, output bytes, shard count, and validation.
- [ ] Separate checkpoint bytes from parameter count.
- [ ] Use the same hardware, filesystem, cache policy, inputs, and operation for
      every point in a reported comparison.

Small smoke cases may run on any CUDA-capable Linux host. Use the larger GPU
backend for 7B+ execution and downstream evaluation when the smaller GPU cannot
hold the required inference workload. CPU/I/O scaling should still be reported
as CPU/I/O work rather than attributed to the GPU.

## Phase 4: return results and integrate on macOS

- [ ] Copy or merge only completed, audited result summaries.
- [ ] Keep raw artifacts under `log/revision_tests/<run_id>/` or in the external
      archival location; do not add large outputs to Git.
- [ ] Re-run analysis from the imported raw records.
- [ ] Link every paper table to its protocol, command, manifest, and run ID.
- [ ] State the current failure-publication behavior exactly as observed.
- [ ] Limit scale and downstream claims to the largest and strongest completed
      experiments.
- [ ] Update the revision-plan checkboxes and prepare the narrated demo only
      after the corresponding evidence is final.

## Submission gate

- [ ] Every reported result has a commit, command, environment, manifest, and
      raw-record location.
- [ ] All usability failures and reviews have completed manual bookkeeping.
- [ ] Correctness references are independent of the tested BrainSurgery path.
- [ ] Lossless and lossy claims use appropriate comparison rules.
- [ ] macOS and Linux/CUDA performance measurements are not silently mixed.
- [ ] Negative results and unsupported features remain visible.
- [ ] Paper claims do not exceed the completed evidence.
