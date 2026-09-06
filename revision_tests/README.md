# BrainSurgery revision tests

This is the committed home for the additional evidence requested during the
EACL 2027 review. The coding-agent experiment remains in
[`usability_tests/`](../usability_tests/README.md); this directory covers the
other revision work and keeps its plans synchronized across machines through
Git.

## Work areas

| Area | Question answered | Backend |
|---|---|---|
| [`correctness/`](correctness/README.md) | Are transformations correct, and are tensors outside the declared write-set unchanged? | macOS; Linux confirmation only where backend-dependent |
| [`robustness/`](robustness/README.md) | What happens for malformed plans, failed assertions, interrupted saves, and corrupted inputs? | macOS; Linux confirmation for OS-specific failures |
| [`scaling/`](scaling/README.md) | How do time, peak memory, I/O, and sharding change with checkpoint size? | Linux/CUDA for reported measurements |
| [`downstream/`](downstream/README.md) | Do intentionally lossy rewrites preserve useful model behavior or task quality? | Linux/CUDA |
| [`competing_tools/`](competing_tools/README.md) | How does BrainSurgery compare on operations genuinely shared with other tools? | Linux |
| [`behavioral/`](behavioral/README.md) | How is the prompt suite sourced, covered, and evaluated beyond the original 50 prompts? | macOS for design; Linux/CUDA for model execution |

The review priorities and execution order are in [`plans/`](plans/README.md).

## Artifact layout

Committed material belongs under `revision_tests/`:

```text
revision_tests/
  plans/              reviewer map and multi-machine schedule
  correctness/        fixtures, oracles, runners, compact summaries
  robustness/         fault cases, fault injection, compact summaries
  scaling/            benchmark definitions and analysis
  downstream/         evaluation protocols and analysis
  competing_tools/    common-operation definitions and adapters
  behavioral/         prompt manifest and regression protocol
```

Raw output belongs outside this directory:

```text
log/revision_tests/<run_id>/
  command.txt
  environment.json
  manifest.json
  raw/
  summary.json
```

Checkpoints remain under ignored `models/`. Never commit model weights, the
usability data archive, environments, credentials, or provider caches.

## Run identity

Use a unique underscore-separated run identifier, for example:

```text
2026_09_06_correctness_mac_<short_commit>
2026_09_06_scaling_cuda_<short_commit>
```

Before accepting a result, record:

- repository commit and dirty-worktree state;
- exact command and configuration;
- OS, CPU, RAM, filesystem, Python, and PyTorch versions;
- GPU, driver, and CUDA versions when applicable;
- exact Hugging Face model name and revision;
- input-manifest checksum;
- pass/fail status and a link to the raw run directory.

## Order of work

1. Finish the macOS-only design, fixture, correctness, robustness, and audit
   work.
2. Freeze the protocols and prepare a verified transferable data bundle.
3. Run the Linux-only usability and competing-tool work.
4. Run CUDA-dependent inference, downstream, and 7B+ scaling work.
5. Bring back closed, audited results and then update paper claims.

The full gates and commands are in
[`plans/execution_plan.md`](plans/execution_plan.md).
