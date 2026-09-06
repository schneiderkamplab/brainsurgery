# Robustness and failure semantics

This directory contains the EACL 2027 demo-track robustness evaluation for
BrainSurgery's public checkpoint-transformation CLI. The study characterizes
the current implementation; it does not change runtime or publication
semantics.

The frozen protocol is in `protocol.md`, and the cases are declared in
`cases.yaml`. Run it from the repository root with:

```bash
.venv/bin/python revision_tests/robustness/run.py
```

Raw plans, stdout, stderr, fixtures, and output remnants are written below
`log/revision_tests/<run_id>/robustness/`. Use `--publish-dir` only for a new
compact-results directory after the evaluated source is committed.

Two outcomes are deliberately kept separate:

- **evaluation pass** means the harness elicited and correctly classified the
  frozen expected behavior;
- **observed safe** means the source was unchanged and no new partial output
  was exposed (or a pre-existing destination remained byte-identical).

Consequently, an injected save failure can pass the evaluation while exposing
an unsafe partial-output behavior. Such a result is a finding, not a hidden
test failure.

The injection wrapper in `fault_injector.py` is evaluation instrumentation. It
patches the shard writer only inside its subprocess and is never imported by
the normal CLI cases or by the BrainSurgery package.
