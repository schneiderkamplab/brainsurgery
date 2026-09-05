---
paths:
  - "scripts/**"
  - "skills/**"
---

# Scripts and benchmark rules

@../../scripts/AGENTS.md

- Canonical benchmark runner is `brainsurgery synapse axon-benchmark`.
- All run artifacts go under `log/<run-id>/`. Use `--log-dir log/<run-id>` and
  `--stream-csv log/<run-id>/stream.csv`. Never create top-level `log-*` paths.
  Use a fresh run dir per attempt.
- Set `OMP_NUM_THREADS` and `CUDA_VISIBLE_DEVICES` explicitly for GPU runs.
  Match `--processes` to the GPU or pipeline-parallel plan, do not infer it.
- Report with `python scripts/benchmark_report_3tables.py log/<run-id>` and keep
  the standard table layout.
- Scripts: shebang, `set -euo pipefail`, parameterized, idempotent, no
  machine-local paths. Document every maintained script in `wiki/scripts.md`.
- A script change that implies a `brainsurgery/*` behavior change needs
  approval before touching the package. Semantic benchmark changes get a note
  in `wiki/log.md`.
- Before benchmark, report, or roundtrip tasks, read the matching
  `skills/<name>/SKILL.md`.
