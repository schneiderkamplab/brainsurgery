# scripts/AGENTS.md

Global rules: `../AGENTS.md`
Wiki memory for scripts: `../wiki/AGENTS.md` and `../wiki/scripts.md`

## Scope

- `scripts/` contains operational automation owned by agents.
- Scripts should reduce repetitive benchmark/reporting/debug work.

## Benchmark Execution Conventions

- Canonical runner: `brainsurgery synapse axon-benchmark`.
- All new benchmark/run artifacts go under repo-root `log/`.
  - Use `--log-dir log/<run-id>`.
  - Use `--stream-csv log/<run-id>/stream.csv` when streaming CSV.
  - Do not create new top-level `log-*` files or directories.
- Default run hygiene:
  - set `OMP_NUM_THREADS=<n>`
  - pin GPUs with `CUDA_VISIBLE_DEVICES=...`
  - pass `--log-dir log/<run-id>`, `--stream-csv log/<run-id>/stream.csv`, `--debug-errors` for investigation runs
  - use explicit size filters (`--min-billions-parameters`, `--max-billions-parameters`) as requested
- Parallelism:
  - no pipeline: omit pipeline flags
  - pipeline: pass explicit backend/PP flags and match `--processes` to the intended GPU/PP plan
- Use fresh `log/<run-id>` dirs for reruns; do not mix unrelated attempts in one log dir.

## Reporting Conventions

- Reusable 3-table benchmark status report:
  - `python scripts/benchmark_report_3tables.py log/<run-id>`
- Expected input:
  - `log/<run-id>` contains per-run/per-family `stream.csv` files (recursive scan).
  - Optional `manifest.csv` at `log/<run-id>/manifest.csv` for planned group count.
- Output tables:
  - progress summary (completed/planned/completion/rows/run-active)
  - issue rows (`ERROR`, `masked_top1_eq != True`, or `masked_max_abs_diff >= threshold`)
  - generic vs materialized mismatch rows
- Useful flags:
  - `--abs-threshold` (default `1e-3`)
  - `--max-rows` (default `200`)

## Materialization Conventions

- Reusable generic rematerialization helper:
  - `scripts/rematerialize_all_generic.sh`
- Purpose:
  - Re-materialize all `generic-*.axon` model files under `brainsurgery/synapse/models`.
- Usage:
  - `scripts/rematerialize_all_generic.sh [PARALLEL] [MODELS_ROOT]`
  - defaults: `PARALLEL=8`, `MODELS_ROOT=models`
- Output:
  - writes a summary to `log/rematerialize/` and keeps per-run logs there.
  - script should be preferred over ad-hoc one-off loops for bulk rematerialization.

## Standards

- Use explicit shebang and strict mode where reasonable (`set -euo pipefail`).
- Keep scripts parameterized; avoid hard-coded machine-local assumptions.
- Emit clear stdout/stderr messages for long-running operations.
- Prefer idempotent behavior and stable output paths.

## Change Policy

- Safe refactors and ergonomics updates are allowed.
- Behavioral changes to benchmark semantics require a short note in `wiki/log.md`.
- If script changes imply main package behavior changes, request approval before touching `brainsurgery/*`.
