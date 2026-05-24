---
status: active
last-confirmed: 2026-05-24
owners: agents
confidence: high
---

# Benchmark Workflows

This page records benchmark execution and reporting conventions.

Validated-by: root `AGENTS.md`, `scripts/AGENTS.md`, `scripts/benchmark_report_3tables.py`, and repo inspection on 2026-05-24.

## Canonical Execution

- Use `brainsurgery synapse axon-benchmark`.
- Write new run artifacts under `log/<run-id>`.
- Use `--log-dir log/<run-id>` and `--stream-csv log/<run-id>/stream.csv`.
- Pass `--debug-errors` for investigation runs.
- Set GPU visibility explicitly with `CUDA_VISIBLE_DEVICES=...`.
- Set `OMP_NUM_THREADS=<n>` for large parallel runs.
- Use explicit size filters when requested: `--min-billion-parameters`, `--max-billion-parameters`.

Example:

```bash
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0,1,2,3 \
brainsurgery synapse axon-benchmark \
  brainsurgery/synapse/models \
  --device cuda \
  --processes 4 \
  --max-billion-parameters 4 \
  --axon-backend codegen2-torch \
  --log-dir log/<run-id> \
  --stream-csv log/<run-id>/stream.csv \
  --debug-errors
```

## Reporting

Use:

```bash
python scripts/benchmark_report_3tables.py log/<run-id>
```

The standard report has four tables:

- Progress summary: completed/planned/completion/errors/elapsed/ETA/timed rows/Axon faster/Axon-HF >= 1.0/run-active.
- Issue rows: `ERROR`, `masked_top1_eq != True`, or `masked_max_abs_diff >= threshold`.
- Generic-vs-materialized mismatch rows when both variants exist.
- Rows with `Axon/HF >= 1.0`, sorted by ratio descending.

For chat/status reports, keep output concise and include the run directory.

## Pipeline And Parallelism

- No pipeline: omit pipeline flags and usually match `--processes` to available GPUs.
- Pipeline: pass explicit backend and pipeline parallelism flags, and match `--processes` to the intended GPU/PP layout.
- Do not infer process count from GPU count without checking requested pipeline mode.

## Legacy Artifacts

Older helper scripts still reference top-level `log-*` paths. They are documented in `wiki/scripts.md` as legacy. New runs should not create new top-level benchmark logs.

## Durable Memory Rule

Append `wiki/log.md` only for benchmark outcomes that are durable:

- broad sweep summaries,
- recurring error classes,
- repair/rerun outcomes,
- benchmark infrastructure changes.

Do not append every transient progress update.
