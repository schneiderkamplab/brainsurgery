---
name: report
description: Generate benchmark run summaries as 4 markdown tables from axon-benchmark logs. Use when asked to "report", "status", or "show tables" for a run directory containing stream CSV/result logs, including completion/ETA, timing counts, generic-vs-materialized quality comparison, and Axon/HF >= 1.0 rows.
---

# Report

Generate four markdown tables from a benchmark log directory:
1. Overview/progress (`completed`, `planned`, `completion`, `elapsed`, `ETA`, health/error counters, timed row count, Axon faster count, Axon/HF >= 1.0 count).
2. Issue rows (errors, top1 mismatch, or high max-abs-diff).
3. Generic vs materialized quality comparison by checkpoint.
4. Rows with `Axon/HF >= 1.0`, sorted by the ratio descending.

Use the repository script:
- `/work/dfm/brainsurgery/scripts/benchmark_report_3tables.py`

## Workflow

Run this command:

```bash
conda run -n brainsurgery python scripts/benchmark_report_3tables.py <log_dir> --label "<label>"
```

Optional flags:
- `--abs-threshold <float>`: defaults to `1e-3`.
- `--max-rows <int>`: cap for table 2 and table 3.

If the run is still active, use the same log dir and rerun this command to refresh status.

## Output Contract

When replying, keep exactly four tables in this order:
1. Progress/overview.
2. Issue rows.
3. Generic-vs-materialized comparison.
4. Axon/HF >= 1.0 rows.

If there are no issue rows, print table 2 with a single `(none)` row.
If there are no Axon/HF >= 1.0 rows, print table 4 with a single `(none)` row.
