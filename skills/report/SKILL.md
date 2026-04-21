---
name: report
description: Generate benchmark run summaries as 3 markdown tables from axon-benchmark logs. Use when asked to "report", "status", or "show 3 tables" for a run directory containing stream CSV logs, including completion/ETA and generic-vs-materialized quality comparison.
---

# Report

Generate three markdown tables from a benchmark log directory:
1. Overview/progress (`completed`, `planned`, `completion`, `elapsed`, `ETA`, health/error counters).
2. Issue rows (errors, top1 mismatch, or high max-abs-diff).
3. Generic vs materialized quality comparison by checkpoint.

Use the repository script:
- `/work/training/brainsurgery/scripts/benchmark_report_3tables.py`

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

When replying, keep exactly three tables in this order:
1. Progress/overview.
2. Issue rows.
3. Generic-vs-materialized comparison.

If there are no issue rows, print table 2 with a single `(none)` row.
