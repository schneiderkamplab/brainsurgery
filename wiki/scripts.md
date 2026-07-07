---
status: active
last-confirmed: 2026-05-20
owners: agents
confidence: high
---

# Scripts Inventory

This page is the maintained inventory for `../scripts/`.

Validated-by: repo inspection of `scripts/` on 2026-05-20.

## Current Standards

- Preferred env: `conda run --no-capture-output -n brainsurgery ...` or an activated `brainsurgery` conda env.
- New benchmark artifacts must go under `log/<run-id>`, matching `../scripts/AGENTS.md`.
- Older scripts that write top-level `log-*` paths are legacy convenience helpers; do not copy that output layout for new work.
- Roundtrip scripts default to writing under `tmp/axon-stage-roundtrip-*`.

## Roundtrip Scripts

These scripts validate staged Axon pipeline render/reparse stability.

| Script | Purpose | Default Output |
|---|---|---|
| `scripts/axon_parse_roundtrip.py` | Parse/render/parse stability for raw Axon ASTs. | `tmp/axon-stage-roundtrip-parse` |
| `scripts/axon_resolve_roundtrip.py` | Load+resolve render stability across three generations. | `tmp/axon-stage-roundtrip-resolve` |
| `scripts/axon_normalize_weak_roundtrip.py` | Reparse and renormalize without reresolve. | `tmp/axon-stage-roundtrip-normalize-weak` |
| `scripts/axon_normalize_strong_roundtrip.py` | Reparse, reresolve, and renormalize. | `tmp/axon-stage-roundtrip-normalize-strong` |
| `scripts/axon_normalize_roundtrip.py` | Compatibility entrypoint to strong normalize roundtrip. | same as strong |
| `scripts/axon_elaborate_weak_roundtrip.py` | Reparse, renormalize, and reelaborate without reresolve. | `tmp/axon-stage-roundtrip-elaborate-weak` |
| `scripts/axon_elaborate_strong_roundtrip.py` | Rerun resolve+normalize+elaborate. | `tmp/axon-stage-roundtrip-elaborate-strong` |
| `scripts/axon_flatten_weak_roundtrip.py` | Reparse, renormalize, and reflatten without reresolve. | `tmp/axon-stage-roundtrip-flatten-weak` |
| `scripts/axon_flatten_strong_roundtrip.py` | Rerun resolve+normalize+elaborate+flatten. | `tmp/axon-stage-roundtrip-flatten-strong` |
| `scripts/axon_flatten_roundtrip.py` | Compatibility entrypoint to strong flatten roundtrip. | same as strong |
| `scripts/axon_typecheck2_weak_roundtrip.py` | Reparse, renormalize, and retypecheck without reresolve/reflatten. | `tmp/axon-stage-roundtrip-typecheck2-weak` |
| `scripts/axon_typecheck2_strong_roundtrip.py` | Rerun resolve+normalize+elaborate+flatten+typecheck2. | `tmp/axon-stage-roundtrip-typecheck2-strong` |
| `scripts/axon_typecheck_roundtrip.py` | Typecheck roundtrip wrapper with selectable typechecker flag. | `tmp/axon-stage-roundtrip-typecheck` |
| `scripts/axon_graph_ir_weak_roundtrip.py` | Full pipeline to graph-rendered Axon, then weak graph rerender. | `tmp/axon-stage-roundtrip-graph-ir-weak` |
| `scripts/axon_graph_ir_strong_roundtrip.py` | Rerun resolve+normalize+elaborate+flatten+typecheck2+graph render each generation. | `tmp/axon-stage-roundtrip-graph-ir-strong` |
| `scripts/axon_graph_optimize_weak_roundtrip.py` | Graph IR weak roundtrip with graph optimization enabled. | `tmp/axon-stage-roundtrip-graph-optimize-weak` |
| `scripts/axon_graph_optimize_strong_roundtrip.py` | Graph IR strong roundtrip with graph optimization enabled. | `tmp/axon-stage-roundtrip-graph-optimize-strong` |

Common flags: `--output-dir`, `--keep-existing`, `--include-stale-cache`, stage-specific `--no-validate-*`, and for typed/graph stages `--main-module`, `--show-types`, `--show-inferred-expression-types`.

Depends-on: `scripts/axon_roundtrip_common.py` for shared path discovery, generation writing, and result reporting.

## Benchmark Reporting

| Script | Purpose | Notes |
|---|---|---|
| `scripts/benchmark_report_3tables.py` | Render the standard 4 markdown tables from recursive `axon-benchmark` stream CSV and result JSON logs. | Use for `report`/`status` workflows. |
| `scripts/monitor_axon_benchmark.py` | Rich live monitor for `axon-benchmark` run directories. | Reads recursive `stream.csv`, parent logs, and paired-runner `paired-status.csv`/`paired-runner.log`; shows progress, GPU memory/utilization, active jobs, and recent failures. |
| `scripts/merge_benchmark_results.py` | Merge one or more `axon-benchmark` log directories into a latest-row CSV keyed by Axon file plus checkpoint. | Pass log dirs in precedence order; later logs overwrite stale rows from earlier logs. Adds `*_norm128` columns by dividing 1024-token timings by 8. |
| `scripts/plot_axon_speedup_scatter.py` | Render an SVG log-log scatter plot of HF time vs Axon time from recursive `axon-benchmark` result JSON logs or a merged CSV. | Uses task color, model-kind marker, generic-vs-materialized fill/outline, a `y=x` parity line, and labels only outliers. Use `--normalized-128` with merged CSVs to plot `*_norm128` timing columns. |
| `scripts/plot_axon_ratio_distributions.py` | Render SVG box and violin plots of Axon/HF runtime ratios from a merged benchmark CSV. | Groups by task, model kind, and generic/materialized source. Use `--normalized-128` with merged CSVs to plot `speed_ratio_axon_over_hf_norm128`. |

Example:

```bash
conda run --no-capture-output -n brainsurgery \
  python scripts/benchmark_report_3tables.py log/<run-id> --abs-threshold 1e-3
```

Output tables: progress summary, issue rows, generic-vs-materialized mismatch rows, and Axon/HF >= 1.0 rows.

Example live monitor:

```bash
conda run --no-capture-output -n brainsurgery \
  python scripts/monitor_axon_benchmark.py log/<run-id> --refresh 2
```

Use `--no-watch` for a single non-interactive snapshot.

Example merged CSV:

```bash
conda run --no-capture-output -n brainsurgery \
  python scripts/merge_benchmark_results.py \
  log/<base-run> log/<targeted-rerun> \
  --output log/<merged-run>/results.csv
```

Example speedup scatter:

```bash
conda run --no-capture-output -n brainsurgery \
  python scripts/plot_axon_speedup_scatter.py log/<run-id> \
  --output tmp/axon-speedup-scatter.svg
```

Example grouped ratio distributions:

```bash
conda run --no-capture-output -n brainsurgery \
  python scripts/plot_axon_ratio_distributions.py log/<merged-run>/results.csv \
  --box-output log/<merged-run>/axon-hf-boxplot.svg \
  --violin-output log/<merged-run>/axon-hf-violin.svg \
  --normalized-128
```

## Checkpoint/Test Model Generators

| Script | Purpose | Important Inputs | Outputs |
|---|---|---|---|
| `scripts/create_deepseek_v4_random.py` | Create deterministic DeepSeek V4 test checkpoints for Axon/HF parity tests. | `--output`, `--variant`, `--seed`, `--dtype`, `--max-shard-size`, `--tokenizer`. | HF-compatible checkpoint directory with config, tokenizer files, safetensors, README, and summary JSON. |
| `scripts/create_min4_family_test_models.py` | Create small random test checkpoints for generic Axon families whose real checkpoints are all above 4B parameters. | `--repo-root`, `--output-root`, repeated `--only`, `--seed`, `--dtype`, `--max-shard-size`. | Tiny HF-compatible checkpoints under `models/test/` by default. |

Relationship: `create_min4_family_test_models.py` uses DeepSeek V4 test-model creation logic so the min4B test inventory can be regenerated through one command.

## Materialization

| Script | Purpose | Notes |
|---|---|---|
| `scripts/rematerialize_all_generic.sh` | Bulk materialize every `generic-*.axon` under `brainsurgery/synapse/models`. | Uses `brainsurgery synapse axon-materialize`; writes summaries under `log/rematerialize/`. |

Example:

```bash
PARALLEL=8 MODELS_ROOT=models scripts/rematerialize_all_generic.sh
```

## Targeted Benchmark Helpers

These are narrow helpers for previously investigated benchmark clusters.

| Script | Purpose | Status |
|---|---|---|
| `scripts/run_gemma4_affected_benchmark.sh` | Shell wrapper for the Gemma4 affected benchmark helper. | Active only for that targeted rerun. |
| `scripts/run_gemma4_affected_benchmark_py.py` | Runs selected Gemma4/Gemma4-MoE affected rows through `run_axon_benchmark`. | Uses a default top-level `log-gemma4-*` path; prefer overriding to `log/<run-id>`. |
| `scripts/run_gemma4_moe_g45.py` | Targeted Gemma4 MoE derived-experts benchmark. | Legacy one-off; top-level log path. |
| `scripts/run_rope_freqscale_max10b.py` | Targeted RoPE/frequency-scaling benchmark up to 10B. | Legacy one-off; top-level log path. |

## Legacy Launch/Render Helpers

These helpers predate the current `log/<run-id>` convention.

| Script | Purpose | Current Guidance |
|---|---|---|
| `scripts/launch-max20b.sh` | Launch max-20B benchmark with 6 visible GPUs. | Legacy; writes `log-max20b`. Prefer direct `axon-benchmark` with `log/<run-id>`. |
| `scripts/launch-min20b.sh` | Launch min-20B pipeline benchmark. | Legacy; writes `log-min20b-pp4`. Prefer direct `axon-benchmark` with `log/<run-id>`. |
| `scripts/render-max20b.sh` | Render HTML from `log-max20b/stream.csv`. | Legacy companion to `launch-max20b.sh`. |
| `scripts/render-min20b.sh` | Render HTML from `log-min20b-pp4/stream.csv`. | Legacy companion to `launch-min20b.sh`. |

## Maintenance Rule

When a script is created, removed, or semantically changed:

- Update this page in the same change.
- If benchmark behavior changes, append a dated note to `wiki/log.md`.
- If the change alters canonical execution policy, update `scripts/AGENTS.md` or root `AGENTS.md`.
