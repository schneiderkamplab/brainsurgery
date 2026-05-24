---
name: benchmark
description: Run brainsurgery Axon benchmarks correctly. Use when asked to benchmark models, rerun affected rows, smoke-test fidelity, choose codegen2/runtime2/pipeline2 options, monitor active runs, or create log-dir/stream-csv artifacts under repo-root log/.
---

# Benchmark

Use the canonical runner:

```bash
conda run -n brainsurgery brainsurgery synapse axon-benchmark <axon paths...>
```

## Defaults

- Write every new run under `log/<run-id>`.
- Always pass `--log-dir log/<run-id>` and `--stream-csv log/<run-id>/stream.csv`.
- For investigation runs, pass `--debug-errors`.
- Prefer `--dtype float32` unless the user explicitly requests another dtype.
- Prefer `--axon-backend codegen2-torch` and `--axon-typechecker typecheck2`.
- Do not use `--optimize-ast`, `--optimize-graph`, `--canonicalize`, or pipeline flags unless explicitly requested.
- Use `--checkpoint` for targeted reruns instead of editing model files or checkpoint lists.
- Use `CUDA_VISIBLE_DEVICES=...` and `OMP_NUM_THREADS=...` when the user specifies GPUs or parallelism.

## Common Commands

Single smoke row:

```bash
RUN=log/<run-id>
mkdir -p "$RUN"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 \
conda run -n brainsurgery brainsurgery synapse axon-benchmark \
  brainsurgery/synapse/models/<family>/<file>.axon \
  --checkpoint <checkpoint-id> \
  --device cuda:0 \
  --processes 1 \
  --dtype float32 \
  --axon-backend codegen2-torch \
  --axon-typechecker typecheck2 \
  --log-dir "$RUN" \
  --stream-csv "$RUN/stream.csv" \
  --debug-errors 2>&1 | tee "$RUN/run.log"
```

Size-bounded sweep:

```bash
RUN=log/<run-id>
mkdir -p "$RUN"
conda run -n brainsurgery brainsurgery synapse axon-benchmark \
  brainsurgery/synapse/models \
  --device cuda \
  --processes <n> \
  --dtype float32 \
  --axon-backend codegen2-torch \
  --axon-typechecker typecheck2 \
  --min-billion-parameters <min> \
  --max-billion-parameters <max> \
  --log-dir "$RUN" \
  --stream-csv "$RUN/stream.csv" \
  --debug-errors 2>&1 | tee "$RUN/run.log"
```

Pipeline2 large-model run:

```bash
RUN=log/<run-id>
mkdir -p "$RUN"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
conda run -n brainsurgery brainsurgery synapse axon-benchmark \
  brainsurgery/synapse/models \
  --device cuda \
  --processes <workers> \
  --pipeline-parallel-size <pp> \
  --axon-backend pipeline2-torch \
  --axon-typechecker typecheck2 \
  --dtype float32 \
  --min-billion-parameters <min> \
  --max-billion-parameters <max> \
  --log-dir "$RUN" \
  --stream-csv "$RUN/stream.csv" \
  --debug-errors 2>&1 | tee "$RUN/run.log"
```

## Reporting

For status/report requests, use the `report` skill or run:

```bash
conda run -n brainsurgery python scripts/benchmark_report_3tables.py log/<run-id> \
  --label "<label>" \
  --abs-threshold 1e-3 \
  --max-rows 80
```

## Debugging

- Check active processes with `ps -eo pid,ppid,stat,etime,cmd | rg "axon-benchmark|<run-id>"`.
- Inspect per-row failures under `log/<run-id>/log-*.txt`.
- Use `rg -n "Traceback|ERROR|Exception|RuntimeError|ValueError|AssertionError" log/<run-id>`.
- If a row fails under optional flags, rerun the same row without the flag before changing code.
- Do not rewrite `.axon` model files unless the model file is clearly wrong; compiler/runtime fixes must be generic and respect `AGENTS.md`.

## Output Contract

When starting a benchmark, state:

- run directory
- exact backend/typechecker/optimizer flags
- size/checkpoint selection
- process/GPU plan

When reporting, give the standard four tables via the `report` skill.
