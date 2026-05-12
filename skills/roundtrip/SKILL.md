---
name: roundtrip
description: Run and debug Axon stage roundtrip tests. Use when asked to roundtrip parser, resolve, normalize, elaborate, flatten, typecheck2, Graph IR, optimize-ast, or optimize-graph outputs, especially with pytest-xdist parallel execution and weak/strong distinctions.
---

# Roundtrip

Use roundtrip tests to check that staged Axon rendering is canonical and stable.

## Defaults

- Use `conda run -n brainsurgery pytest` for pytest-based runs.
- Use `-n 8` for broad all-model runs when parallelism is requested.
- Use default xdist scheduling or `--dist load`, not `--dist loadfile`, for parametrized per-file tests.
- Write ad-hoc run logs under `log/<run-id>`.
- Prefer pytest tests over ad-hoc scripts for full-set runs because pytest gives per-file failures and tracebacks.

## Weak vs Strong

- Weak means the rendered output is reparsed and only the intentionally selected suffix of the pipeline is rerun.
- Strong means all prior required stages are rerun from the rendered output.
- For `typecheck2` and Graph IR roundtrips, normalize is always required because it disambiguates zero-arg calls from names.

## Common Commands

Full AST-optimized Graph IR roundtrip without Graph optimizer:

```bash
RUN=log/<run-id>
mkdir -p "$RUN"
conda run -n brainsurgery pytest -q -n 8 --dist load --tb=short -ra \
  tests/test_synapse_axon_graph_ir_roundtrip.py::test_ast_optimized_graph_ir_weak_roundtrip_is_canonical \
  tests/test_synapse_axon_graph_ir_roundtrip.py::test_ast_optimized_graph_ir_strong_roundtrip_is_canonical \
  2>&1 | tee "$RUN/pytest.log"
```

All Graph IR roundtrips:

```bash
conda run -n brainsurgery pytest -q -n 8 --dist load --tb=short -ra \
  tests/test_synapse_axon_graph_ir_roundtrip.py
```

Single file debug with persistent artifacts:

```bash
conda run -n brainsurgery python scripts/axon_graph_ir_weak_roundtrip.py \
  --optimize-ast \
  --no-show-types \
  --output-dir tmp/<debug-dir> \
  brainsurgery/synapse/models/<family>/<file>.axon
```

## Debugging

- If a pytest run appears serialized, check whether `--dist loadfile` was used; replace it with `--dist load`.
- If a roundtrip is unstable, compare `render1` vs `render2` and `render2` vs `render3`.
- If only `render1` differs and `render2 == render3`, the stage converges after one normalization pass but its renderer is not canonical by construction.
- For Graph IR render churn, first look for single-use alias/list/tuple temps before changing optimizer semantics.
- Do not weaken validators to make roundtrips pass.

## Output Contract

Report:

- command/run dir
- checked count, failed count, unstable count
- whether failures are weak, strong, or both
- representative diff class if unstable
