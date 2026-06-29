---
status: active
last-confirmed: 2026-05-20
owners: agents
confidence: high
---

# Axon Roundtrip Workflows

This page records the current staged roundtrip script contracts.

Validated-by: repo inspection of `scripts/axon_*roundtrip.py` and `tests/test_synapse_axon_graph_ir_roundtrip.py` on 2026-05-20.

## Definitions

- Weak roundtrip: reparse the rendered artifact and rerun only the non-optional downstream stages required by that script. It intentionally omits at least one earlier stage.
- Strong roundtrip: reparse the rendered artifact and rerun all earlier stages up to the target stage.
- Graph optimized roundtrip: graph rendering with `optimize_graph=True`.
- Safe optimized roundtrip: AST safe optimization and graph optimization enabled together by the pytest wrapper.

## Stage Contracts

| Stage | Weak Contract | Strong Contract |
|---|---|---|
| parse | No weak/strong distinction; parse/render/parse stability only. | Same. |
| resolve | No weak/strong distinction; load+resolve render stability. | Same. |
| normalize | Reparse and renormalize without reresolve. | Reparse, reresolve, renormalize. |
| elaborate | Reparse, renormalize, elaborate without reresolve. | Rerun resolve, normalize, elaborate. |
| flatten | Reparse, renormalize, flatten without reresolve. | Rerun resolve, normalize, elaborate, flatten. |
| typecheck2 | Reparse, renormalize, typecheck without reresolve/reflatten. | Rerun resolve, normalize, elaborate, flatten, typecheck2. |
| graph-ir | Full first render, then weak graph rerender path. See note below. | Rerun resolve, normalize, elaborate, flatten, typecheck2, lower-to-graph, graph-render. |

## Graph IR Note

Graph IR can represent nested `GraphExpr` operands. If graph-rendered Axon intentionally uses non-flat but valid Axon syntax, the weak graph roundtrip path must include enough frontend work to accept and lower that syntax again. This is an active contract to watch when optimizer passes introduce nested expressions.

Depends-on: graph renderer in `brainsurgery/synapse/axon/graph_ir/render.py` and lowering in `brainsurgery/synapse/axon/graph_ir/core.py`.

## Pytest Integration

Primary test file: `tests/test_synapse_axon_graph_ir_roundtrip.py`.

Expected use with xdist:

```bash
conda run --no-capture-output -n brainsurgery \
  pytest -q -n 8 tests/test_synapse_axon_graph_ir_roundtrip.py
```

Use focused scripts directly when investigating a single stage:

```bash
conda run --no-capture-output -n brainsurgery \
  python scripts/axon_graph_ir_weak_roundtrip.py brainsurgery/synapse/models/gpt2/gpt2.axon \
  --optimize-graph --output-dir tmp/roundtrip-debug
```

## Maintenance Rule

When stage ordering changes, update:

- the relevant `scripts/axon_*roundtrip.py` script,
- `tests/test_synapse_axon_graph_ir_roundtrip.py` if pytest coverage changes,
- this page,
- `wiki/log.md` for nontrivial contract changes.
