# Axon Paper TODO

1. Fix stale evaluation numbers. The paper still reports old tinygrad snapshot data and an outdated coverage table (`433` pairs). Current declared matrix was `553` pairs, `256` model files, `274` checkpoints, `54` families.

2. Add roundtrip results table. Include the same declared model/checkpoint matrix, not special tests. Current known issue: flatten weak/strong roundtrips fail for `523/553` pairs due to default-elaboration instability after rendering flat code.

3. Mark optimize as deferred. The paper says optimization is optional/excluded, but it should explicitly state that optimize is being rewritten and is not part of the claimed stable pipeline/evaluation.

4. Re-check Elaborate/Flatten after the implementation contract is fixed. The text now documents the intended contract: elaboration may insert missing defaults only for formals not already covered by positional, keyword, or path arguments, and flatten must not repair defaults. The implementation still needs the corresponding flat roundtrip fix before this can be marked fully done.

5. Add stronger examples. Done for the first pass: the paper now has a running example following a small Axon fragment through normalize, elaborate, flatten, typecheck2, and Graph IR. Later passes can replace or augment it with a real model-family excerpt.

6. Add Graph IR detail. Done for the first pass: Graph lowering and validation now describe `GraphProgram`, `GraphModule`, `GraphNode`, `GraphValue`, operands, paths, multi-output binds, core ops, and validation checks. The running example now includes a TikZ Graph IR figure.

7. Improve related work / citations. Bibliography is minimal. Add MLIR, XLA, TVM, Torch FX / Dynamo, Transformers, tinygrad, shape/type-system references.

8. Refresh type-system claims after implementation stabilizes. `docs/axon-type-rules.md` was updated, and the paper type section is closer now, but it should be cross-checked again after the current typecheck2/roundtrip work settles.

9. Add validation matrix. A compact table should list each stage, required input invariant, output invariant, validator/test, and current pass/fail status.

10. Rebuild PDF after edits. `main.pdf` currently builds, but should be regenerated after the above documentation changes.
