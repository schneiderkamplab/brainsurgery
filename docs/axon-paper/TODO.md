# Axon Paper TODO

1. Add a section on supported primitive ops. Include a subsection for each op class and a table over all ops in that class. Where appropriate, give a mathematical formulation of the semantics. Where a mathematical formulation would be misleading or too broad, give a brief, precise textual explanation of the semantics.

2. Add a section on the builtins library. Include one subsection per builtins file explaining its purpose, plus a table of function signatures and concise semantics. Clearly separate exported functions from internal helper functions.

3. Fix stale evaluation numbers. Done for the current pass: the paper now reports `553` pairs, `256` model files, `274` checkpoints, `54` families, `301` generic pairs, and `31` test pairs, and stale partial tinygrad fidelity snapshots have been removed.

4. Add roundtrip results table. Done for the current pass: the paper reports the full roundtrip suite result, `4384` collected tests with `4123` ordinary passes, `261` non-strict XPass optimized-Graph trackers, and zero failures.

5. Document optimize split. Done for the first pass: the paper now describes `optimize-ast` and `optimize-graph` as separate optional phases, both off by default and outside the correctness-critical path.

6. Re-check Elaborate/Flatten after the implementation contract is fixed. Done for the first pass: the text now states the validated contract that elaboration inserts defaults only for uncovered formals, flatten does not repair defaults, and weak/strong roundtrips preserve the call surface.

7. Add stronger examples. Done for the first pass: the paper now has a running example following a small Axon fragment through normalize, elaborate, flatten, typecheck2, and Graph IR. Later passes can replace or augment it with a real model-family excerpt.

8. Add Graph IR detail. Done for the first pass: Graph lowering and validation now describe `GraphProgram`, `GraphModule`, `GraphNode`, `GraphValue`, operands, paths, multi-output binds, core ops, and validation checks. The running example now includes a TikZ Graph IR figure.

9. Improve related work / citations. Bibliography is minimal. Add MLIR, XLA, TVM, Torch FX / Dynamo, Transformers, tinygrad, shape/type-system references.

10. Refresh type-system claims after implementation stabilizes. `docs/axon-type-rules.md` was updated, and the paper type section is closer now, but it should be cross-checked again after the current typecheck2/roundtrip work settles.

11. Add validation matrix. A compact table should list each stage, required input invariant, output invariant, validator/test, and current pass/fail status.

12. Rebuild PDF after edits. `main.pdf` currently builds, but should be regenerated after the above documentation changes.
