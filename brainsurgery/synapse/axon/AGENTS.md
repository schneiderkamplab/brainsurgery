# brainsurgery/synapse/axon/AGENTS.md

Global policy: `../../../AGENTS.md`
Synapse policy: `../AGENTS.md`

## Scope

- Parser, typechecker, lowering, codegen, materialization, and pipeline planning.

## Allowed Changes

- Conservative bug fixes with targeted regression checks.
- Better diagnostics and invariant checks.
- Internal refactors that preserve public behavior and output contracts.

## Requires Approval

- Grammar/typing/lowering changes that broaden or alter language semantics.
- Any model-specific special casing in this layer.

## Unwanted Changes

- Embedding HF/model quirks directly into compiler/runtime layers.
- Compatibility hacks that bypass proper type/lowering invariants.
- Model-family name checks (direct or indirect) in parser/typecheck/lowering/codegen paths.
- Parallel shape metadata in AST nodes. Tensor dimensions must come from `TypeExpr` only.
- Parser-side semantic rewrites such as point-free eta expansion or constant evaluation.
