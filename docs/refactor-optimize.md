# Axon Optimize Refactor Plan

This document classifies the current Axon optimizer pieces by semantic safety and
defines the hardening path for optional AST and Graph IR optimization.

The current rule should be:

- `optimize-ast` and `optimize-graph` are optional and off by default
- safe local rewrites may become a small always-available conservative pass
- speculative rewrites stay disabled until they have explicit semantic preconditions,
  validators, and fidelity coverage
- anything required by Graph IR/codegen should be represented as a stage contract,
  not hidden inside optional optimization

## Current Status

`optimize_flat_typed_axon_file` currently mixes several categories:

- required backend-shape normalization
- rooted dead-definition pruning
- atomic alias cleanup
- literal/local constant folding
- temp elimination
- parameter pruning
- path/template rewrites
- specialization
- inlining
- repeated re-typecheck/re-validate fixpoint orchestration

That mixture is the main problem. The passes do not all have the same semantic risk.
Some are safe as typed AST rewrites. Some are only safe under narrow preconditions.
Some are better done on typed Graph IR instead of Axon AST.

Implementation status:

- `optimize_safe_flat_typed_axon_file` exists as an optional conservative pre-Graph
  optimizer. It runs rooted pruning, atomic alias cleanup, and literal/atomic-only
  folding to fixpoint with `typecheck2` and typed validation after each iteration.
- `optimize_graph_program` exists as an optional typed Graph IR optimizer. It
  currently performs graph pruning, atomic alias cleanup, dead temp elimination for
  total `core.*` nodes, literal-only constant folding, conservative single-callsite
  specialization, and constrained safe inlining of total-pure helper modules.
- Graph IR pruning is metadata-aware: type annotations, dim metadata, constraints,
  and path-template symbols are treated as dependencies. This prevents pruning
  constants such as `CFG`/shape names that only appear in templates or typed
  metadata.
- Safe inlining no longer treats zero-arg atomic constants as ordinary
  single-callsite helpers. Atomic constants stay as constants by default; they are
  only substituted by the explicit, opt-in constant-dim substitution pass.
- Optional graph constant-dim substitution exists behind
  `GraphOptimizeConfig.constant_dim_substitution=False`. It is local,
  constraint-gated, and validates each candidate rewrite. A module must both
  reference the constant in dim/type metadata and carry a local equality
  constraint such as `VOCAB_SIZE = 151936` before the substitution can apply.
- Graph IR validation now includes a conservative type/arity verifier for typed
  operands, module calls, core ops, stale value references, and node/module output
  contracts. It intentionally accepts current Axon polymorphism such as `Any`,
  type variables, symbolic dimension binders, row variables, optional-null
  defaults, Dim/Int coercions, and Int-to-Float widening.
- Graph IR has an explicit effect model. Unknown calls are `partial_pure` by
  default; only known-total `core.*` ops are currently `total_pure`. Graph DCE and
  safe inlining consult this model instead of a local ad-hoc allowlist.
- Smoke validation:
  `generic-llama4.axon` with checkpoint `test/Llama4-Test`, backend
  `codegen2-torch`, dtype `float32`, device `cuda:0`, max length 16 passed Graph
  IR validation and codegen execution. Result: `masked_top1_eq=True`,
  `masked_max_abs_diff=0.010181993246078491`. Log:
  `log/smoke-llama4-graph-validate-20260511c`.
- Both are off by default. CLI switches are `--optimize-ast` and
  `--optimize-graph`.
- Broad graph-optimized weak and strong roundtrip coverage currently passes:
  `tests/test_synapse_axon_graph_ir_roundtrip.py::{test_graph_optimized_graph_ir_weak_roundtrip_is_canonical,test_graph_optimized_graph_ir_strong_roundtrip_is_canonical}`
  produced `512 passed` on 2026-05-12 after fixing the graph closure,
  path-template substitution, atomic-constant inlining, and loop-helper return
  type propagation clusters.
- The old broad AST optimizer implementation remains as internal experimental
  code, but the user-facing `--optimize` flag has been removed.
- There is still no full Graph IR inference engine. Graph optimization relies on
  typed Graph IR input, type-preserving rewrites, the graph verifier, and
  conservative effect defaults.

## Removed Backend-Required Normalize

The old backend-required normalize stage has been removed from the active
codegen2/Graph IR pipeline. Graph IR supports multi-output nodes natively, so
list destructuring is represented structurally instead of being rewritten into
backend-specific `_list_index` scaffolding.

The removed pass used to rewrite:

```axon
x, y, z <- Tensor.chunk value parts=3
```

becomes roughly:

```axon
tmp <- Tensor.chunk value parts=3
x <- _list_index tmp 0
y <- _list_index tmp 1
z <- _list_index tmp 2
```

Current decision:

- keep no legacy backend-required stage just for old backend constraints
- keep no `validate.backend_required` stage
- if a backend cannot consume multi-output nodes, that backend should lower or split
  the graph IR explicitly, not force all Axon lowering through `_list_index`

## Safe Passes

These are safe candidates for a conservative pre-Graph-IR optimizer, provided they
keep retyping and validation after each iteration.

### Rooted Unreachable Definition Pruning

This is safe when it uses the shared resolve/typecheck reachability logic and an
explicit selected main module.

Rules:

- use `resolve_main_module`
- use shared `prune_unreachable_definitions`
- never rely on an implicit "last module" default after main selection is known
- preserve pragmas and type aliases
- validate closed/flat/typed shape after pruning

This should not live as optimizer-specific call-graph code.

### Atomic Alias Cleanup

Safe:

```axon
y <- x
return y
```

to:

```axon
return x
```

Safe atomic expressions are names, scalar literals, `null`, strings, paths, and
containers whose elements are all atomic.

Required constraints:

- do not duplicate non-atomic expressions
- do not cross rebinding of the same name
- preserve return arity and typed metadata
- re-typecheck after rewriting

### Literal-Only Constant Folding

Safe when both sides are genuinely literal or the rewrite is an algebraic identity
whose behavior is independent of evaluation effects.

Safe examples:

```axon
x <- 1 + 2
y <- true and flag
z <- null == null
```

Unsafe unless proven:

```axon
x <- f() == f()
y <- null == maybe_partial()
```

Current implementation should be treated as safe only for:

- int/int arithmetic and comparisons with literal operands
- float/float arithmetic with literal operands
- bool/bool operations with literal operands
- `true or x`, `false and x`, `x and true`, `x or false` only if they do not
  introduce or remove evaluation of partial expressions incorrectly
- `x == x` only when `x` is atomic and already evaluated, not an arbitrary call
- null-vs-non-null folding only when non-null is derived from a typed value that
  is already evaluated, not by deleting a partial expression

Hardening requirement:

- the folding helper should expose a predicate like `is_safe_fold_operand`
- every fold case should document whether it is literal-only, atomic-only, or
  type-derived
- tests should include rejected folds for calls/config/param lookups

### Retype And Validate

Safe and required.

Every optimization iteration should:

- validate input invariants
- apply one or more monotonic rewrites
- re-run typecheck
- validate typed output
- stop on structural fixpoint

If constraints are consumed, they must be refreshed by re-typecheck before the next
constraint-consuming pass.

## Unsafe Or Experimental Passes

These should stay out of default optimization until hardened.

### `_atomicize_call_args_statements`

Purpose:

- make complex call arguments into explicit temporary binds
- produce a shape that is easier for downstream codegen/runtime
- example intended rewrite:

```axon
y <- f (g x)
```

to:

```axon
tmp <- g x
y <- f tmp
```

Problem:

Axon is eager except for ternary branches. Hoisting out of a ternary can change
semantics:

```axon
y <- cond ? good : bad()
```

must not become:

```axon
tmp <- bad()
y <- cond ? good : tmp
```

because `bad()` was previously evaluated only when `cond` was false.

Hardening:

- only atomicize inside an expression when the expression position is eager
- never hoist from ternary branches to a prefix outside the ternary
- either leave branch expressions nested or introduce explicit branch-local graph
  structure at Graph IR level
- add validator coverage that atomicization preserves ternary laziness

Recommendation:

- do not run this pass on Axon AST by default
- prefer doing backend-specific argument normalization on Graph IR, where branch
  semantics can be represented explicitly

### Purity-Based Dead Code Elimination

Current risk:

The optimizer currently treats calls as pure if their arguments are pure. That is
not sound for Axon.

Example:

```axon
main :: Tensor[B,S,D]
main = do
  unused <- Params.param @@missing
  return x
```

Deleting `unused` removes a missing-parameter error. That changes semantics.

Other partial operations:

- `Config.int @@missing`
- `List.index xs i`
- tensor shape/index ops with invalid dims
- any primitive that can reject at runtime
- parameter or config lookup

Hardening:

- do not classify calls as pure by default
- introduce an explicit primitive/definition effect model:
  - `total_pure`: can be removed/duplicated/reordered
  - `partial_pure`: deterministic but may fail; can be common-subexpression
    optimized only if evaluation is preserved
  - `effectful`: cannot be removed/reordered
- until this exists, DCE may only delete unused binds whose RHS is atomic or a
  literal/container of atomics

### Single-Use Pure-Bind Inlining

This has the same purity issue as DCE, plus an ordering issue.

Constrain it to:

- atomic RHS
- literal/container RHS
- already-evaluated names
- no call RHS unless the effect model proves total purity
- no movement across ternary branch boundaries
- no change that produces non-flat Axon unless the next stage explicitly accepts it

### Constant-Param Specialization

Useful but not safe enough as a broad default.

Hardening requirements:

- only specialize parameters whose actuals are atomic literals or paths
- do not specialize on values that may be rebound in recursive SCCs
- use SCC analysis to avoid non-termination
- require a monotonic metric:
  - number of removed params decreases, or
  - number of callsites decreases, or
  - rendered/structural hash reaches fixpoint without cycles
- bind callsites against the old signature and rewrite against the new signature
  through one shared signature-rewrite utility
- preserve constraints by substituting specialized params into constraints
- re-typecheck immediately after specialization
- reject the pass result if typecheck changes observable signature semantics

### Single-Callsite Specialization

Safe only when the callsite is unique and actuals are safe.

Additional requirements:

- do not specialize main
- do not specialize recursive self-calls
- do not specialize across recursive SCCs unless a formal recursive specialization
  rule exists
- do not specialize on path templates unless all template variables are explicitly
  in lexical scope and represented structurally
- include tests with recursive loop helpers, optional caches, and path templates

### Multi-Statement / Module Inlining

Inlining can be correct, but it is a high-risk transformation.

Hardening:

- inline only straight-line modules:
  - no `if`
  - no `for` / `repeat`
  - no `scope`
  - no `yield`
  - no branch-sensitive ternary hoisting
- require one callsite unless a separate duplication-cost policy exists
- freshen names structurally, not by regex
- substitute type variables, dim variables, path variables, and constraints through
  one shared substitution engine
- validate closedness after inlining
- validate flatness after inlining
- re-typecheck after inlining
- compare call result type before/after inlining

Inlining in expression position is especially risky. It should not introduce prefix
statements that are evaluated outside the expression's original lazy branch context.

### Constraint-Driven Branch Folding

Constraint folding is safe only if constraints are fresh and provenance-aware.

Requirements:

- constraints must be produced by the latest typecheck run
- optimizer must mark constraints stale after any rewrite that changes:
  - definitions
  - call arguments
  - guards
  - params
  - return expressions
  - types/dims
- any pass using constraints must run only after fresh typecheck
- constraints need provenance:
  - local fact
  - branch guard
  - callsite guard
  - propagated interprocedural guard
- branch folding may only use facts whose guard dominates the folded expression
- after folding, re-typecheck and revalidate constraints

Do not use global per-module facts as if they held unconditionally unless the
constraint store explicitly marks them unconditional.

### Loop / Path Helper Prefixing

The phrase refers to optimizer logic that detects generated loop helper definitions
from their names and rewrites absolute path templates in their bodies.

Example shape:

```axon
gpt2__loop_h_recur_continue_2 ...
```

The helper name encodes that it belongs to loop `h` with loop variable `i`; the
optimizer then tries to prefix paths such as:

```axon
@@attn.c_proj
```

into something like:

```axon
@@'h.{i}.attn.c_proj'
```

Problem:

This is semantic information inferred from generated names. That is brittle.

Hardening:

- flatten should attach structural metadata to generated helpers:
  - original loop name
  - loop variable
  - lexical scope path
  - template variables in scope
- optimize should consume that metadata, not parse helper names
- path composition should operate on structured `AxonExprPath`, not strings
- validate that all template placeholders are lexically bound after rewrite
- if metadata is absent, skip the pass rather than guessing

## Graph IR Versus Pre-Graph Optimization

Most optimization should eventually happen at Graph IR level.

Better pre-Graph-IR passes:

- rooted pruning
- atomic alias cleanup
- literal-only constant folding
- safe path template normalization that still needs Axon lexical metadata
- readability/canonical naming if the output is an Axon inspection artifact

Better Graph IR passes:

- dead temporary elimination
- CSE
- pure op folding
- backend argument normalization
- shape-based simplification
- graph-level branch/select simplification
- MoE selected-expert rewrite from compact select/scatter form to top-k-axis
  batched expert execution
- op fusion
- custom kernel selection
- layout planning
- pipeline partitioning

Reasons:

- Graph IR has typed operands and explicit node outputs.
- Graph IR can preserve multi-output destructuring without lowering through lists.
- Backend constraints belong closer to backend IR.
- Eager/lazy evaluation hazards are easier to make explicit if branch/select nodes
  have clear semantics.
- Future custom kernels need graph-level shape/type facts.

Recommendation:

- keep pre-Graph optimizer small and conservative
- move aggressive optimization to typed Graph IR
- define an explicit Graph IR effect/partiality model before DCE, inlining, or CSE
  over calls

Graph-level pruning is still needed even though resolve/typecheck prune Axon
definitions. Graph optimization can introduce specialized clones, inline helpers,
and remove callsites; pruning keeps the executable graph rooted at `main` after
those graph-only rewrites.

A full Graph IR typechecker is not currently required for the implemented safe
passes, but the validation boundary should grow. The near-term target is a
Graph IR verifier that checks:

- all operands are typed
- op outputs preserve the declared node/module output types
- specialization substitutes inputs consistently in nodes and outputs
- inlining preserves the original call result arity and result types
- no non-total call is deleted, duplicated, or reordered

## Missing Work

The current implementation is a first conservative slice. Before any optimizer is
enabled by default, these gaps need to be closed.

### Graph IR Verifier / Type Checker

Status: first conservative verifier implemented. Remaining items below are the
next hardening steps.

`validate_graph_program` is structural. It checks references, module existence,
input/output names, and duplicate definitions. It does not yet prove that graph
rewrites preserve type and arity semantics.

Missing:

- stricter signature-substitution checks for specialization
- explicit before/after call-result equivalence checks for inlining
- better distinction between alpha-equivalent dim binders and accidental dim
  mismatch
- validation of lowered constraints against graph values
- optional stricter mode for tests that should reject broad `Any`/type-variable
  compatibility

Until the remaining checks exist, graph optimization must stay conservative and
off by default.

### Effect / Partiality Model

Status: first explicit model implemented. Unknown calls default to
`partial_pure`; known-total `core.alias`, `core.ascribe`, `core.list`,
`core.tuple`, and `core.binary.*` are `total_pure`. `core.select` remains
`partial_pure` because the selected branch may be partial and must not be deleted.

Graph DCE currently deletes only known-total `core.*` nodes. This is safe but
limited. Most Axon calls must be treated as partial unless explicitly proven
otherwise, because deleting an unused call can delete a required runtime error.

Missing:

- primitive effect declarations
- richer derived definition effect inference with SCC-aware fixed-point reporting
- effect propagation through nested graph expressions
- richer distinction between:
  - `total_pure`: removable, duplicable, reorderable
  - `partial_pure`: deterministic but may fail; evaluation must be preserved
  - `effectful`: must not be removed, duplicated, or reordered
- tests that `Params.*`, `Config.*`, `List.index`, and invalid tensor ops are not
  deleted unless an effect rule permits it

Broader DCE, CSE, and inlining should wait for this model.

### Specialization Hardening

Current graph specialization is deliberately narrow: it specializes only safe
literal/path actuals, does not specialize `main`, and is mainly intended for
single-callsite helpers.

Status:

- SCC analysis is implemented for specialization.
- Recursive SCCs and self-recursive modules are not specialized.
- Modules with constraints are not specialized until constraint substitution is
  implemented.
- Path actuals are specialized only when they do not contain template
  placeholders.
- Specialized clone names are collision-safe.
- Graph validation runs after clone creation and callsite rewrite.
- Tests cover successful single-callsite literal specialization and rejection of
  recursive self-specialization.

Missing:

- constraint substitution and validation
- structural substitution for path templates and template variables
- rewriting support for nested module calls, or explicit validation that no nested
  callsites are being specialized accidentally
- monotonic termination metric beyond iteration cap
- benchmark coverage on generated loop helpers and path-templated helpers

### Inlining Hardening

Current graph inlining only inlines helpers proven `total_pure` by the graph
effect model. It is intentionally limited and only rewrites top-level call nodes.

Status:

- Inlining skips `main`, recursive SCCs, and modules with constraints.
- Inlining requires exactly one callsite and exactly one top-level call node.
- Nested expression callsites are not inlined yet.
- The call node must pass all args positionally, with matching input/output arity.
- Actual/formal and returned/output types are checked with the same compatibility
  rules as `validate_graph_program`.
- Inlining rewrites module returns through the same substitution used for nodes.
- Graph validation runs after inlining and pruning.
- Tests cover single-output helpers, multi-output helpers, path operands,
  constrained helpers, and nested expression callsites.

Missing:

- constraint substitution
- explicit branch/laziness checks for nested expression inlining
- cost policy for duplicating work
- tests for tuple/list-valued helpers
- broader benchmark coverage on generated model graphs

Until then, broad module inlining belongs behind `--optimize-graph` only.

### Graph Roundtrip / Stability Tests

Focused unit tests exist. Graph IR roundtrip scripts now accept an
`--optimize-graph` switch, and dedicated optimized weak/strong script entrypoints
exist for manual or CI runs.

Status:

- `scripts/axon_graph_optimize_weak_roundtrip.py` runs the weak optimized Graph IR
  roundtrip.
- `scripts/axon_graph_optimize_strong_roundtrip.py` runs the strong optimized
  Graph IR roundtrip.
- A focused weak pytest covers `generic-gpt2-kv.axon` with graph optimization.
- A focused weak pytest covers `generic-llama4.axon` with graph optimization.
- Graph IR now treats path-template placeholders such as `{CFG}` as first-class
  dependencies for validation, DCE liveness, and pruning.
- `_topk` no longer infers a fake concrete rank from variadic `Tensor[..S]`;
  when rank is unknown, the primitive rule declines and leaves the broader
  wrapper signature.
- Graph IR validation reuses the same variadic dim compatibility rule for value
  references that it uses for type compatibility.
- Graph IR validation no longer compares instantiated call-result types against
  unsubstituted generic callee return signatures; it still validates actual
  arguments, local node output contracts, module returns, and graph closedness.
- Typecheck2 now re-annotates return expressions after unification against an
  explicit return signature, and branch joins preserve rank for matching
  variadic tensor branches.
- Full non-optimized Graph IR weak and strong roundtrips pass for all 256 model
  `.axon` files under `pytest-xdist -n 8`.
- Broader optimized roundtrips are intentionally not enabled in pytest yet.
  Strong optimized reparsing still needs a cleaner contract for already-closed
  graph-rendered artifacts.

Missing:

- tests across all model `.axon` files, run with `pytest-xdist`
- stable textual diffs for failures, similar to the Axon stage roundtrips
- resolving whether strong optimized graph roundtrip should reresolve closed
  graph-rendered artifacts from arbitrary temp paths or parse them directly

### Benchmark / Fidelity Validation

The flags are wired but not validated on full model families.

Status:

- `--optimize-ast` smoke passed on `generic-gpt2-kv.axon` with
  `openai-community/gpt2`: `masked_top1_eq=True`, max abs diff
  `0.0001068115234375`.
- Historical broad AST optimizer smokes passed on:
  - `generic-gpt2-kv.axon` with `openai-community/gpt2`:
    `masked_top1_eq=True`, max abs diff `0.0001068115234375`
  - `bert-base-uncased.axon` with `google-bert/bert-base-uncased`:
    `masked_top1_eq=True`, max abs diff `2.956390380859375e-05`
  - `generic-llama4.axon` with `test/Llama4-Test`:
    `masked_top1_eq=True`, max abs diff `0.010181993246078491`
  - `generic-olmoe.axon` with `test/OLMoE-Test`:
    `masked_top1_eq=True`, max abs diff `5.364418029785156e-07`
- The broad AST smoke exposed and fixed two generic primitive/type metadata bugs:
  `_topk` should not preserve the eliminated `IdxTensor` alias in core typed AST,
  and `_tensor_like` should use the reference tensor shape.

Missing:

- smoke benchmark with `--optimize-graph` alone
- combined `--optimize-ast --optimize-graph` smoke passed on
  `generic-gpt2-kv.axon` with `openai-community/gpt2`: `masked_top1_eq=True`,
  max abs diff `3.0517578125e-05`. Log:
  `log/smoke-optimize-flags-20260511`.
- all-file combined optimized weak Graph IR roundtrip is now covered as a
  non-strict xfail tracker. Latest run:
  `pytest -q -n 8 tests/test_synapse_axon_graph_ir_roundtrip.py -k safe_optimized_graph_ir_weak_roundtrip_is_canonical`
  produced 66 stable files and 190 files with changed rendered text across
  generations. The hard all-file roundtrip requirements remain the non-optimized
  weak and strong Graph IR roundtrips.
- max-4B combined optimized fidelity is covered by an opt-in long pytest:
  `BS_LONG=1 pytest tests/test_synapse_optimizer_fidelity_max4b.py`. It runs all
  `generic-*.axon` files through `axon-benchmark` with
  `--max-billion-parameters 4`, `--optimize-ast`, and `--optimize-graph`, and
  requires masked top-1 equality plus max absolute diff below `1e-2`.
- max-1B and max-4B fidelity comparison against the current non-optimized default
- targeted large-model reruns for families affected by path templates, loops, MoE,
  cache helpers, and multi-output calls

Any optimizer default change requires clean fidelity evidence first.

### Legacy Optimizer Decomposition

The old broad AST optimizer implementation still exists internally and still
mixes safe, unsafe, and backend-shape rewrites. It is no longer exposed through a
CLI flag.

Missing:

- move every safe pre-Graph rewrite into `optimize_safe`
- move graph-suitable rewrites into `optimize_graph`
- delete or quarantine any remaining legacy-only rewrites that exist for old backend contracts
- ensure no required backend behavior is hidden behind optional optimization

### Graph-Level Constraints

Graph IR does not yet own a fresh constraint store.

Missing:

- lower typed constraints into Graph IR, or recompute graph constraints directly
- mark constraints stale after any graph rewrite that changes operands, guards,
  calls, outputs, or types
- validate that branch facts dominate the folded expression before using them
- support guarded facts for `core.select`, null checks, bool checks, and dim/int
  equalities

Until this exists, constraint-driven branch folding should stay out of graph
optimization.

### Higher-Level Graph Optimizations

The current graph optimizer performs cleanup, not substantial backend optimization.

Missing future passes:

- CSE for total operations
- shape/dim simplification
- dead branch elimination with fresh constraints
- backend argument normalization where needed
- op fusion
- layout planning
- custom kernel selection
- pipeline partitioning support
- MoE selected-expert rewrite from compact `where/scatter` form to top-k-axis
  batched expert execution

### MoE Selected-Expert Rewrite

Current derived MoE code often has an expert-centric shape:

```axon
routed <- Tensor.zeros_like hidden
routed <- for e <- [0..E) carry (routed) do
  x_sel, token_idx, topk_pos, sel_scores <- MoE.select hidden topk_scores topk_indices e
  x_upd <- expert_body e x_sel
  routed <- MoE.scatter_add routed token_idx x_upd sel_scores
  yield routed
return routed
```

`MoE.select` currently reaches the primitive `_where_indices`, which produces
data-dependent compact index tensors. That is hard for tensor backends without a
native nonzero/compaction primitive and should not be handled by backend
special-casing normal Axon modules.

A future Graph IR optimization can recognize the semantic pattern, with structural
checks rather than module-name hacks, and rewrite it into selected-expert batched
execution over the router top-k axis:

```text
hidden_sel = unsqueeze(hidden, 2) expanded to [B,T,K,D]
gate_up    = indexed_expert_linear(gate_up_weight, hidden_sel, topk_indices)
act        = expert activation/body over [B,T,K,*]
down       = indexed_expert_linear(down_weight, act, topk_indices)
routed     = sum(down * unsqueeze(topk_scores, -1), dim=2)
```

Required preconditions:

- the loop range covers expert ids and has no side effects beyond the routed carry
- `MoE.select` consumes the same `hidden`, `topk_scores`, and `topk_indices` for
  each expert
- the selected expert id is used only to select expert parameters, not as an
  arbitrary term with other semantics
- `MoE.scatter_add` is the inverse accumulation of `MoE.select`
- the expert body is straight-line and preserves the leading selected-token axis
- all parameter paths and expert weight layouts are represented structurally

The rewrite target should be generic Graph IR primitives or canonical library ops
such as indexed expert linear/gathered expert matmul. Backends can then implement
those efficiently:

- torch can use advanced indexing plus `einsum`/batched matmul
- tinygrad can use the same selected-weight indexing pattern used by its local
  MoE example (`weight[sel]`) and reduce over the top-k axis

This pass is an optimization, not a semantic crutch. If the pattern is not proven,
the graph should remain unchanged; a backend that cannot lower `_where_indices`
should then fail with an unsupported primitive error.

## Proposed Refactor Slices

1. Keep backend-required normalization removed.
- Graph IR/codegen2 should handle multi-output nodes directly.
- Required backend behavior must be represented in Graph IR or in explicit backend lowering.
- Do not reintroduce non-optional AST rewrites for old backend contracts.

2. Introduce `optimize-ast` / `optimize_safe`.
- Contains rooted pruning, atomic alias cleanup, literal-only folding.
- Runs to fixpoint with typecheck/validate.
- No call DCE, no specialization, no inlining.

3. Add optimizer validators.
- `validate.optimized_safe`
- checks no stale typed metadata
- checks no branch-hoisted statements
- checks typed/flat/closed invariants

4. Add an effect model.
- Start with conservative defaults: calls are partial unless declared otherwise.
- Primitive declarations can opt into `total_pure`.
- User definitions inherit the least-safe effect of their body.

5. Harden specialization and inlining.
- Shared signature rewrite utility.
- Shared substitution utility for terms/types/dims/constraints/paths.
- SCC-aware termination checks.
- Explicit metadata for generated helpers.

6. Move broader rewrites to Graph IR.
- DCE
- CSE
- backend argument normalization
- op fusion
- shape/layout driven optimizations

## Default Policy

For now:

- `--optimize-ast` and `--optimize-graph` should remain off by default
- full `optimize_flat_typed_axon_file` should be treated as internal experimental
  code, not as a CLI/backend contract
- safe pieces should be split out before enabling any optimizer by default
- benchmark/fidelity runs should include both non-optimized and safe-optimized paths
  before changing defaults
