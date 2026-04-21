# Reimplement Axon Plan

## Execution Plan First: Flat-Core Desugaring

This section defines the concrete rollout plan before the rest of the agenda below.

### Objective

Move to a flat, desugared core Axon where:

- function bodies are straight statement lists in `do ... return ...`
- control flow is desugared to helper definitions
- `select` (ternary) remains the only non-straight functional control concept (for lazy branch evaluation)
- loops are represented via recursion after desugaring

### Phase 1: Introduce explicit `for ... carry ... yield ...`

Add the new loop surface first, with strict typing/arity checks.

Example:

```axon
x, new_kv <- for@h i <- [0..L) carry (x, new_kv) yield (gpt2_loop_step@h i x attn_mask past_kv new_kv)
```

Rules:

- loop LHS arity/types must match `carry` and `yield` arity/types
- `yield` is the per-iteration next state
- loop scope instantiation remains deterministic

### Phase 2: Add path templating in loop scopes and paths

Support template segments in scope/path strings using in-scope values (including loop vars).

Scope behavior:

- if template contains placeholders, instantiate placeholders each iteration
- if no placeholder appears, keep existing behavior and append `.<loop_value>`

Examples:

```axon
for@h i <- [0..L) ...
for@layers.{i}.attn i <- [0..L) ...
for@{base_scope}.block.{i} i <- [0..L) ...
```

Absolute-path strategy:

- flatten to absolute paths where possible
- for dynamic indices, use absolute templates (for example `@@h.{i}.attn.c_attn.weight`) and instantiate during lowering/codegen

#### Proposed Implementation Plan (Phase 2)

1. Define syntax + AST representation.
- Introduce a dedicated parsed form for templated segments (do not keep raw strings as the semantic form).
- Allow placeholders only in path-like positions (`scope@...`, `@...`, `@@...`) and reject elsewhere.
- Do not keep legacy string-based path representation.
- Parse all paths into structured path AST/IR nodes (templated and non-templated).
- Treat string paths as invalid syntax (hard error, no compatibility shim).
- Use this as the basis to remove ad-hoc root/prefix mechanisms (`root=...`, `prefix_path=...`, similar path-concatenation conventions) in favor of explicit templated path construction.

2. Add a path-template normalization pass (surface-only).
- Normalize all path-like literals into one canonical internal form:
  - static path segments
  - placeholder segments bound to in-scope symbols
- Reject unresolved placeholders early with precise source spans.
- Reject unsupported placeholder value types (must be statically representable as path segments).

3. Add loop-aware instantiation semantics.
- For `for@...` with templates:
  - instantiate placeholders from the current loop environment.
- For `for@...` without templates:
  - preserve current auto-append behavior (`.<loop_value>`).
- Make this behavior explicit in one helper used by both typecheck and lowering (single source of truth).

4. Add typed path IR (first slice).
- Introduce a minimal `PathExpr` IR for resolved paths:
  - `PathLiteral(parts=[...])`
  - `PathTemplate(parts=[literal|placeholder])`
- Keep string emission only at backend boundary; internal passes consume `PathExpr`.

5. Lowering integration with deterministic materialization.
- During lowering, materialize concrete paths when all placeholders are known constants.
- If placeholders remain symbolic at lowering time, emit canonical runtime template form and defer expansion to runtime/codegen in one place only.
- Remove ad-hoc string concatenation for scopes/paths from individual ops.

6. Tests and acceptance gates.
- Parser tests:
  - valid templates, invalid placeholders, and illegal contexts.
- Typecheck tests:
  - unresolved placeholder rejection and loop-scope symbol visibility.
- Lowering tests:
  - concrete instantiation parity with existing behavior.
  - mixed static/dynamic template segments.
- Regression tests on representative models:
  - one dense decoder (GPT-like), one encoder model, one MoE model.

7. Rollout + cleanup.
- Land behind a temporary feature flag for one PR cycle.
- Run full parity sweep and compare:
  - compile success rate
  - fidelity flags
  - runtime regression budget
- Remove flag and legacy path-string branches after parity holds.

8. Path-typed op audit.
- Audit all existing primitive and wrapper ops for parameters that eventually resolve to tensor/state-dict paths.
- Require those parameters to be structured path type in signatures and IR (not `String`/`Any` placeholders).
- Add validation that rejects non-path-typed values at parse/typecheck boundaries.
- Track migration status per op and block new path-like kwargs that are not path-typed.

#### Phase 2 Deliverables

- Canonical template syntax + parser support
- `PathExpr` core representation (initial version)
- Shared template-instantiation helper
- Replacement of path/scope string heuristics in lowering hot paths
- Test suite covering parser/typecheck/lowering + model smoke cases

#### Phase 2 Exit Criteria

- No unresolved template/path ambiguity reaches lowering.
- No model parity regression attributable to scope/path resolution.
- No duplicate path-resolution logic remains across typecheck/lowering/runtime.

### Phase 2.2b: Overloaded Type Signatures (Exact-Match)

Add support for multiple type signatures per definition name, with strict overload resolution.

Example:

```axon
mul :: Tensor[..S] -> Float -> Tensor[..S]
mul :: Float -> Tensor[..S] -> Tensor[..S]
mul :: Float -> Float -> Float
mul = _mul
```

Rules:

- No coercive matching in overload resolution.
- Ambiguous matches are hard errors.
- No legacy fallback to "first signature wins" unless exactly one candidate matches.

#### Proposed Implementation Plan (Phase 2.2b)

1. Parser + AST support for repeated signatures.
- Permit multiple `name :: ...` signatures for the same definition name in one module.
- Store signature groups in source order for deterministic diagnostics.

2. Signature table refactor in typecheck.
- Change `name -> ModuleSignature` to `name -> list[ModuleSignature]`.
- Keep path-parameter metadata per overload.

3. Overload candidate filtering.
- Filter by:
  - path-arg arity from `abc@p1@p2` sugar
  - positional/kwarg arity and required/defaulted arguments
  - kwarg names
- Reject invalid calls before type unification.

4. Exact-match type unification per candidate.
- Reuse existing unification logic (including symbolic dims/rest dims).
- Candidate matches only if all args and returns typecheck without coercion.

5. Ambiguity handling.
- `0` matches: existing "no matching signature" style error.
- `>1` matches: explicit ambiguous-overload error listing candidate signatures and conflicting argument positions.

6. Kwarg/default/path semantics integration.
- Apply `abc@p1@p2` sugar generically to first unbound `Path`-typed positional arguments per candidate.
- Resolve kwargs/defaults against each candidate before matching.
- Do not add alias-specific special casing.

7. Tests.
- Positive:
  - scalar/scalar and tensor/scalar overloads for the same symbol.
  - path-sugar binding with multiple path-typed parameters.
  - symbolic-dim overloads with one unique match.
- Negative:
  - ambiguous candidates.
  - missing/extra kwargs with overload sets.
  - path-suffix count mismatch.

#### Phase 2.2b Exit Criteria

- Overload selection is deterministic and exact-match only.
- Ambiguous calls fail with actionable diagnostics.
- Existing non-overloaded modules preserve behavior.
- No lowering/runtime changes needed beyond consuming resolved overload identity.

### Phase 3: Resolve imports into a self-contained linked program

Before flattening, build a closed linked program IR:

1. parse source files
2. resolve imports/exports across modules
3. collect reachable definitions from selected main entrypoint
4. apply canonical namespacing/renaming to avoid collisions
5. keep this as linked in-memory IR (single-file emission optional)

Goal:

- desugaring and later stages run on a self-contained program with no unresolved imports

### Phase 4: Add explicit desugaring/flattening stage

Insert a `Surface Axon -> Flat Core Axon` pass into the pipeline:

1. parse/import-load
2. syntax/AST validation
3. path templating normalization
4. import/link resolution to closed program
5. **desugar/flatten (new)**
6. typecheck on flat core
7. lower on flat core

Desugaring responsibilities:

- ANF-style flattening into one statement per bind
- helper extraction for loop/branch bodies
- loop-to-recursion desugaring
- call canonicalization (kwargs/defaults)
- lexical path normalization to absolute/templated-absolute form

### Flat-core style and examples

#### Loop recursion form

```axon
loop i x new_kv = do
  done <- (i >= L)
  out <- done ? (x, new_kv) : (loop_else i x new_kv)
  return out

loop_else i x new_kv = do
  x, new_kv <- step i x new_kv
  i <- i + 1
  out <- loop i x new_kv
  return out
```

#### Ternary stays as `select`

`select` remains the canonical lazy-branch form in core lowering; branch bodies can be helper calls:

```axon
out <- cond ? (then_helper args...) : (else_helper args...)
```

### Phase 5: Simplify later stages

After flat-core desugaring is in place, simplify subsequent stages:

- typecheck: remove sugar-specific cases, check only flat core constructs
- lowering: structural translation only, minimal heuristics
- path resolution: no ambient/root guessing for model code, lexical and explicit resolution

### Phase 6: Continue with the remaining reimplement-Axon agenda

Once phases 1-4 are stable, continue executing the numbered plan below:

- primitive contracts/spec layer
- formal type system and constraint-based checker
- coercion policy
- typed primitive schemas
- further lowering cleanup

## 1. Primitive Contract / Spec Layer

Define a machine-readable contract for every primitive:

- Preconditions:
  - input types, ranks, shapes, dtypes, and value-domain constraints
- Postconditions:
  - output type, rank, shape, dtype, and arity guarantees
- Invariants:
  - preserved properties (mask semantics, ordering properties, structural constraints)
- Mathematical semantics:
  - backend-independent equation-level behavior
- Error semantics:
  - deterministic failure classes and expected messages

Drive testing and assertions from these contracts:

- contract-level unit tests
- property-based tests for invariants
- optional runtime debug assertions generated from pre/postconditions

## 2. Desugaring / Normalization Stage

Introduce an explicit `Surface Axon -> Core Axon` transformation pass.

Responsibilities:

- ANF-like flattening:
  - decompose nested expressions into a sequence of assignments/binds
- Type alias expansion:
  - unfold aliases into canonical core types
- `?:` lowering:
  - replace ternary laziness with generated helper definitions per branch
- Call canonicalization:
  - normalize kwargs/defaults into one canonical core call form
- Name/import normalization:
  - rewrite names to explicit canonical references

Goal:

- downstream stages receive a small, uniform, low-sugar core IR

## 3. Scope + Parameter-Path Resolution Stage

Add a dedicated static resolution pass after desugaring.

Responsibilities:

- resolve each identifier to a unique symbol
- resolve module/member references to canonical targets
- resolve parameter paths to typed path IR nodes (e.g. `PathExpr`)
- reject unresolved or ambiguous names before type checking

Goal:

- remove string/path heuristics from checker and lowering

## 4. Formal Core Type System Design

Specify a formal typing system for core IR:

- value types:
  - `Int`, `Float`, `Bool`, `String`, `Null`, optional, tuples, lists
- tensor types:
  - base tensor kinds + symbolic dimensions
- dimension expressions:
  - arithmetic and symbolic forms
- module signatures and polymorphism
- typing judgments for:
  - expressions, calls, binds, returns, conditionals/branch joins

Goal:

- clear, auditable typing rules with explicit soundness intent

## 5. Strict Constraint-Based Type/Shape Checker

Implement a checker/solver over core IR.

Responsibilities:

- infer and validate:
  - dimensions, ranks, arities, tuple/list structure, return arity
- enforce:
  - shape constraints from signatures and primitive contracts
- branch typing:
  - consistent join rules and refinement behavior
- diagnostics:
  - deterministic error classes + source spans

Goal:

- remove ad-hoc rule fragments and special-case checking paths

## 6. Formal Coercion Policy (After Formal Types)

Define coercions only after the formal type system and checker are in place.

Responsibilities:

- specify a minimal coercion lattice
- encode coercions centrally in typing rules
- remove implicit/op-local coercion hacks

Goal:

- explicit, uniform coercion behavior with no hidden conversions

## 7. Typed Primitive-Op Schemas

Represent primitives declaratively in one schema source.

Each primitive schema should include:

- positional/kwarg signature and arity
- input/output typing rules
- shape and dtype propagation rules
- const-foldability and compile-time requirements
- runtime validation hooks

Consumers:

- type checker
- lowering validator
- runtime/codegen assertion layer

Goal:

- one source of truth for primitive semantics and constraints

## 8. Lowering Cleanup ("Dumb Lowering")

Refactor lowering into structural translation from typed core IR.

Responsibilities:

- translate already-typed, already-resolved core nodes
- avoid semantic reconstruction in lowering
- avoid name/path/type/shape heuristics
- assert previously proven invariants instead of re-deriving them

Goal:

- predictable lowering with minimal ad-hoc logic and minimal behavior drift risk
