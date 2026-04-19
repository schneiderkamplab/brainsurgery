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
