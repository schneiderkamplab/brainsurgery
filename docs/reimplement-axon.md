# Reimplement Axon Plan

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
