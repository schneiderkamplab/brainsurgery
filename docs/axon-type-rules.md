# Axon Type Rules

This document records the implemented typing rules for the active Axon
pipeline.  It describes `typecheck2`, not the removed legacy typechecker and
not an aspirational type system.

The implementation lives primarily in:

- `brainsurgery/synapse/axon/typecheck2/core.py`
- `brainsurgery/synapse/axon/typecheck_shared.py`
- `brainsurgery/synapse/ops/*.py`

## Scope

These rules apply to the flat, closed Axon AST after:

- parse
- load
- materialize, when a materialized artifact is requested
- resolve
- validate
- normalize
- elaborate
- flatten

and before:

- Graph IR lowering
- Graph IR validation
- `codegen2-*`, `runtime2-*`, and `pipeline2-*` backends

Assumed input invariants:

- the program is closed from the selected MAIN module
- unreachable definitions have already been pruned or are pruned by typecheck2
- imports have been resolved
- type aliases are valid and have valid arity
- path sugar parameters have been desugared to ordinary `Path`-typed parameters
- statement-level `scope`, `for`, and `if` are absent from flat input
- calls that depend on optional/default parameters have already been elaborated
- unresolved callees are errors, except for registered primitive ops

Typecheck2 validates flat input before checking and validates the fully typed
output before returning it.

## Type Language

The implemented type language includes:

- `Any`
- `Bool`
- `Dim`
- `Float`
- `Int`
- `Null`
- `Path`
- `String`
- `List[T]`
- tuples `(T1, ..., Tn)`
- optionals `?T`
- tensor types `Tensor[d1, ..., dn]`
- tensor sub-bases such as `IdxTensor[...]`
- named type aliases, e.g. `CacheLayer[B,H,T,DH]`
- type variables, usually generated from unknown named type positions

Dimension tokens may be:

- integer literals
- symbolic names such as `B`, `S`, `MODEL_DIM`, or `DH`
- variadic dimension binders such as `..S`
- symbolic binary expressions over `+`, `-`, `*`, and `/`

Type aliases are expanded during unification and type normalization.  Aliases
may have fixed or variadic dimension parameters.

## Main-Module Checking

Typecheck2 is MAIN-module based.

The checker:

1. resolves the effective MAIN module
2. computes reachability from that MAIN module
3. prunes unreachable definitions
4. validates the flat program
5. checks demanded definitions from MAIN
6. typechecks any remaining reachable definitions in generic mode when needed
7. canonicalizes generated helper signatures and equivalent dimension names
8. validates the typed program
9. repeats the whole pass up to a small fixed bound until the typed AST reaches
   a deterministic fixpoint

This means standalone exported builtin definitions are not independently
checked unless they are reachable from the active MAIN module or checked by a
separate builtin-oriented test mode.

## Typing Environment

The checker environment contains:

- term bindings from definition parameters and earlier bind statements
- ordinary `Path` bindings from desugared path parameters
- dimension-name bindings from symbols appearing in parameter and return types
- module definitions
- type aliases
- expression definitions for simple aliases and constants
- type-variable substitutions
- dimension substitutions

There is no separate exported solver object in the active typecheck2 result.
Symbolic facts are currently represented by substitutions, inferred expression
types, inferred dimensions, and any constraints already present on definitions.
Some older helper functions can collect `Constraint` objects, but the active
typecheck2 path does not rely on a complete public constraint store.

## Expression Metadata

Every typed expression receives:

- `inferred_type`
- `inferred_arity`
- `inferred_dims` when the inferred type is tensor-shaped

Typed validation rejects missing arity metadata and rejects tensor expressions
whose inferred dimensions are absent.  Tuple arity must match tuple item count.

## Base Expression Rules

Names are typed by environment lookup.  An unbound name is a typecheck error.

Literals are initially typed as:

- integer literal: `Int`
- float literal: `Float`
- boolean literal: `Bool`
- string literal: `String`
- `null`: `Null`
- path literal: `Path`

Lists are typed by joining item types.  Empty lists have item type `Any`.
Tuples are typed as tuple types with one item type per element.

Parentheses preserve the inner type and arity.

Type ascriptions are checked, not erased:

```axon
(expr :: T)
```

The checker infers the inner expression type, unifies it with `T`, and keeps
the ascription alive in the typed AST.  Numeric literals inside ascriptions or
typed contexts may be retagged to the expected numeric type.

## Numeric Unification and Literal Retagging

The implementation treats numeric scalar unification as follows:

- `Int` with `Int` yields `Int`
- `Float` with `Float` yields `Float`
- `Dim` with `Dim` yields `Dim`
- `Int` with `Float` yields `Float`
- `Dim` with `Float` yields `Float`
- `Int` with `Dim` yields `Dim`

This is broader than a pure `Dim -> Int` coercion rule.  In practice, integer
literals are often retagged to `Dim` or `Float` when the surrounding expected
type demands it, for example in `List[Dim]` shape arguments or floating-point
keyword defaults.

## Unification

Unification first applies substitutions and expands type aliases.

Implemented cases:

- type variables bind with an occurs check
- `Any` unifies with the other side and returns the other side
- numeric scalar cases follow the numeric rules above
- `Null` combined with a non-null type produces an optional type
- `?A` and `?B` unify by unifying `A` and `B`
- `?A` and `B` unify as `?unify(A, B)`
- lists unify itemwise
- tuples unify itemwise and must have equal arity
- tensors unify by compatible base and dimension sequence
- named types unify when names and arguments match, or via alias expansion

Tensor base compatibility allows the generic base `Tensor` to unify with more
specific tensor bases.  Incompatible non-generic bases are rejected.

## Dimension Unification

Dimension unification normalizes substitutions before comparison.

Implemented behavior:

- identical dimension tokens unify directly
- two symbolic names unify by choosing the more readable/preferred name
- a symbolic name may bind to a literal, another symbol, a tuple of variadic
  dims, or a symbolic expression
- recursive dimension bindings are rejected
- equal integer literals unify; unequal integer literals are a mismatch
- binary dimension expressions with the same operator unify recursively
- simple arithmetic identities are simplified, including neutral elements and
  some inverse patterns such as `(H * DH) / H = DH`

Dimension sequence unification supports at most the implemented variadic
patterns.  `Tensor[..S]` can match a whole concrete shape, preserving prefix
and suffix dimensions where declared.  Multiple variadic binders in positions
that cannot be matched deterministically are rejected.

Generated dimensions are later canonicalized.  The checker prefers readable
existing names over generated names where equivalence can be established from
substitutions or resolvable dimension aliases.

## Dimension Aliases and Constants

Simple expression definitions can serve as dimension aliases.

Examples:

```axon
MODEL_DIM = H * DH
HEAD_DIM = MODEL_DIM / H
```

When a type contains `MODEL_DIM`, typecheck2 may resolve it to the underlying
dimension expression for equivalence checks and may later canonicalize back to
the shorter or more meaningful name.  The alias resolver handles names,
integer literals, zero-argument calls, arithmetic expressions, and statically
known ternaries.

## Calls

All non-primitive user and builtin Axon calls use the same general mechanism.
Builtin wrappers are not special-cased by name.

For a call:

```axon
f a1 ... an kw1=v1 ... kwm=vm
```

typecheck2:

1. looks up `f` as a module definition
2. instantiates the callee signature with fresh type and dimension variables
3. consumes explicit path arguments first when the callee has desugared path
   parameters
4. binds positional and keyword arguments to formal parameters
5. typechecks each actual expression
6. unifies actual types with formal types
7. records dimension bindings from actual expressions when a formal parameter
   is dimension-typed
8. typechecks the callee body under the call-site environment when safe
9. uses the call-site-specialized return type when it is more precise than the
   declared header type
10. restores caller substitutions so callee-local inference does not leak
    arbitrarily into the caller

Missing required arguments are errors and the diagnostic explicitly says to
run elaborate before flatten/typecheck2.  Too many positional arguments are
errors.

Current behavior for unknown keyword arguments is conservative but permissive:
known keyword arguments are checked against formals; unknown keyword entries
are carried through in the typed call rather than rejected by this path.  Other
validators or lowering may still reject them.

## Path Parameters

Flat Axon has no callee path sugar and no path-sugar parameters.  Paths are
ordinary `Path`-typed arguments.

Typecheck2 still recognizes two path-argument sources:

- explicit `module.path_params`
- ordinary leading parameters whose declared type is `Path`

This supports the current desugared flat representation.  A call missing a
required path argument is a typecheck error.

## Primitive Calls

Primitive calls are identified through the primitive-op registry in
`brainsurgery/synapse/ops`.

A primitive module must expose:

- `OP_NAME`
- `LOWERING_TYPE_SIGNATURE`

It may also expose:

- `type_rule(...)`

For a primitive call, typecheck2:

1. looks up the primitive by normalized op name
2. checks positional arguments against `LOWERING_TYPE_SIGNATURE["args"]`
3. checks known keyword arguments against `LOWERING_TYPE_SIGNATURE["kwargs"]`
4. rejects explicit `null` when the primitive signature requires a non-optional
   type
5. invokes the primitive `type_rule` if present
6. otherwise derives the return type from `LOWERING_TYPE_SIGNATURE["returns"]`
7. treats `"dynamic"` return as "same as the first argument" when possible

Primitive-specific behavior belongs in primitive `type_rule` hooks, not in the
general call checker and not in model-specific compiler logic.

Implemented primitive rule categories include:

- elementwise/broadcasting tensor rules, e.g. `_add`, `_mul`, `_where`, `_eq`,
  `_le`, logical operations
- axis and shape transforms, e.g. `_reshape`, `_expand`, `_permute`,
  `_transpose`, `_chunk`, `_split`, `_concat`, `_slice`, `_repeat`,
  `_unsqueeze`, `_sum`, `_topk`
- tensor creation, e.g. `_zeros`, `_empty`, `_full`, `_tensor_like`,
  `_zeros_like`, `_empty_like`
- indexing and sequence operations, e.g. `_gather`, `_scatter`,
  `_index_add`, `_list_init`, `_list_append`, `_list_index`, `_list_length`
- neural-network primitives, e.g. `_embedding`, `_linear`,
  `_expert_linear`, `_layernorm`, `_rmsnorm`, activations, `_softmax`,
  `_matmul`
- configuration and parameter primitives, e.g. `_config_*`, `_params_*`,
  `_param`

## Binary Operators

Comparison operators are:

```text
== != < <= > >=
```

For tensor comparisons, tensor shapes are broadcast when possible and the
result is tensor-shaped.  For non-tensor comparisons, operands are unified and
the result is `Bool`.

Boolean operators are:

```text
and or
```

They require boolean operands and return `Bool`.

Arithmetic operators are:

```text
+ - * /
```

Tensor arithmetic broadcasts tensor dimensions when possible.  Tensor-scalar
arithmetic preserves the tensor shape.  Scalar arithmetic follows the numeric
unification rules above.

## Branches and Null Refinement

Expression-level ternaries and `if` expressions require a boolean condition.

The branch result type is a join:

- tuple branches join itemwise
- optional branches join inner types and preserve optionality
- tensor branches join equivalent dimensions, fall back to broadcast joining
  where appropriate, and otherwise fail through normal unification
- other branches unify normally

Typecheck2 also performs limited branch environment refinement for null
checks:

```axon
x == null
x != null
```

If `x : ?T`, then the null branch treats `x` as `Null` and the non-null branch
treats `x` as `T`.

Some statically resolvable boolean and null comparisons are evaluated for
choosing a branch during type inference.  This is typechecking behavior, not a
general optimizer.

## Returns and Multi-Target Binds

Multiple returned values are represented by tuple types.  Return/yield
statements are checked against the expected definition return type when one is
available.

Return checking preserves protected header dimension names where possible.
This prevents a literal or generated call-site dimension from overwriting a
meaningful signature dimension unless the program actually constrains them to
be equal.

Multi-target binds destructure:

- tuples of matching arity
- lists by repeating the list item type for each target
- unconstrained type variables by binding them to a tuple of fresh type
  variables

Destructuring a non-tuple/non-list/non-type-variable value is an error.

This is why Axon can typecheck source patterns such as:

```axon
q, k, v <- Tensor.chunk qkv parts=3
```

when `Tensor.chunk` is typed as returning a `List[...]` or a tuple-like
structure with compatible item type.

## Recursive and Generated Helpers

Generated loop and branch helpers are ordinary definitions.  There is no
separate compatibility typing mode for helpers.

Implementation details:

- recursive calls use the current instantiated signature when a module is
  already active
- non-recursive calls may typecheck the callee body under the call-site
  environment to refine returns
- generated helper signatures are normalized and canonicalized after checking
- equivalent generated dimensions may be renamed to deterministic
  `__gdimN` names when no better source name is available
- readable non-generated dimension names are preserved when they are protected
  by a signature or environment
- typecheck2 repeats up to five passes and returns once the typed AST reaches
  a fixed point; short deterministic cycles are broken by choosing the
  structurally minimal rendering

Helpers must not rely on backend fallbacks or model-specific behavior.  If a
helper cannot be represented as a normal typed Axon definition, typecheck2
should fail.

## Fresh Names and Canonicalization

Fresh type variables use names such as `__tcN` during checking.  Fresh
dimension variables use names such as `__dN` during local inference and may be
canonicalized to `__gdimN` in generated signatures.

The checker does not choose fresh names arbitrarily when a better semantic
name is available.  It prefers readable existing names, especially names from
headers, call-site arguments, and closed dimension aliases.  This improves
rendered typed Axon and downstream diagnostics.

## Any

`Any` is still part of the implemented type language.  It is used for genuinely
unknown generic positions and for some broad primitive signatures.

Rules:

- `Any` unifies with the other side and yields the other side
- primitive rules should refine `Any` whenever enough operand information is
  available
- compiler stages should not introduce `Any` as a compatibility fallback for
  unresolved callees or model-specific failures

## Current Limitations

These are current implementation facts, not target rules:

- typecheck2 does not currently expose a complete public constraint store over
  dims, ints, bools, and nullability
- unknown keyword arguments on user calls are not rejected by the main call
  binding path
- `Null` unification is permissive for general user-call typing, while
  primitive calls explicitly reject `null` for non-optional primitive
  parameters
- some primitive signatures remain intentionally broad and rely on type-rule
  hooks for precision
- optimization is not part of the correctness-critical typechecking contract

When these limitations are tightened in code, this document should be updated
in the same change.
