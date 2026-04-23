# Axon Type Rules

This document records the intended typing rules for closed, flat Axon.

It is a language/IR contract document, not a line-by-line description of the
current implementation. Where the implementation is still incomplete, this file
states the target rule.

## Scope

These rules apply to the closed, flat Axon IR after:

- resolve
- validate
- flatten

and before / during:

- typecheck
- optimize re-typecheck iterations
- lowering

Assumed invariants:

- the AST is closed
- there is no import/path/scope sugar left
- control flow is already flattened
- paths are explicit `Path`-typed expressions
- omitted kwargs / optional pargs that survive flattening should already have
  explicit values

## Core Judgment

Typing has the usual form:

```text
Gamma |- e : T
```

where `Gamma` contains:

- value bindings
- module signatures
- primitive op signatures
- file/module constants
- path bindings
- type aliases
- accumulated substitutions and constraints

The result of typecheck is:

- per-expression inferred type
- per-expression inferred dims / arity
- typed module headers
- collected constraints

## Call Rule

All calls follow one uniform rule.

For a call:

```text
f a1 ... an kw1=v1 ... kwm=vm
```

typechecking proceeds as follows.

### 1. Look up the callee scheme

The callee may be:

- a user/module definition
- a builtin Axon definition
- a primitive op

Each callee must provide a callable signature. Primitive ops may additionally
provide an extra `type_rule` hook.

Conceptually:

```text
f : forall alpha, delta. P => T1 -> ... -> Tk -> R
```

where:

- `alpha` are ordinary type variables
- `delta` are dim variables
- `P` are additional constraints
- `T1..Tk` are parameter types
- `R` is the return type

### 2. Freshly instantiate the scheme

Each call site gets fresh type and dim variables.

This avoids accidental sharing between unrelated calls and between recursive
helper instantiations.

### 3. Match actuals to formals

Arguments are matched by the canonical function interface:

- positional args fill params left-to-right
- kwargs fill named params
- duplicate assignment of one param is rejected
- unknown kwargs are rejected
- all required params must be present

After flattening, the preferred invariant is that all defaults are already made
explicit in the AST.

### 4. Typecheck the actuals

Each actual expression is typed under the current environment:

```text
Gamma |- ai : Ai
```

### 5. Unify actuals with parameters

For each supplied argument, unify the actual type with the instantiated
parameter type.

Conceptually:

```text
Ai <= Ti[sigma]
```

where `<=` means "accepted by this parameter position".

In the base rule, this is ordinary unification plus the explicitly allowed
coercions listed below.

### 6. Apply extra callee constraints

After argument matching/unification:

- apply declared module constraints
- apply primitive-op-specific shape/type rules
- accumulate any resulting symbolic constraints

### 7. Produce the instantiated return type

The call result type is the instantiated return type after substitutions and
constraints are applied.

```text
Gamma |- f(a1,...,an,kw...) : R[sigma]
```

## Module Calls

Ordinary module calls should use only the general call rule.

That means:

- no special helper-only typing discipline
- no ad hoc looser fallback for generated helpers
- no compatibility path for optimizer-created modules

If optimize introduces or clones a helper, it must still be representable as an
ordinary callable with a real signature.

### Required Invariant

Every call in typed flat Axon must have a closed instantiated callee signature
at that program point.

If the compiler cannot provide one, typecheck should fail rather than silently
falling back to `Any`.

## Primitive Op Calls

Primitive ops are also ordinary functions, but they may additionally carry
op-specific typing logic that cannot be expressed by a simple signature alone.

So each primitive has:

- a base signature
- optionally a `type_rule(...)` hook

The hook refines the result type and/or emits additional equalities /
constraints after the base call matching has succeeded.

This is the intended place for rules such as:

- `_concat`
- `_matmul`
- `_reshape`
- `_chunk`
- `_embedding`
- `_layernorm`
- `_permute`
- `_transpose`
- `_where`

### Design Rule

Primitive-specific behavior belongs in the primitive's own type rule, not in
the general call checker and not in model-specific logic.

## Return and Destructuring Rules

Multi-result calls are not a separate calling mode. They are ordinary tuple
returns.

So:

- a callee returning multiple values has tuple return type
- a multi-target bind destructures a tuple (or list where explicitly allowed)
- tuple arity must match exactly

Examples:

```text
f :: A -> (B, C)
```

then:

```text
x, y <- f a
```

is valid only if the RHS type is a tuple of arity 2.

Destructuring a non-tuple/non-list value is a type error.

## Optionality and Null

`?T` is part of the type system and should not be encoded by ad hoc term checks.

Base rules:

- `null : Null`
- `null` may unify with `?T`
- a parameter of type `?T` accepts values of type `T` and `Null`
- a parameter of type `T` does not accept `?T` without proof/refinement

Branches can refine nullability:

- in `x == null`, the true branch may assume `x : Null`
- in `x != null`, the true branch may assume `x : T` when `x : ?T`

Flattened ternary/branch guards should feed the constraint store so later
passes can exploit these facts.

## Coercions

Only explicit, centrally defined coercions should exist.

Current intended coercion direction:

- `Dim -> Int` implicit

This supports the common case where symbolic dims are passed to runtime-size
parameters.

Examples:

- tensor shape-derived dims used in integer op parameters
- slice/reshape/size-related runtime arguments

The following should not be implicit unless explicitly adopted and documented:

- `Int -> Dim`
- `Float -> Int`
- `Any -> ...`

If additional coercions are introduced later, they should be added here and
implemented in one central typing path.

## Dims and Symbolic Constraints

Dims are symbolic and are not restricted to literals.

The constraint store should be able to carry symbolic relations involving:

- dims
- ints
- bools
- null-ness

Examples:

```text
K = P + S
D = H * DH
window != null
flag = true
```

The store is allowed to be more expressive than a fully decidable solver. The
consumer of constraints may restrict itself to a decidable fragment.

## Primitive-Rule Categories

The main kinds of primitive typing rules are:

### Elementwise shape agreement

Examples:

- `_add`
- `_mul`
- `_where` (with mask / branch agreement)

Typical rule:

- inputs must have compatible shapes
- output shape follows that same shape

### Axis-transforming tensor rules

Examples:

- `_permute`
- `_transpose`
- `_chunk`
- `_reshape`
- `_concat`

Typical rule:

- output dims are computed from input dims plus op parameters

Examples:

- `_concat(x, y, dim=d)`:
  output axis `d = xd + yd`
- `_chunk(x, parts=n)`:
  result is a tuple/list of tensors whose split axis is constrained by `n`
- `_reshape(x, shape=[...])`:
  output dims come from the provided shape expression

### Linear algebra rules

Examples:

- `_matmul`
- `_linear`
- `_embedding`
- `_layernorm`

Typical rule:

- batch/outer dims are preserved
- contracting dims must match
- result dims are computed from operand structure and explicit parameters

### Structure-preserving rules

Examples:

- `_zeros_like`
- `_empty_like`
- `_softmax`
- activation ops

Typical rule:

- output type/shape follows the input tensor shape

## Constraints from Control Flow

Flat Axon has no statement-level `if`, but ternary guards and flattened helper
structure still provide branch information.

Typecheck should record guard facts where cheap and sound:

- bool guard truth/falsity
- null / non-null branch facts
- equalities / inequalities induced by comparisons when useful

Optimize may then use refreshed constraints after each rewrite iteration.

## Constants

File/module constants are typed like ordinary expressions, in dependency order.

Constant typing should:

- reject cycles
- produce typed constant expressions
- make constant types available to later module typing

## Recursive and Generated Helpers

Recursive SCCs should be typed as ordinary mutually recursive callable groups.

Requirements:

- seed signatures are only an initialization device
- the final typed helper signatures must reflect the converged body typing
- generated helpers must not keep stale placeholder headers if body typing has
  become more precise
- helpers should never degrade to `Any` merely because they were generated by
  flatten or optimize

This area is still an active implementation frontier.

## Current Open Implementation Gap

The main gap relative to this document is:

- some optimizer-generated helpers still degrade to underspecified signatures
  (`Any`)
- repeated optimize + re-typecheck iterations can still drift helper interface
  dims in ways that violate the intended uniform call rule

The correct direction is:

- use one uniform call typing path for modules and helpers
- keep primitive-specific logic in primitive `type_rule` hooks
- reject unresolved helper signatures instead of accepting `Any` fallbacks
