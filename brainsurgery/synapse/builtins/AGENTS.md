# brainsurgery/synapse/builtins/AGENTS.md

Global policy: `../../../AGENTS.md`
Synapse policy: `../AGENTS.md`

## Scope

- Builtins are the supported Axon surface used by models.

## Allowed Changes

- Derived-op refactors that reduce duplication and keep semantics stable.
- Signature cleanup and deprecation removal when callers are migrated.
- Explicit imports/exports and namespace hygiene improvements.

## Requires Approval

- Any change that intentionally changes math semantics, masking behavior, cache behavior, or attention behavior across many models.
- Reintroduction of deprecated primitive bridges.

## Unwanted Changes

- Model-specific branching in builtins.
- Hidden behavior switches that are not reflected in signatures/docs.
- Model-specific absolute default paths (`@@...`) in builtin signatures.
- Repeating the same primitive `_xyz` op across many builtins.

## Primitive Reference Rule

- Each primitive `_xyz` op may be referenced in builtins only once, in exactly one canonical alias/wrapper/derived definition.
- Other builtins should call that canonical definition instead of re-calling the primitive directly.
