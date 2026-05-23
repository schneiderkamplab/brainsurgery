---
status: active
last-confirmed: 2026-05-20
owners: agents
confidence: high
---

# Axon Compiler Policy

This page records non-negotiable compiler/runtime policy for Axon stages.

Validated-by: root `AGENTS.md` and project direction as of 2026-05-20.

## No Definition-Name Special-Casing

Compiler and runtime stages must not give special treatment to ordinary Axon definitions by name.

This applies to:

- parser and normalize,
- elaborate and flatten,
- validate and typecheck2,
- AST optimization,
- Graph IR lowering,
- Graph IR optimization,
- Graph IR rendering,
- codegen2 backends,
- runtime2 execution helpers,
- core builtins implemented in Axon.

Forbidden examples:

- `if callee == "Config.dim"` in typecheck, optimize, lowering, or codegen.
- `if module_name.startswith("NN.")` in generic compiler logic.
- Treating `Tensor.size`, `Cache.past_length`, `Attention.reshape_heads`, `NN.embedding`, or any “wrapper” differently because of its definition name.
- Model-family routing or HF namespace quirks in compiler/runtime layers.

Correct approach:

- Encode operation semantics on primitives, not on wrapper/module names.
- Let ordinary Axon definitions typecheck, inline, specialize, lower, and codegen through the same generic rules as every other definition.
- If a wrapper needs better behavior, fix its Axon signature/body or the primitive type/effect/usage metadata it calls.
- If a generic stage needs more information, extend the stage contract or IR metadata generically.

## Primitive Semantics Are Allowed

Primitive operations may have semantic metadata and type rules because they are the compiler/runtime boundary.

Allowed examples:

- primitive type rules for `_reshape`, `_expand`, `_chunk`, `_linear`, `_embedding`, `_tensor_size`;
- primitive purity/effect/usage metadata;
- primitive lowering/codegen implementations.

Constraint: primitive rules must be generic and operation-level. They must not branch on model family, checkpoint namespace, or a normal Axon wrapper definition.

## Model-Specific Handling Boundary

Model-specific special casing is allowed only in HF loading/config-adaptation integration paths listed by root `AGENTS.md`.

It is not allowed in:

- `brainsurgery/synapse/axon/*`,
- `brainsurgery/synapse/builtins/*.axon`,
- `brainsurgery/synapse/ops/*` except for generic primitive semantics,
- `runtime.py`, `runtime2`, `pipeline_*`,
- codegen/lowering/typechecker/optimizer modules.

If such special casing is found:

- report it,
- name the target module and offending branch,
- propose a generic replacement,
- remove it only with validation that existing migrated model families still pass.

## Stage Design Rule

Each stage should operate from syntax, types, primitive metadata, graph structure, constraints, domains, purity/usage, and path semantics.

It should not infer meaning from ordinary definition names.

Definitions such as `Config.dim`, `NN.linear`, `Attention.attention`, `Cache.update`, and `Tensor.size` are just Axon definitions unless they are primitive operations with leading underscore names or otherwise explicitly represented as primitives in the IR.

## Review Checklist

Before landing compiler/runtime changes, check:

- Does this branch inspect an ordinary Axon definition name?
- Could the same behavior be derived from primitive metadata, type rules, effect/usage analysis, constraints, or graph shape?
- Would a renamed but equivalent Axon definition still work?
- Does the change preserve model-family independence?
- Is there a test guarding the generic behavior rather than one checkpoint-specific case?
