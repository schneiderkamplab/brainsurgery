# brainsurgery/synapse/ops/AGENTS.md

Global policy: `../../../AGENTS.md`
Synapse policy: `../AGENTS.md`

## Scope

- Python primitive op implementations (`_xyz`) and runtime-facing low-level behavior.

## Allowed Changes

- Correctness fixes aligned with existing builtin semantics.
- Better error diagnostics for missing paths, bad kwargs/pargs, and shape/type issues.
- Conservative performance fixes that preserve numerical behavior.

## Requires Approval

- Changes with broad numerical impact across model families.
- New fallback behavior (CPU/device/path/etc.) that alters failure semantics.

## Unwanted Changes

- Silent fallback logic masking true errors.
- Adding model-specific conditionals in primitive implementations.
