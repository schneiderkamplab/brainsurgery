# brainsurgery/synapse/models/AGENTS.md

Global policy: `../../../AGENTS.md`
Synapse policy: `../AGENTS.md`

## Scope

- Applies to `brainsurgery/synapse/models/*/*.axon`.

## Allowed Changes

- Import cleanups and migration from deprecated builtins to current builtins.
- Conservative fixes for incorrect call signatures, masks, cache usage, or path wiring.
- Materialization/regeneration updates when generic source is the authority.

## Requires Approval

- Introducing model-specific behavior outside HF loading/config.
- Changes that alter semantics across a model family without benchmark validation plan.
- New compatibility wrappers for legacy call patterns.

## Unwanted Changes

- Direct primitive `_op` use in model files.
- Reintroducing deprecated Compat-style calls when non-deprecated builtins exist.
- Copy-pasting equivalent logic across many model files instead of shared builtins.
