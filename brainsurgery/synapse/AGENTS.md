# brainsurgery/synapse/AGENTS.md

Global policy: `../../AGENTS.md`

## Scope

- Applies to parser/typechecker/lowering/codegen/runtime, builtins, ops, and model axons.

## Rules of Engagement

- Prefer generic reusable fixes over model-specific hacks.
- Model-specific special casing is strictly limited to HF loading/config paths.
- No model-specific branching in parser, lowering, codegen, runtime, or core builtins without explicit approval.
- HF namespace/path quirks belong only in integration/loading entrypoints, not in compiler/runtime/builtins.
- Core builtins must not define model-specific absolute default paths (`@@...`) in signatures.

## Approval Gates

- Required approval:
  - semantic changes to compiler/runtime behavior
  - changes likely to alter fidelity/perf across many models
  - compatibility shims
- Allowed without extra approval:
  - conservative bug fixes
  - diagnostics improvements
  - dead-code cleanup with no behavior change

## Explicitly Unwanted

- Silent fallbacks that hide path/type errors.
- Duplicating logic across builtins when a shared definition is possible.
- HF/model-family string routing logic in `axon/*`, runtime2/pipeline2/codegen2 paths, or `builtins/*.axon`.
