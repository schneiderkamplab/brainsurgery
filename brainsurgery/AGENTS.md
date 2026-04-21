# brainsurgery/AGENTS.md

Global policy: `../AGENTS.md`

## Scope

- Applies to all Python subpackages under `brainsurgery/`.

## Import/Export Boundary Rules

1. Modules in the same subpackage may import each other directly.
2. Imports from other subpackages must go through that subpackage's `__init__.py`.
3. Cross-subpackage/public symbols:
   - public names have no leading `_`
   - explicitly exported via `__all__`
4. Internal-only symbols should use a leading `_`.

## Change Discipline

- Avoid compatibility shims unless explicitly requested.
- Prefer shared reusable abstractions over repeated special-case logic.
- If a module becomes very large, split it along coherent responsibility boundaries.
