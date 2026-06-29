# docs/AGENTS.md

Global repo policy is defined in `../AGENTS.md`.
This file defines `docs/`-specific collaboration rules.

## Ownership Model

- `docs/` is jointly maintained by user and agent.
- Treat existing user-authored structure and tone as primary constraints.
- Do not rewrite large sections opportunistically; keep edits scoped to the requested change.

## Allowed Changes Without Extra Approval

- Clarifications, consistency fixes, command examples, and stale-reference cleanup.
- Updating usage counts, tables, and links after verified code changes.
- Adding small focused docs for new workflows already implemented in code.

## Changes That Require Explicit Approval

- Renaming/removing major documentation files.
- Changing normative policy language that affects workflow contracts.
- Introducing new compatibility policies or deprecation strategies.

## Documentation Quality Rules

- Commands must be copy/paste-safe.
- Prefer explicit flags and environment variables over implicit defaults.
- Keep benchmark reporting format aligned with root `AGENTS.md` 3-table standard.
- If behavior differs by backend/runtime/codegen, state that explicitly.
