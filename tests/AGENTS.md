# tests/AGENTS.md

Global policy: `../AGENTS.md`

## Test Contract

- If tests fail, resolve by one of:
  - fix tests only, or
  - remove/update obsolete tests, or
  - obtain explicit approval before changing code outside `tests/`.

## Allowed Without Extra Approval

- Update assertions for already-approved behavior.
- Remove stale tests that no longer reflect intended behavior (with rationale in commit/message).
- Improve fixture stability and reduce flakiness.

## Requires Approval

- Any main-package (`brainsurgery/*`) change made to satisfy failing tests.
- Broad test-suite policy shifts (mass skips, loosened quality gates).

## Quality Rules

- Keep tests deterministic and minimal.
- Prefer focused regressions over broad brittle integration assertions.
- When removing a test, document why it is obsolete and what covers the behavior now.
- Keep policy guard tests for AGENTS constraints (special-casing boundaries and restricted-layer pattern checks).
