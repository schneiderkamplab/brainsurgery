---
paths:
  - "tests/**"
  - "test_vllm_changes.py"
---

# Test rules

@../../tests/AGENTS.md

- Failing test: fix the test, or remove an obsolete test with rationale, or
  ask before changing `brainsurgery/*` to make it pass.
- Keep tests deterministic and small. Prefer a focused regression over a broad
  integration assertion.
- Policy guard tests (`test_agents_policy_guards.py`) must stay in place. Do not
  loosen them or add allowlist entries without approval.
- Fixtures in `conftest.py` may download models from HuggingFace. Do not add
  new tests that require a fresh download unless the existing fixtures cover it.
- Run the narrowest relevant file first, then broaden. Use `-n 8 --dist load`
  for parametrized per-model suites. Avoid `--dist loadfile`.
- The full `pytest -q` run is a pre-commit hook and is slow. Run it only when
  asked or before a commit the user requested.
