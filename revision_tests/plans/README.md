# Revision test plans

- [`revision_plan.md`](revision_plan.md) maps each reviewer concern to a
  proposed response and keeps out-of-scope items visible without promising
  their implementation.
- [`execution_plan.md`](execution_plan.md) lists work that runs here on macOS
  first, followed by work that requires Linux or CUDA.
- [`linux_handoff.md`](linux_handoff.md) is the ordered receiving-machine
  runbook, including the Codex first-cell transcript gate.
- [`linux_handoff_manifest.json`](linux_handoff_manifest.json) records the
  verified transfer-bundle identity and pinned base-model revisions.
- [`claim_boundaries.md`](claim_boundaries.md) is the manuscript-level gate for
  sharding, distributed systems, performance, usability, and failure-semantics
  claims.

These tracked copies are canonical. Update them here so every machine receives
the same plan through Git; do not maintain divergent copies under `private/`.
