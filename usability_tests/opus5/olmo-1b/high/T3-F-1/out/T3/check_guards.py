"""Negative tests: prove the required checks in solution.py fail loudly.

Each case corrupts the targeting and asserts that solution.py exits non-zero
*before* touching the output directory.  Run from the sandbox root:

    .venv/bin/python out/T3/check_guards.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import solution  # noqa: E402

BASELINE = solution.cast_targets()
SNAPSHOT = sorted(p.name for p in solution.OUT_DIR.iterdir())

CASES = {
    "undercount (111 targets)": BASELINE[:-1],
    "over-broad targeting (embeddings swept in)": [
        "model.embed_tokens.weight",
        *BASELINE[1:],
    ],
    "lm_head swept in": ["lm_head.weight", *BASELINE[1:]],
}

failures = []
for label, targets in CASES.items():
    solution.cast_targets = lambda targets=targets: list(targets)
    try:
        solution.main()
    except solution.CheckFailed as exc:
        print(f"OK   {label}: {exc}")
    else:
        failures.append(label)
        print(f"MISS {label}: no check fired")
    finally:
        solution.cast_targets = solution.__dict__["cast_targets"]

solution.cast_targets = lambda: list(BASELINE)
after = sorted(p.name for p in solution.OUT_DIR.iterdir())
if after != SNAPSHOT:
    failures.append(f"output directory changed during negative tests: {after}")

if failures:
    print(f"GUARD TESTS FAILED: {failures}", file=sys.stderr)
    raise SystemExit(1)
print("all guards fired; output directory untouched")
