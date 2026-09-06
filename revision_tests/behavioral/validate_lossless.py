#!/usr/bin/env python3
"""Independently verify the frozen behavioral checkpoint is exact and sharded."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from revision_tests.scaling.oracle import compare_output, verify_huggingface_revision

EXPECTED_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
SHARD_BYTES = 256 * 1024 * 1024


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--transformed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    revision = verify_huggingface_revision(args.source, EXPECTED_REVISION)
    comparison = compare_output(
        args.source,
        args.transformed,
        target_regex=r"wte\.weight",
        factor=1.0,
        shard_size_bytes=SHARD_BYTES,
    )
    result = {
        "protocol_id": "eacl2027_behavioral_lossless_gpt2_v1",
        "source_revision": revision,
        "comparison": comparison,
        "passed": revision["passed"] and comparison["passed"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"lossless tensor gate: {comparison['tensors_passed']}/"
        f"{comparison['tensors_checked']} exact; passed={str(result['passed']).lower()}"
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
