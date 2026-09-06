#!/usr/bin/env python3
"""Frozen safetensors persistence adapter for torch-state-bridge case R01."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path

from safetensors.torch import load_file, save_file
from torch_state_bridge import state_bridge, state_bridge_preview

EXPECTED_VERSION = "0.1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    version = importlib.metadata.version("torch-state-bridge")
    if version != EXPECTED_VERSION:
        raise SystemExit(f"expected torch-state-bridge {EXPECTED_VERSION}, found {version}")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    source = load_file(str(args.input), device="cpu")
    rules = args.rules.read_text(encoding="utf-8")
    mapping, _unchanged, collisions = state_bridge_preview(source, rules)
    if collisions:
        raise RuntimeError(f"preview found collisions: {sorted(collisions)}")
    changed = sum(old != new for old, new in mapping.items())
    if changed < 1:
        raise RuntimeError("expected at least one renamed key")
    transformed = state_bridge(source, rules, detect_collision=True)
    args.output.parent.mkdir(parents=True)
    save_file(transformed, str(args.output))
    print(f"renamed {changed} of {len(source)} keys with torch-state-bridge {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
