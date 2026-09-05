#!/usr/bin/env python
"""Write or verify a checksum manifest of everything a participant or grader sees.

    .venv/bin/python usability_tests/make_manifest.py            # write manifest.sha256
    .venv/bin/python usability_tests/make_manifest.py --verify   # compare against it

Covers, for every target: the base checkpoint files, the generated fine-tunes
and LoRA adapter, the hidden references, and the doc pack. Two machines whose
manifests verify run exactly the same study: same inputs, same references,
same documentation. Paths are relative to this directory and follow symlinks.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOTS = ("inputs", "references", "docpack")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def files() -> list[Path]:
    """Every regular file under the roots, following directory and file symlinks."""
    import os
    out = []
    for root in ROOTS:
        base = HERE / root
        if not base.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(base, followlinks=True):
            dirnames.sort()
            for name in sorted(filenames):
                p = Path(dirpath) / name
                if p.resolve().is_file():
                    out.append(p)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.sha256")
    args = parser.parse_args()
    if args.verify:
        expected = {}
        for line in args.manifest.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                digest, rel = line.split("  ", 1)
                expected[rel] = digest
        bad, missing = [], []
        seen = set()
        for p in files():
            rel = str(p.relative_to(HERE))
            seen.add(rel)
            if rel not in expected:
                continue
            if sha256(p) != expected[rel]:
                bad.append(rel)
        missing = sorted(set(expected) - seen)
        for rel in bad:
            print(f"MISMATCH {rel}")
        for rel in missing:
            print(f"MISSING  {rel}")
        print(f"verified {len(expected) - len(bad) - len(missing)}/{len(expected)} files"
              + ("" if not (bad or missing) else " -- DO NOT RUN THE STUDY ON THIS COPY"))
        return 1 if (bad or missing) else 0
    lines = ["# sha256 of inputs, references and doc pack; verify with make_manifest.py --verify"]
    for p in files():
        lines.append(f"{sha256(p)}  {p.relative_to(HERE)}")
    args.manifest.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.manifest} ({len(lines) - 1} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
