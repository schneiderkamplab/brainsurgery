#!/usr/bin/env python3
"""Export text-only competing-tool records with local identifiers redacted."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

TEXT_SUFFIXES = {".json", ".md", ".txt", ".yaml", ".yml"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-nonreportable",
        action="store_true",
        help="Permit export of a record that failed reporting gates (never for paper results)",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sanitize_value(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            key: "<REDACTED_HOSTNAME>"
            if key == "hostname"
            else sanitize_value(item, replacements)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_value(item, replacements) for item in value]
    if isinstance(value, str):
        for private, public in sorted(
            replacements.items(), key=lambda item: len(item[0]), reverse=True
        ):
            value = value.replace(private, public)
    return value


def build_replacements(run_root: Path, arguments: dict[str, Any]) -> dict[str, str]:
    replacements = {
        str(Path.home()): "<USER_HOME>",
        str(run_root): "<RUN_ROOT>",
    }
    source_model = arguments.get("source_model")
    if source_model:
        replacements[str(Path(source_model).absolute())] = "<SOURCE_MODEL>"
    for key, label in (
        ("brainsurgery_cli", "<BRAINSURGERY_ENV>"),
        ("mergekit_cli", "<MERGEKIT_ENV>"),
        ("torch_state_bridge_python", "<TORCH_STATE_BRIDGE_ENV>"),
    ):
        executable = arguments.get(key)
        if executable:
            replacements[str(Path(executable).absolute().parent.parent)] = label
    return {private: public for private, public in replacements.items() if private != "/"}


def export_records(
    run_root: Path, output_root: Path, *, allow_nonreportable: bool = False
) -> dict[str, Any]:
    run_root = run_root.resolve()
    output_root = output_root.absolute()
    if not run_root.is_dir():
        raise ValueError(f"input run directory does not exist: {run_root}")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite export directory: {output_root}")
    if run_root == output_root or run_root in output_root.parents:
        raise ValueError("output must not be inside the input run directory")
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    if not summary.get("reported_eligible") and not allow_nonreportable:
        raise ValueError("refusing to export a non-reportable run without override")
    arguments = json.loads((run_root / "arguments.json").read_text(encoding="utf-8"))
    replacements = build_replacements(run_root, arguments)
    output_root.mkdir(parents=True)
    copied = []
    for source in sorted(run_root.rglob("*")):
        if not source.is_file() or source.suffix.lower() not in TEXT_SUFFIXES:
            continue
        relative = source.relative_to(run_root)
        target = output_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        text = source.read_text(encoding="utf-8")
        if source.suffix.lower() == ".json":
            value = sanitize_value(json.loads(text), replacements)
            text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        else:
            text = sanitize_value(text, replacements)
        target.write_text(text, encoding="utf-8")
        copied.append(relative.as_posix())
    forbidden = sorted(replacements, key=len, reverse=True)
    leaks = []
    for target in sorted(output_root.rglob("*")):
        if not target.is_file():
            continue
        text = target.read_text(encoding="utf-8")
        for private in forbidden:
            if private in text:
                leaks.append(f"{target.relative_to(output_root)} contains {private!r}")
    if leaks:
        raise RuntimeError("anonymization leak check failed: " + "; ".join(leaks))
    manifest = {
        "export_kind": "anonymized_text_metadata_only",
        "protocol_id": summary.get("protocol_id"),
        "run_id": summary.get("run_id"),
        "reported_eligible": summary.get("reported_eligible"),
        "excluded_binary_payloads": True,
        "files": {
            relative: sha256_file(output_root / relative) for relative in copied
        },
        "warning": (
            "Automated redaction covers recorded local roots and hostname. "
            "A human anonymity review is still required before submission."
        ),
    }
    (output_root / "ANONYMIZED_EXPORT_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    args = parse_args()
    manifest = export_records(
        args.input, args.output, allow_nonreportable=args.allow_nonreportable
    )
    print(f"exported {len(manifest['files'])} text records to {args.output}")
    print(manifest["warning"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
