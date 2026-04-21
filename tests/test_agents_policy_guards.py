from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "brainsurgery" / "synapse" / "models"
BUILTINS_DIR = REPO_ROOT / "brainsurgery" / "synapse" / "builtins"


def _model_family_names() -> set[str]:
    return {
        path.name.lower()
        for path in MODELS_DIR.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }


def _restricted_files() -> list[Path]:
    out: list[Path] = []
    out.extend((REPO_ROOT / "brainsurgery" / "synapse" / "axon").rglob("*.py"))
    out.extend((REPO_ROOT / "brainsurgery" / "synapse" / "ops").glob("*.py"))
    out.extend(
        [
            REPO_ROOT / "brainsurgery" / "synapse" / "runtime.py",
            REPO_ROOT / "brainsurgery" / "synapse" / "pipeline_runtime.py",
            REPO_ROOT / "brainsurgery" / "synapse" / "pipeline_backend.py",
            REPO_ROOT / "brainsurgery" / "synapse" / "codegen.py",
        ]
    )
    return out


def test_no_model_family_conditionals_in_restricted_layers() -> None:
    families = _model_family_names()
    allowlisted = {
        REPO_ROOT / "brainsurgery" / "synapse" / "axon" / "tokenization.py",
    }
    conditionish = re.compile(r"(==|!=|startswith\(|endswith\(|\bin\s*\{)")
    quoted = re.compile(r'"([A-Za-z0-9_\-]+)"|\'([A-Za-z0-9_\-]+)\'')
    hits: list[str] = []

    for file_path in _restricted_files():
        if file_path in allowlisted or not file_path.exists():
            continue
        for lineno, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
            if conditionish.search(line) is None:
                continue
            tokens = [a or b for a, b in quoted.findall(line)]
            if any(token.lower() in families for token in tokens):
                hits.append(f"{file_path.relative_to(REPO_ROOT)}:{lineno}:{line.strip()}")

    assert not hits, "Found model-family conditional logic in restricted layers:\n" + "\n".join(
        hits
    )


def test_builtins_do_not_use_absolute_default_paths() -> None:
    bad_default = re.compile(r"\?[A-Za-z_][A-Za-z0-9_]*\s*=\s*@@")
    hits: list[str] = []
    for file_path in sorted(BUILTINS_DIR.glob("*.axon")):
        for lineno, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
            if bad_default.search(line):
                hits.append(f"{file_path.relative_to(REPO_ROOT)}:{lineno}:{line.strip()}")
    assert not hits, (
        "Builtins contain absolute @@ default-path kwargs; "
        "pass paths from callers instead:\n" + "\n".join(hits)
    )


def test_each_builtin_primitive_reference_is_single_source() -> None:
    token = re.compile(r"\b_[a-z][a-z0-9_]*\b")
    counts: Counter[str] = Counter()
    locations: dict[str, list[str]] = {}

    for file_path in sorted(BUILTINS_DIR.glob("*.axon")):
        for lineno, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
            for match in token.findall(line):
                counts[match] += 1
                locations.setdefault(match, []).append(
                    f"{file_path.relative_to(REPO_ROOT)}:{lineno}"
                )

    repeated = {name: locs for name, locs in locations.items() if counts[name] > 1}
    assert not repeated, (
        "Primitive _xyz references must have a single canonical builtin definition:\n"
        + "\n".join(f"{name}: {', '.join(locs)}" for name, locs in sorted(repeated.items()))
    )
