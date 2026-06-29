from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
import os
import tempfile

from brainsurgery.synapse import (
    ast_equal,
    checkpoint_pragma_entries,
    group_output_name,
    load_materialize_context,
    materialize_axon_file,
    normalize_checkpoint_name,
    parse_axon_program_from_path,
    render_axon_file,
)


def _replace_checkpoints(ast: Any, checkpoints: list[str]) -> Any:
    pragmas = dict(ast.pragmas)
    pragmas["checkpoints"] = checkpoints if len(checkpoints) != 1 else checkpoints[0]
    tokenizer = _tokenizer_pragma_for_checkpoints(pragmas.get("tokenizer"), checkpoints)
    if tokenizer is None:
        pragmas.pop("tokenizer", None)
    else:
        pragmas["tokenizer"] = tokenizer
    return replace(
        ast,
        pragmas=pragmas,
        imported_members=dict(ast.imported_members),
        type_aliases=dict(ast.type_aliases),
    )


def _pragma_occurrences(value: object) -> tuple[object, ...]:
    if isinstance(value, dict) and set(value) == {"__pragma_occurrences__"}:
        occurrences = value["__pragma_occurrences__"]
        if isinstance(occurrences, list | tuple):
            return tuple(occurrences)
    return (value,)


def _tokenizer_pragma_for_checkpoints(
    raw: object,
    checkpoints: list[str],
) -> object | None:
    if raw is None:
        return None
    selected: list[list[str]] = []
    globals_: list[str] = []
    checkpoint_set = set(checkpoints)
    for occurrence in _pragma_occurrences(raw):
        if isinstance(occurrence, str) and occurrence:
            globals_.append(occurrence)
            continue
        if (
            isinstance(occurrence, list | tuple)
            and len(occurrence) == 2
            and all(isinstance(item, str) and item for item in occurrence)
        ):
            checkpoint, tokenizer = str(occurrence[0]), str(occurrence[1])
            if checkpoint in checkpoint_set:
                selected.append([checkpoint, tokenizer])
            continue
        raise ValueError(f"Unsupported TOKENIZER pragma while materializing: {raw!r}")
    if selected:
        return selected[0] if len(selected) == 1 else selected
    if globals_:
        unique = sorted(set(globals_))
        if len(unique) != 1:
            raise ValueError("conflicting global TOKENIZER pragmas while materializing")
        return unique[0]
    return None


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def run_axon_materialize_workflow(
    *,
    axon_path: Path,
    checkpoints: list[str] | None = None,
    models_root: Path = Path("models"),
) -> list[Path]:
    resolved_axon = axon_path.resolve()
    resolved_models_root = models_root.resolve()
    if not resolved_axon.exists():
        raise FileNotFoundError(f"Axon file not found: {resolved_axon}")

    parsed = parse_axon_program_from_path(resolved_axon)
    declared = checkpoint_pragma_entries(parsed.pragmas)
    requested = list(checkpoints or declared)
    if not requested:
        raise ValueError(f"No CHECKPOINTS pragma entries found in {resolved_axon}")

    grouped: list[tuple[Any, list[str]]] = []
    for checkpoint in requested:
        context = load_materialize_context(checkpoint=checkpoint, models_root=resolved_models_root)
        materialized = materialize_axon_file(parsed, context=context)
        for group_ast, group_checkpoints in grouped:
            if ast_equal(group_ast, materialized):
                group_checkpoints.append(checkpoint)
                break
        else:
            grouped.append((materialized, [checkpoint]))

    written: list[Path] = []
    expected: set[Path] = set()
    stale_candidates: set[Path] = set()
    for body_ast, body_checkpoints in grouped:
        out_name = f"{group_output_name(body_checkpoints)}.axon"
        out_path = resolved_axon.parent / out_name
        rendered = render_axon_file(_replace_checkpoints(body_ast, body_checkpoints))
        _atomic_write_text(out_path, rendered)
        expected.add(out_path.resolve())
        written.append(out_path)
        for checkpoint in body_checkpoints:
            stale_candidates.add(
                (resolved_axon.parent / f"{checkpoint.split('/')[-1]}.axon").resolve()
            )
            stale_candidates.add(
                (resolved_axon.parent / f"{normalize_checkpoint_name(checkpoint)}.axon").resolve()
            )

    for stale_path in stale_candidates:
        if stale_path not in expected and stale_path.exists():
            stale_path.unlink()

    return written


__all__ = ["run_axon_materialize_workflow"]
