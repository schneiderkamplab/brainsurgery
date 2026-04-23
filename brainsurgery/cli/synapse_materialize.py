from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

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
    return replace(
        ast,
        pragmas=pragmas,
        imported_members=dict(ast.imported_members),
        constants=dict(ast.constants),
        type_aliases=dict(ast.type_aliases),
    )


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
        out_path.write_text(rendered, encoding="utf-8")
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
