from __future__ import annotations

from dataclasses import dataclass

from ..ast.nodes import AxonExpr
from ..ast.types import TypeAliasDef, TypeExpr


@dataclass(frozen=True)
class CstSignature:
    definition_decl: str
    type_signature: CstFunctionType


@dataclass(frozen=True)
class CstPathTypeParam:
    name: str | None
    type_expr: TypeExpr


@dataclass(frozen=True)
class CstFunctionType:
    path_params: tuple[CstPathTypeParam, ...]
    arg_types: tuple[TypeExpr, ...]
    return_type: TypeExpr


@dataclass(frozen=True)
class CstDefParam:
    name: str
    default_expr: AxonExpr | None = None


@dataclass(frozen=True)
class CstDefinition:
    definition_decl: str
    args: tuple[CstDefParam, ...]
    rhs: AxonExpr


@dataclass(frozen=True)
class CstGlobalBinding:
    name: str
    rhs: AxonExpr


@dataclass(frozen=True)
class CstDefinitionSource:
    signature: CstSignature | None
    definition: CstDefinition


@dataclass(frozen=True)
class CstProgramSource:
    modules: tuple[CstDefinitionSource, ...]
    global_bindings: tuple[CstGlobalBinding, ...]
    imports: tuple[str, ...]
    imported_members: dict[str, tuple[str, ...]]
    exports: tuple[str, ...]
    pragmas: dict[str, object]
    type_aliases: dict[str, TypeAliasDef]


__all__ = [
    "CstDefParam",
    "CstDefinition",
    "CstFunctionType",
    "CstGlobalBinding",
    "CstDefinitionSource",
    "CstPathTypeParam",
    "CstProgramSource",
    "CstSignature",
]
