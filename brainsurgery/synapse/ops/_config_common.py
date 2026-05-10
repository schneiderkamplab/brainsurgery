from __future__ import annotations

import re
from typing import Any, Mapping

from ..axon.ast.path import (
    path_expr_template_text,
    resolve_path_expr_to_key,
    runtime_value_to_path_expr,
)


def _raw_args(node_spec: dict[str, Any]) -> list[Any]:
    raw = node_spec.get("_args")
    if isinstance(raw, list):
        return list(raw)
    if raw is None:
        return []
    return [raw]


def _resolve_scalar_ref(
    value: Any, env: dict[str, Any], symbols: Mapping[str, int | float | bool]
) -> Any:
    if isinstance(value, str):
        if value in env:
            return env[value]
        if value in symbols:
            return symbols[value]
    return value


def _resolve_key(
    *,
    op_name: str,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    symbols: Mapping[str, int | float | bool],
) -> str:
    args = _raw_args(node_spec)
    if not args:
        raise ValueError(f"{op_name} requires key positional arg")
    raw_key = args[0]
    key = _resolve_scalar_ref(raw_key, env, symbols)
    return resolve_path_expr_to_key(key, {**symbols, **env}, op_name=op_name)


def _resolve_key_template(
    *,
    op_name: str,
    key: str,
    env: dict[str, Any],
    symbols: Mapping[str, int | float | bool],
) -> str:
    if "{" not in key and "}" not in key:
        return key
    out: list[str] = []
    i = 0
    while i < len(key):
        ch = key[i]
        if ch == "}":
            raise ValueError(f"{op_name} key template has unmatched '}}': {key!r}")
        if ch != "{":
            out.append(ch)
            i += 1
            continue
        j = key.find("}", i + 1)
        if j < 0:
            raise ValueError(f"{op_name} key template has unmatched '{{': {key!r}")
        name = key[i + 1 : j].strip()
        if not name:
            raise ValueError(f"{op_name} key template has empty placeholder: {key!r}")
        if name in env:
            value = env[name]
        elif name in symbols:
            value = symbols[name]
        else:
            raise ValueError(f"{op_name} key template placeholder {name!r} is not defined")
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(
                f"{op_name} key template placeholder {name!r} must resolve to scalar, got {type(value).__name__}"
            )
        out.append(str(value))
        i = j + 1
    resolved = "".join(out)
    resolved = ".".join(part for part in resolved.split(".") if part)
    if not resolved:
        raise ValueError(f"{op_name} key must resolve to non-empty string")
    return resolved


def _resolve_default(
    *,
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    symbols: Mapping[str, int | float | bool],
) -> tuple[bool, Any]:
    args = _raw_args(node_spec)
    if len(args) < 2:
        return False, None
    raw_default = _resolve_scalar_ref(args[1], env, symbols)
    if isinstance(raw_default, dict) and "_expr" in raw_default:
        return True, model._eval_expr(raw_default, env, symbols)
    return True, raw_default


def _config_root(spec: dict[str, Any]) -> dict[str, Any]:
    model = spec.get("model", {})
    if isinstance(model, dict):
        model_cfg = model.get("config")
        if isinstance(model_cfg, dict):
            return model_cfg
    top_cfg = spec.get("config")
    if isinstance(top_cfg, dict):
        return top_cfg
    return {}


def _config_lookup(config: dict[str, Any], key: str) -> tuple[bool, Any]:
    current: Any = config
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _coerce_int(*, op_name: str, key: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{op_name} key {key!r} expected int, got {type(value).__name__}")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        raw = value.strip()
        if raw and (raw.isdigit() or (raw[0] in {"+", "-"} and raw[1:].isdigit())):
            return int(raw)
    raise ValueError(f"{op_name} key {key!r} expected int, got {type(value).__name__}")


def _coerce_float(*, op_name: str, key: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{op_name} key {key!r} expected float, got {type(value).__name__}")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        raw = value.strip()
        if raw:
            try:
                return float(raw)
            except ValueError:
                pass
    raise ValueError(f"{op_name} key {key!r} expected float, got {type(value).__name__}")


def _coerce_str(*, op_name: str, key: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{op_name} key {key!r} expected string, got {type(value).__name__}")
    return value


def _resolve_config_value(
    *,
    op_name: str,
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    symbols: Mapping[str, int | float | bool],
) -> tuple[str, bool, Any]:
    key = _resolve_key(op_name=op_name, node_spec=node_spec, env=env, symbols=symbols)
    config = _config_root(model.spec)
    found, value = _config_lookup(config, key)
    if found:
        return key, found, value
    has_default, default_value = _resolve_default(
        model=model,
        node_spec=node_spec,
        env=env,
        symbols=symbols,
    )
    if has_default:
        return key, True, default_value
    raise KeyError(f"{op_name} missing required config key: {key}")


__all__ = [
    "_coerce_float",
    "_coerce_int",
    "_coerce_str",
    "_config_lookup",
    "_config_root",
    "_resolve_config_value",
    "_resolve_default",
    "_resolve_key",
    "_resolve_key_template",
    "_resolve_scalar_ref",
]
