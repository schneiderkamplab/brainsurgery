from __future__ import annotations

from typing import Any, Mapping


def _raw_args(node_spec: dict[str, Any]) -> list[Any]:
    raw = node_spec.get("_args")
    if isinstance(raw, list):
        return list(raw)
    if raw is None:
        return []
    return [raw]


def _uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


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
    if not isinstance(key, str) or not key:
        raise ValueError(f"{op_name} key must resolve to non-empty string")
    return key


def _resolve_default(
    *,
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    symbols: Mapping[str, int | float | bool],
) -> tuple[bool, Any]:
    args = _raw_args(node_spec)
    if len(args) < 3:
        return False, None
    raw_default = _resolve_scalar_ref(args[2], env, symbols)
    if isinstance(raw_default, dict) and "_expr" in raw_default:
        return True, model._eval_expr(raw_default, env, symbols)
    return True, raw_default


def _resolve_root(
    *,
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    symbols: Mapping[str, int | float | bool],
) -> str:
    args = _raw_args(node_spec)
    if len(args) < 2:
        return ""
    raw_root = args[1]
    root = _resolve_scalar_ref(raw_root, env, symbols)
    if isinstance(root, dict) and "_expr" in root:
        root = model._eval_expr(root, env, symbols)
    if root is None:
        return ""
    if not isinstance(root, str):
        raise ValueError("config root must resolve to string")
    return root


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
    root = _resolve_root(model=model, node_spec=node_spec, env=env, symbols=symbols)
    full_key = f"{root}.{key}" if root else key
    if root:
        root_found, root_value = _config_lookup(config, root)
        if not root_found:
            found, value = False, None
        elif isinstance(root_value, dict):
            found, value = _config_lookup(root_value, key)
        else:
            found, value = False, None
    else:
        found, value = _config_lookup(config, key)
    if found:
        return full_key, found, value
    has_default, default_value = _resolve_default(
        model=model,
        node_spec=node_spec,
        env=env,
        symbols=symbols,
    )
    if has_default:
        return full_key, True, default_value
    raise KeyError(f"{op_name} missing required config key: {full_key}")


def _compile_value_expr(*, emitter: Any, value: Any, env: dict[str, str]) -> str:
    return emitter._expr_code(value, env)


def _compile_lookup_lines(
    *,
    op_name: str,
    emitter: Any,
    node_spec: dict[str, Any],
    env: dict[str, str],
    indent: str,
) -> tuple[list[str], str, str, str]:
    lines: list[str] = []
    root_var = emitter._fresh("cfg_root")
    lookup_root_var = emitter._fresh("cfg_lookup_root")
    key_var = emitter._fresh("cfg_key")
    full_key_var = emitter._fresh("cfg_full_key")
    found_var = emitter._fresh("cfg_found")
    value_var = emitter._fresh("cfg_value")
    part_var = emitter._fresh("cfg_part")
    root_part_var = emitter._fresh("cfg_root_part")
    args = _raw_args(node_spec)
    if not args:
        raise ValueError(f"{op_name} requires key positional arg")
    key_expr = _compile_value_expr(emitter=emitter, value=args[0], env=env)
    root_expr = _compile_value_expr(
        emitter=emitter, value=args[1] if len(args) >= 2 else "", env=env
    )

    spec = getattr(emitter, "spec", {})
    model = spec.get("model", {}) if isinstance(spec, dict) else {}
    model_config = model.get("config") if isinstance(model, dict) else None
    config_literal = model_config if isinstance(model_config, dict) else {}

    lines.append(f"{indent}{root_var} = {repr(config_literal)}")
    lines.append(f"{indent}if not isinstance({root_var}, dict):")
    lines.append(f"{indent}    {root_var} = {{}}")
    lines.append(f"{indent}{lookup_root_var} = {root_expr}")
    lines.append(f"{indent}if {lookup_root_var} is None:")
    lines.append(f"{indent}    {lookup_root_var} = ''")
    lines.append(f"{indent}elif not isinstance({lookup_root_var}, str):")
    lines.append(f"{indent}    raise ValueError('config root must resolve to string')")
    lines.append(f"{indent}{key_var} = {key_expr}")
    lines.append(f"{indent}if not isinstance({key_var}, str) or not {key_var}:")
    lines.append(
        f"{indent}    raise ValueError({op_name!r} + ' key must resolve to non-empty string')"
    )
    lines.append(
        f"{indent}{full_key_var} = ({lookup_root_var} + '.' + {key_var}) if {lookup_root_var} else {key_var}"
    )
    lines.append(f"{indent}{found_var} = True")
    lines.append(f"{indent}{value_var} = {root_var}")
    lines.append(f"{indent}if {lookup_root_var}:")
    lines.append(f"{indent}    for {root_part_var} in {lookup_root_var}.split('.'):")
    lines.append(
        f"{indent}        if isinstance({value_var}, dict) and {root_part_var} in {value_var}:"
    )
    lines.append(f"{indent}            {value_var} = {value_var}[{root_part_var}]")
    lines.append(f"{indent}        else:")
    lines.append(f"{indent}            {found_var} = False")
    lines.append(f"{indent}            break")
    lines.append(f"{indent}if {found_var}:")
    lines.append(f"{indent}    for {part_var} in {key_var}.split('.'):")
    lines.append(f"{indent}        if isinstance({value_var}, dict) and {part_var} in {value_var}:")
    lines.append(f"{indent}            {value_var} = {value_var}[{part_var}]")
    lines.append(f"{indent}        else:")
    lines.append(f"{indent}            {found_var} = False")
    lines.append(f"{indent}            break")
    return lines, full_key_var, found_var, value_var


def _compile_default_lines(
    *,
    op_name: str,
    emitter: Any,
    node_spec: dict[str, Any],
    env: dict[str, str],
    indent: str,
    found_var: str,
    key_var: str,
    value_var: str,
) -> list[str]:
    lines: list[str] = []
    args = _raw_args(node_spec)
    if len(args) >= 3:
        default_expr = _compile_value_expr(emitter=emitter, value=args[2], env=env)
        lines.append(f"{indent}if not {found_var}:")
        lines.append(f"{indent}    {value_var} = {default_expr}")
        return lines
    lines.append(f"{indent}if not {found_var}:")
    lines.append(
        f"{indent}    raise KeyError({op_name!r} + ' missing required config key: ' + {key_var})"
    )
    return lines


def _unsupported_interpret(
    *,
    op_name: str,
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    node_path: str,
    scope: str,
    symbols: Mapping[str, int | float | bool],
) -> None:
    del model, node_spec, env, node_path, scope, symbols
    raise NotImplementedError(f"{op_name} interpret() is not implemented")


def _unsupported_compile(
    *,
    op_name: str,
    emitter: Any,
    node_spec: dict[str, Any],
    env: dict[str, str],
    node_path_var: str,
    scope_var: str,
    indent: str,
) -> list[str]:
    del emitter, node_spec, env, node_path_var, scope_var, indent
    raise NotImplementedError(f"{op_name} compile() is not implemented")


__all__ = [
    "_coerce_float",
    "_coerce_int",
    "_coerce_str",
    "_compile_default_lines",
    "_compile_lookup_lines",
    "_config_lookup",
    "_config_root",
    "_resolve_config_value",
    "_resolve_default",
    "_resolve_key",
    "_resolve_root",
    "_resolve_scalar_ref",
    "_unsupported_compile",
    "_unsupported_interpret",
    "_uses_node_path",
]
