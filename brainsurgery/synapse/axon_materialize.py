from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

import safetensors
import torch

from .axon.grammar import ParsedDefParam, ParsedProgramSource, ParsedSignature, parse_program_source
from .axon.type_system import render_type
from .axon.types import (
    AxonBind,
    AxonExpr,
    AxonExprBinary,
    AxonExprBind,
    AxonExprBool,
    AxonExprCall,
    AxonExprDo,
    AxonExprFloat,
    AxonExprIf,
    AxonExprInt,
    AxonExprLambda,
    AxonExprList,
    AxonExprName,
    AxonExprNull,
    AxonExprParen,
    AxonExprPath,
    AxonExprPipe,
    AxonExprString,
    AxonExprTernary,
    AxonExprTuple,
    AxonKwargValue,
    AxonRepeat,
    AxonReturn,
    AxonScopeBind,
    AxonStatement,
    AxonYield,
)

_NOT_EVALUABLE = object()
_Number = int | float


def _normalize_checkpoint_name(repo_id: str) -> str:
    name = repo_id.split("/")[-1]
    if name.endswith("-pt"):
        return name[: -len("-pt")]
    if name.endswith("-it"):
        return name[: -len("-it")]
    return name


def _group_output_name(checkpoints: list[str]) -> str:
    names = [checkpoint.split("/")[-1] for checkpoint in checkpoints]
    unique_names = sorted(set(names), key=lambda name: (len(name), name))
    for candidate in unique_names:
        if all(name == candidate or name.startswith(candidate + "-") for name in names):
            return candidate
    normalized_names = {_normalize_checkpoint_name(checkpoint) for checkpoint in checkpoints}
    if len(normalized_names) == 1:
        return next(iter(normalized_names))
    return unique_names[0]


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(payload).__name__}")
    return payload


def _checkpoint_pragma(parsed: ParsedProgramSource) -> list[str]:
    raw = parsed.pragmas.get("checkpoints")
    if isinstance(raw, tuple):
        return [str(item) for item in raw]
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if isinstance(raw, str):
        return [raw]
    return []


def _resolve_config_key_template(key: str, env: dict[str, object], op_name: str) -> str:
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
        if name not in env:
            raise ValueError(f"{op_name} key template placeholder {name!r} is not defined")
        value = env[name]
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


def _resolve_config_path_key(raw: object, env: dict[str, object], op_name: str) -> str:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{op_name} expects one non-empty Path key")
    if raw.startswith("@@"):
        key = raw[2:]
    elif raw.startswith("@"):
        key = raw[1:]
    else:
        raise ValueError(f"{op_name} expects Path key (expected @... or @@...)")
    if not key:
        raise ValueError(f"{op_name} expects one non-empty Path key")
    if len(key) >= 2 and key[0] == "'" and key[-1] == "'":
        key = key[1:-1].replace("\\'", "'").replace("\\\\", "\\")
    return _resolve_config_key_template(key, env, op_name)


def _expr_config_lookup(config: dict[str, object], key: str) -> tuple[bool, Any]:
    value: Any = config
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return False, None
        value = value[part]
    return True, value


def _index_weight_keys(model_dir: Path) -> set[str]:
    out: set[str] = set()
    for name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "model.bin.index.json",
    ):
        path = model_dir / name
        if not path.exists():
            continue
        payload = _load_json(path)
        weight_map = payload.get("weight_map")
        if isinstance(weight_map, dict):
            out.update(str(key) for key in weight_map.keys())
    return out


def _safetensors_weight_keys(model_dir: Path) -> set[str]:
    out: set[str] = set()
    errors: list[tuple[Path, Exception]] = []
    for path in sorted(model_dir.glob("*.safetensors")):
        try:
            st = safetensors.safe_open(str(path), framework="pt")
        except Exception as exc:
            errors.append((path, exc))
            continue
        out.update(str(key) for key in st.keys())
    if not out and errors:
        first_path, first_exc = errors[0]
        raise type(first_exc)(f"{first_path}: {first_exc}")
    return out


def _torch_weight_keys(model_dir: Path) -> set[str]:
    candidates = sorted(model_dir.glob("*.bin")) + sorted(model_dir.glob("*.pt"))
    for path in candidates:
        loaded = torch.load(path, map_location="cpu")
        if isinstance(loaded, dict):
            state = loaded.get("state_dict") if "state_dict" in loaded else loaded
            if isinstance(state, dict):
                return {str(key) for key in state.keys()}
    return set()


def _checkpoint_state_keys(model_dir: Path) -> set[str]:
    indexed = _index_weight_keys(model_dir)
    if indexed:
        return indexed
    safetensor_keys = _safetensors_weight_keys(model_dir)
    if safetensor_keys:
        return safetensor_keys
    torch_keys = _torch_weight_keys(model_dir)
    if torch_keys:
        return torch_keys
    raise FileNotFoundError(f"No checkpoint weights found in {model_dir}")


def _params_has_root(keys: set[str], root: str) -> bool:
    if root == "":
        return True
    prefix = f"{root}."
    return any(key == root or key.startswith(prefix) for key in keys)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_number(value: object) -> _Number:
    if not _is_number(value):
        raise TypeError(f"Expected numeric value, got {type(value).__name__}")
    return cast(_Number, value)


def _expr_from_scalar(value: object) -> AxonExpr:
    def _parse_path_token(raw: str) -> AxonExprPath | None:
        token = raw.strip()
        if not token.startswith("@"):
            return None
        absolute = token.startswith("@@")
        body = token[2:] if absolute else token[1:]
        if not body:
            return None
        if len(body) >= 2 and body[0] == "'" and body[-1] == "'":
            quoted = body[1:-1].replace("\\'", "'").replace("\\\\", "\\")
            parts = tuple(part for part in quoted.split(".") if part)
        else:
            parts = tuple(part for part in body.split(".") if part)
        if not parts:
            return None
        return AxonExprPath(absolute=absolute, parts=parts)

    if value is None:
        return AxonExprNull()
    if isinstance(value, bool):
        return AxonExprBool(value=value)
    if isinstance(value, int) and not isinstance(value, bool):
        return AxonExprInt(value=value)
    if isinstance(value, float):
        return AxonExprFloat(value=value)
    if isinstance(value, str):
        parsed_path = _parse_path_token(value)
        if parsed_path is not None:
            return parsed_path
        return AxonExprString(value=value)
    if isinstance(value, list):
        return AxonExprList(items=tuple(_expr_from_scalar(item) for item in value))
    if isinstance(value, tuple):
        return AxonExprTuple(items=tuple(_expr_from_scalar(item) for item in value))
    raise ValueError(f"Unsupported literal value for materialization: {value!r}")


def _eval_expr(
    expr: AxonExpr,
    *,
    env: dict[str, object],
    config: dict[str, object],
    state_keys: set[str],
    resolve_names: bool = True,
) -> object:
    if isinstance(expr, AxonExprName):
        if expr.name in env:
            return env[expr.name]
        if not resolve_names:
            raise ValueError(f"name resolution disabled: {expr.name}")
        raise ValueError(f"unknown runtime name: {expr.name}")
    if isinstance(expr, AxonExprInt):
        return expr.value
    if isinstance(expr, AxonExprFloat):
        return expr.value
    if isinstance(expr, AxonExprBool):
        return expr.value
    if isinstance(expr, AxonExprNull):
        return None
    if isinstance(expr, AxonExprString):
        return expr.value
    if isinstance(expr, AxonExprPath):
        return expr.to_source()
    if isinstance(expr, AxonExprList):
        return [
            _eval_expr(
                item,
                env=env,
                config=config,
                state_keys=state_keys,
                resolve_names=resolve_names,
            )
            for item in expr.items
        ]
    if isinstance(expr, AxonExprTuple):
        return tuple(
            _eval_expr(
                item,
                env=env,
                config=config,
                state_keys=state_keys,
                resolve_names=resolve_names,
            )
            for item in expr.items
        )
    if isinstance(expr, AxonExprParen):
        return _eval_expr(
            expr.inner,
            env=env,
            config=config,
            state_keys=state_keys,
            resolve_names=resolve_names,
        )
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        cond = _eval_expr(
            expr.cond,
            env=env,
            config=config,
            state_keys=state_keys,
            resolve_names=resolve_names,
        )
        branch = expr.true_expr if bool(cond) else expr.false_expr
        return _eval_expr(
            branch,
            env=env,
            config=config,
            state_keys=state_keys,
            resolve_names=resolve_names,
        )
    if isinstance(expr, AxonExprBinary):
        left = _eval_expr(
            expr.left,
            env=env,
            config=config,
            state_keys=state_keys,
            resolve_names=resolve_names,
        )
        right = _eval_expr(
            expr.right,
            env=env,
            config=config,
            state_keys=state_keys,
            resolve_names=resolve_names,
        )
        op = expr.op
        if op == "+":
            if isinstance(left, str) and isinstance(right, str):
                return left + right
            if _is_number(left) and _is_number(right):
                return _as_number(left) + _as_number(right)
            raise ValueError("binary '+' expects numeric or string operands")
        if op == "-":
            if not _is_number(left) or not _is_number(right):
                raise ValueError("binary '-' expects numeric operands")
            return _as_number(left) - _as_number(right)
        if op == "*":
            if not _is_number(left) or not _is_number(right):
                raise ValueError("binary '*' expects numeric operands")
            return _as_number(left) * _as_number(right)
        if op == "/":
            if not _is_number(left) or not _is_number(right):
                raise ValueError("binary '/' expects numeric operands")
            left_number = _as_number(left)
            right_number = _as_number(right)
            if isinstance(left_number, int) and isinstance(right_number, int):
                if right_number != 0 and left_number % right_number == 0:
                    return left_number // right_number
            return left_number / right_number
        if op == "%":
            if not _is_number(left) or not _is_number(right):
                raise ValueError("binary '%' expects numeric operands")
            return _as_number(left) % _as_number(right)
        if op == "==":
            return left == right
        if op == "!=":
            return left != right
        if op == "<":
            if not _is_number(left) or not _is_number(right):
                raise ValueError("binary '<' expects numeric operands")
            return _as_number(left) < _as_number(right)
        if op == "<=":
            if not _is_number(left) or not _is_number(right):
                raise ValueError("binary '<=' expects numeric operands")
            return _as_number(left) <= _as_number(right)
        if op == ">":
            if not _is_number(left) or not _is_number(right):
                raise ValueError("binary '>' expects numeric operands")
            return _as_number(left) > _as_number(right)
        if op == ">=":
            if not _is_number(left) or not _is_number(right):
                raise ValueError("binary '>=' expects numeric operands")
            return _as_number(left) >= _as_number(right)
        if op == "and":
            return bool(left) and bool(right)
        if op == "or":
            return bool(left) or bool(right)
        raise ValueError(f"unsupported binary operator {op!r}")
    if isinstance(expr, AxonExprCall):
        raw_callee = expr.callee.strip()
        is_absolute_path = "@@" in raw_callee
        parse_callee = raw_callee.replace("@@", "@", 1) if is_absolute_path else raw_callee
        callee_parts = parse_callee.split("@")
        callee = callee_parts[0]
        callee_paths = callee_parts[1:]
        if callee == "has_root":
            callee = "Params.has_root"
        elif callee == "root":
            callee = "Params.root"

        def _eval_kwarg_value(value: AxonKwargValue) -> object:
            if isinstance(value, AxonExpr):
                return _eval_expr(
                    value,
                    env=env,
                    config=config,
                    state_keys=state_keys,
                    resolve_names=resolve_names,
                )
            return value

        args = [
            _eval_expr(
                arg,
                env=env,
                config=config,
                state_keys=state_keys,
                resolve_names=resolve_names,
            )
            for arg in expr.args
        ]
        kwargs = {key: _eval_kwarg_value(value) for key, value in expr.kwargs.items()}

        def _path_token_from_suffix(token: str, *, absolute: bool) -> str:
            raw = token.strip()
            if not raw:
                raise ValueError("empty @path suffix is not allowed")
            return ("@@" if absolute else "@") + raw

        if callee.startswith("Config.") and callee_paths:
            if args:
                raise ValueError(
                    f"{callee} does not allow both positional key and @path suffix key"
                )
            if len(callee_paths) != 1:
                raise ValueError(f"{callee} expects exactly one @path suffix key")
            args = [_path_token_from_suffix(callee_paths[0], absolute=is_absolute_path)]

        if callee in {
            "sqrt",
            "Prelude.sqrt",
            "Math.sqrt",
            "log",
            "Prelude.log",
            "Math.log",
            "exp",
            "Prelude.exp",
            "Math.exp",
            "sin",
            "Prelude.sin",
            "Math.sin",
            "cos",
            "Prelude.cos",
            "Math.cos",
        }:
            if len(args) != 1 or not _is_number(args[0]):
                raise ValueError(f"{callee} expects one numeric argument")
            arg = float(_as_number(args[0]))
            fn_name = callee.split(".", 1)[-1]
            if fn_name == "sqrt":
                return math.sqrt(arg)
            if fn_name == "log":
                return math.log(arg)
            if fn_name == "exp":
                return math.exp(arg)
            if fn_name == "sin":
                return math.sin(arg)
            if fn_name == "cos":
                return math.cos(arg)
            raise ValueError(f"unsupported unary expression call: {callee!r}")
        if callee in {"pow", "Prelude.pow", "Math.pow"}:
            if len(args) != 2 or not _is_number(args[0]) or not _is_number(args[1]):
                raise ValueError("pow expects two numeric arguments")
            return math.pow(float(_as_number(args[0])), float(_as_number(args[1])))
        if callee in {"abs", "Prelude.abs"}:
            if len(args) != 1 or not _is_number(args[0]):
                raise ValueError("abs expects one numeric argument")
            return abs(_as_number(args[0]))
        if callee in {"min", "Prelude.min"}:
            if not args or any(not _is_number(arg) for arg in args):
                raise ValueError("min expects numeric arguments")
            return min(_as_number(arg) for arg in args)
        if callee in {"max", "Prelude.max"}:
            if not args or any(not _is_number(arg) for arg in args):
                raise ValueError("max expects numeric arguments")
            return max(_as_number(arg) for arg in args)
        if callee == "Config.value":
            if kwargs:
                raise ValueError("Config.value does not support kwargs")
            if len(args) < 1 or len(args) > 2:
                raise ValueError("Config.value expects positional arguments: key [, default]")
            key = _resolve_config_path_key(args[0], env, "Config.value")
            has_default = len(args) >= 2
            default_value = args[1] if has_default else None
            found, value = _expr_config_lookup(config, key)
            if not found:
                if not has_default:
                    raise KeyError(f"missing required config key: {key}")
                value = default_value
            return value
        if callee in {
            "Config.has_key",
            "Config.has_value",
            "Config.int",
            "Config.float",
            "Config.str",
            "Config.bool",
            "Config.list",
        }:
            if len(args) != 1:
                raise ValueError(f"{callee} expects one non-empty Path key")
            if "root" in kwargs:
                raise ValueError(f"{callee} does not support root")
            key = _resolve_config_path_key(args[0], env, callee)
            found, value = _expr_config_lookup(config, key)
            if callee == "Config.has_key":
                if "default" in kwargs:
                    raise ValueError("Config.has_key does not support default")
                return bool(found)
            if callee == "Config.has_value":
                if "default" in kwargs:
                    raise ValueError("Config.has_value does not support default")
                return bool(found) and value is not None
            if not found:
                if "default" not in kwargs:
                    raise KeyError(f"missing required config key: {key}")
                value = kwargs["default"]
            if callee == "Config.int":
                if isinstance(value, bool):
                    raise ValueError("Config.int expected int")
                if isinstance(value, int):
                    return int(value)
                if isinstance(value, str):
                    raw = value.strip()
                    if raw and (raw.isdigit() or (raw[0] in {"+", "-"} and raw[1:].isdigit())):
                        return int(raw)
                raise ValueError("Config.int expected int")
            if callee == "Config.float":
                if isinstance(value, bool):
                    raise ValueError("Config.float expected float")
                if isinstance(value, (int, float)):
                    return float(value)
                if isinstance(value, str) and value.strip():
                    return float(value.strip())
                raise ValueError("Config.float expected float")
            if callee == "Config.str":
                if not isinstance(value, str):
                    raise ValueError("Config.str expected string")
                return value
            if callee == "Config.bool":
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    raw = value.strip().lower()
                    if raw == "true":
                        return True
                    if raw == "false":
                        return False
                raise ValueError("Config.bool expected bool")
            if callee == "Config.list":
                if isinstance(value, list):
                    return value
                if isinstance(value, tuple):
                    return list(value)
                raise ValueError("Config.list expected list")
            raise ValueError(f"Unsupported config expression call: {callee}")
        if callee in {"Params.has_root", "Params.root"}:
            if len(args) != 1 or not isinstance(args[0], str):
                raise ValueError(f"{callee} expects one string root argument")
            root = args[0]
            if callee == "Params.has_root":
                return bool(_params_has_root(state_keys, root))
            default = kwargs.get("default", "")
            if not isinstance(default, str):
                raise ValueError("Params.root default must resolve to string")
            return root if _params_has_root(state_keys, root) else default
        raise ValueError(f"unsupported call expression: {callee!r}")
    raise ValueError(f"expression is not statically materializable: {type(expr).__name__}")


def _try_eval_expr(
    expr: AxonExpr,
    *,
    env: dict[str, object],
    config: dict[str, object],
    state_keys: set[str],
    resolve_names: bool = True,
) -> object:
    try:
        return _eval_expr(
            expr,
            env=env,
            config=config,
            state_keys=state_keys,
            resolve_names=resolve_names,
        )
    except Exception:
        return _NOT_EVALUABLE


def _materialize_kwarg_value(
    value: AxonKwargValue,
    *,
    env: dict[str, object],
    config: dict[str, object],
    state_keys: set[str],
    resolve_names: bool = True,
) -> AxonKwargValue:
    if isinstance(value, AxonExpr):
        return _materialize_expr(
            value,
            env=env,
            config=config,
            state_keys=state_keys,
            resolve_names=resolve_names,
        )
    return value


def _materialize_expr(
    expr: AxonExpr,
    *,
    env: dict[str, object],
    config: dict[str, object],
    state_keys: set[str],
    resolve_names: bool = True,
) -> AxonExpr:
    def _bool_literal(node: AxonExpr) -> bool | None:
        if isinstance(node, AxonExprBool):
            return bool(node.value)
        if isinstance(node, AxonExprParen):
            return _bool_literal(node.inner)
        return None

    evaluated = _try_eval_expr(
        expr,
        env=env,
        config=config,
        state_keys=state_keys,
        resolve_names=resolve_names,
    )
    if evaluated is not _NOT_EVALUABLE:
        return _expr_from_scalar(evaluated)

    if isinstance(expr, AxonExprParen):
        inner = _materialize_expr(
            expr.inner,
            env=env,
            config=config,
            state_keys=state_keys,
            resolve_names=resolve_names,
        )
        return AxonExprParen(inner=inner)
    if isinstance(expr, AxonExprIf):
        cond = _materialize_expr(
            expr.cond,
            env=env,
            config=config,
            state_keys=state_keys,
            resolve_names=resolve_names,
        )
        true_expr = _materialize_expr(
            expr.true_expr,
            env=env,
            config=config,
            state_keys=state_keys,
            resolve_names=resolve_names,
        )
        false_expr = _materialize_expr(
            expr.false_expr,
            env=env,
            config=config,
            state_keys=state_keys,
            resolve_names=resolve_names,
        )
        cond_val = _bool_literal(cond)
        if cond_val is True:
            return true_expr
        if cond_val is False:
            return false_expr
        return AxonExprIf(cond=cond, true_expr=true_expr, false_expr=false_expr)
    if isinstance(expr, AxonExprTernary):
        cond = _materialize_expr(
            expr.cond,
            env=env,
            config=config,
            state_keys=state_keys,
            resolve_names=resolve_names,
        )
        true_expr = _materialize_expr(
            expr.true_expr,
            env=env,
            config=config,
            state_keys=state_keys,
            resolve_names=resolve_names,
        )
        false_expr = _materialize_expr(
            expr.false_expr,
            env=env,
            config=config,
            state_keys=state_keys,
            resolve_names=resolve_names,
        )
        cond_val = _bool_literal(cond)
        if cond_val is True:
            return true_expr
        if cond_val is False:
            return false_expr
        return AxonExprTernary(cond=cond, true_expr=true_expr, false_expr=false_expr)
    if isinstance(expr, AxonExprBinary):
        return AxonExprBinary(
            op=expr.op,
            left=_materialize_expr(
                expr.left,
                env=env,
                config=config,
                state_keys=state_keys,
                resolve_names=resolve_names,
            ),
            right=_materialize_expr(
                expr.right,
                env=env,
                config=config,
                state_keys=state_keys,
                resolve_names=resolve_names,
            ),
        )
    if isinstance(expr, AxonExprList):
        return AxonExprList(
            items=tuple(
                _materialize_expr(
                    item, env=env, config=config, state_keys=state_keys, resolve_names=resolve_names
                )
                for item in expr.items
            )
        )
    if isinstance(expr, AxonExprTuple):
        return AxonExprTuple(
            items=tuple(
                _materialize_expr(
                    item, env=env, config=config, state_keys=state_keys, resolve_names=resolve_names
                )
                for item in expr.items
            )
        )
    if isinstance(expr, AxonExprCall):
        return AxonExprCall(
            callee=expr.callee,
            args=tuple(
                _materialize_expr(
                    arg, env=env, config=config, state_keys=state_keys, resolve_names=resolve_names
                )
                for arg in expr.args
            ),
            kwargs={
                key: _materialize_kwarg_value(
                    value,
                    env=env,
                    config=config,
                    state_keys=state_keys,
                    resolve_names=resolve_names,
                )
                for key, value in expr.kwargs.items()
            },
        )
    if isinstance(expr, AxonExprPipe):
        return AxonExprPipe(
            value=_materialize_expr(
                expr.value,
                env=env,
                config=config,
                state_keys=state_keys,
                resolve_names=resolve_names,
            ),
            stages=tuple(
                _materialize_expr(
                    stage,
                    env=env,
                    config=config,
                    state_keys=state_keys,
                    resolve_names=resolve_names,
                )
                for stage in expr.stages
            ),
        )
    if isinstance(expr, AxonExprBind):
        return AxonExprBind(
            value=_materialize_expr(
                expr.value,
                env=env,
                config=config,
                state_keys=state_keys,
                resolve_names=resolve_names,
            ),
            var=expr.var,
            body=_materialize_expr(
                expr.body,
                env=env,
                config=config,
                state_keys=state_keys,
                resolve_names=resolve_names,
            ),
        )
    if isinstance(expr, AxonExprLambda):
        return AxonExprLambda(
            var=expr.var,
            body=_materialize_expr(
                expr.body,
                env=env,
                config=config,
                state_keys=state_keys,
                resolve_names=resolve_names,
            ),
        )
    if isinstance(expr, AxonExprDo):
        return AxonExprDo(
            body=tuple(
                _materialize_statement(
                    stmt, env=env, config=config, state_keys=state_keys, resolve_names=resolve_names
                )
                for stmt in expr.body
            ),
            inline=expr.inline,
        )
    return expr


def _materialize_statement(
    stmt: AxonStatement,
    *,
    env: dict[str, object],
    config: dict[str, object],
    state_keys: set[str],
    resolve_names: bool = True,
) -> AxonStatement:
    if isinstance(stmt, AxonBind):
        return AxonBind(
            targets=stmt.targets,
            expr=_materialize_expr(
                stmt.expr,
                env=env,
                config=config,
                state_keys=state_keys,
                resolve_names=resolve_names,
            ),
        )
    if isinstance(stmt, AxonReturn):
        return AxonReturn(
            values=tuple(
                _materialize_expr(
                    value,
                    env=env,
                    config=config,
                    state_keys=state_keys,
                    resolve_names=resolve_names,
                )
                for value in stmt.values
            )
        )
    if isinstance(stmt, AxonYield):
        return AxonYield(
            values=tuple(
                _materialize_expr(
                    value,
                    env=env,
                    config=config,
                    state_keys=state_keys,
                    resolve_names=resolve_names,
                )
                for value in stmt.values
            )
        )
    if isinstance(stmt, AxonRepeat):
        return AxonRepeat(
            name=stmt.name,
            var=stmt.var,
            to_expr=_materialize_expr(
                stmt.to_expr,
                env=env,
                config=config,
                state_keys=state_keys,
                resolve_names=resolve_names,
            ),
            from_expr=_materialize_expr(
                stmt.from_expr,
                env=env,
                config=config,
                state_keys=state_keys,
                resolve_names=resolve_names,
            ),
            step_expr=_materialize_expr(
                stmt.step_expr,
                env=env,
                config=config,
                state_keys=state_keys,
                resolve_names=resolve_names,
            ),
            body=tuple(
                _materialize_statement(
                    item, env=env, config=config, state_keys=state_keys, resolve_names=resolve_names
                )
                for item in stmt.body
            ),
        )
    if isinstance(stmt, AxonScopeBind):
        prefix = stmt.prefix
        has_template = ("{" in prefix) or ("}" in prefix)
        if has_template:
            prefix_has_at = prefix.startswith("@")
            key = prefix[1:] if prefix_has_at else prefix
            resolved = _resolve_config_key_template(key, env, "scope")
            prefix = f"@{resolved}" if prefix_has_at else resolved
        return AxonScopeBind(
            targets=stmt.targets,
            prefix=prefix,
            kwargs={
                key: _materialize_kwarg_value(
                    value,
                    env=env,
                    config=config,
                    state_keys=state_keys,
                    resolve_names=resolve_names,
                )
                for key, value in stmt.kwargs.items()
            },
            body=tuple(
                _materialize_statement(
                    item, env=env, config=config, state_keys=state_keys, resolve_names=resolve_names
                )
                for item in stmt.body
            ),
        )
    raise ValueError(f"Unsupported statement type: {type(stmt).__name__}")


def _expr_uses_config_or_params(expr: AxonExpr) -> bool:
    if isinstance(expr, AxonExprCall):
        callee_base = expr.callee.split("@", 1)[0]
        if expr.callee.startswith("Config.") or expr.callee.startswith("Params."):
            return True
        if callee_base in {
            "int",
            "float",
            "bool",
            "list",
            "str",
            "value",
            "has",
            "has_key",
            "has_value",
            "param",
            "has_root",
            "param_scale",
        }:
            return True
        return any(_expr_uses_config_or_params(arg) for arg in expr.args) or any(
            _expr_uses_config_or_params(value)
            for value in expr.kwargs.values()
            if isinstance(value, AxonExpr)
        )
    if isinstance(expr, AxonExprPipe):
        return _expr_uses_config_or_params(expr.value) or any(
            _expr_uses_config_or_params(stage) for stage in expr.stages
        )
    if isinstance(expr, AxonExprBind):
        return _expr_uses_config_or_params(expr.value) or _expr_uses_config_or_params(expr.body)
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        return (
            _expr_uses_config_or_params(expr.cond)
            or _expr_uses_config_or_params(expr.true_expr)
            or _expr_uses_config_or_params(expr.false_expr)
        )
    if isinstance(expr, AxonExprBinary):
        return _expr_uses_config_or_params(expr.left) or _expr_uses_config_or_params(expr.right)
    if isinstance(expr, AxonExprParen):
        return _expr_uses_config_or_params(expr.inner)
    if isinstance(expr, AxonExprList):
        return any(_expr_uses_config_or_params(item) for item in expr.items)
    if isinstance(expr, AxonExprTuple):
        return any(_expr_uses_config_or_params(item) for item in expr.items)
    if isinstance(expr, AxonExprLambda):
        return _expr_uses_config_or_params(expr.body)
    if isinstance(expr, AxonExprDo):
        return any(_statement_uses_config_or_params(stmt) for stmt in expr.body)
    return False


def _statement_uses_config_or_params(stmt: AxonStatement) -> bool:
    if isinstance(stmt, AxonBind):
        return _expr_uses_config_or_params(stmt.expr)
    if isinstance(stmt, AxonReturn):
        return any(_expr_uses_config_or_params(value) for value in stmt.values)
    if isinstance(stmt, AxonRepeat):
        return (
            _expr_uses_config_or_params(stmt.to_expr)
            or _expr_uses_config_or_params(stmt.from_expr)
            or _expr_uses_config_or_params(stmt.step_expr)
            or any(_statement_uses_config_or_params(item) for item in stmt.body)
        )
    if isinstance(stmt, AxonScopeBind):
        return any(
            _expr_uses_config_or_params(value)
            for value in stmt.kwargs.values()
            if isinstance(value, AxonExpr)
        ) or any(_statement_uses_config_or_params(item) for item in stmt.body)
    return False


def _collect_expr_names(expr: AxonExpr, *, out: set[str]) -> None:
    if isinstance(expr, AxonExprName):
        out.add(expr.name)
        return
    if isinstance(expr, AxonExprPath):
        for part in expr.parts:
            start = 0
            while True:
                open_idx = part.find("{", start)
                if open_idx < 0:
                    break
                close_idx = part.find("}", open_idx + 1)
                if close_idx < 0:
                    break
                name = part[open_idx + 1 : close_idx].strip()
                if name:
                    out.add(name)
                start = close_idx + 1
        return
    if isinstance(expr, AxonExprCall):
        for arg in expr.args:
            _collect_expr_names(arg, out=out)
        for value in expr.kwargs.values():
            if isinstance(value, AxonExpr):
                _collect_expr_names(value, out=out)
        return
    if isinstance(expr, AxonExprPipe):
        _collect_expr_names(expr.value, out=out)
        for stage in expr.stages:
            _collect_expr_names(stage, out=out)
        return
    if isinstance(expr, AxonExprBind):
        _collect_expr_names(expr.value, out=out)
        _collect_expr_names(expr.body, out=out)
        return
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        _collect_expr_names(expr.cond, out=out)
        _collect_expr_names(expr.true_expr, out=out)
        _collect_expr_names(expr.false_expr, out=out)
        return
    if isinstance(expr, AxonExprBinary):
        _collect_expr_names(expr.left, out=out)
        _collect_expr_names(expr.right, out=out)
        return
    if isinstance(expr, AxonExprParen):
        _collect_expr_names(expr.inner, out=out)
        return
    if isinstance(expr, AxonExprList):
        for item in expr.items:
            _collect_expr_names(item, out=out)
        return
    if isinstance(expr, AxonExprTuple):
        for item in expr.items:
            _collect_expr_names(item, out=out)
        return
    if isinstance(expr, AxonExprLambda):
        _collect_expr_names(expr.body, out=out)
        return
    if isinstance(expr, AxonExprDo):
        for stmt in expr.body:
            _collect_statement_names(stmt, out=out)
        return


def _collect_statement_names(stmt: AxonStatement, *, out: set[str]) -> None:
    if isinstance(stmt, AxonBind):
        _collect_expr_names(stmt.expr, out=out)
        return
    if isinstance(stmt, AxonReturn):
        for value in stmt.values:
            _collect_expr_names(value, out=out)
        return
    if isinstance(stmt, AxonRepeat):
        _collect_expr_names(stmt.to_expr, out=out)
        _collect_expr_names(stmt.from_expr, out=out)
        _collect_expr_names(stmt.step_expr, out=out)
        for item in stmt.body:
            _collect_statement_names(item, out=out)
        return
    if isinstance(stmt, AxonScopeBind):
        prefix = stmt.prefix
        start = 0
        while True:
            open_idx = prefix.find("{", start)
            if open_idx < 0:
                break
            close_idx = prefix.find("}", open_idx + 1)
            if close_idx < 0:
                break
            name = prefix[open_idx + 1 : close_idx].strip()
            if name:
                out.add(name)
            start = close_idx + 1
        for kwarg_value in stmt.kwargs.values():
            if isinstance(kwarg_value, AxonExpr):
                _collect_expr_names(kwarg_value, out=out)
        for item in stmt.body:
            _collect_statement_names(item, out=out)
        return


def _prune_unused_constants(
    constants: dict[str, AxonExpr],
    *,
    module_bodies: dict[str, AxonExpr],
) -> dict[str, AxonExpr]:
    used: set[str] = set()
    for body in module_bodies.values():
        _collect_expr_names(body, out=used)

    changed = True
    while changed:
        changed = False
        for name in tuple(used):
            expr = constants.get(name)
            if expr is None:
                continue
            before = len(used)
            _collect_expr_names(expr, out=used)
            if len(used) != before:
                changed = True

    return {name: expr for name, expr in constants.items() if name in used}


def _float_text(value: float, lexeme: str | None = None) -> str:
    if lexeme:
        return lexeme
    text = repr(value)
    if text == "-0.0":
        return "0.0"
    return text


def _scalar_text(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return _float_text(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_scalar_text(item) for item in value) + "]"
    if isinstance(value, tuple):
        inner = ", ".join(_scalar_text(item) for item in value)
        if len(value) == 1:
            inner += ","
        return "(" + inner + ")"
    raise ValueError(f"Unsupported scalar text value: {value!r}")


def _kwarg_text(value: AxonKwargValue) -> str:
    if isinstance(value, AxonExpr):
        return _expr_text(value)
    return _scalar_text(value)


def _expr_text(expr: AxonExpr) -> str:
    if isinstance(expr, AxonExprName):
        return expr.name
    if isinstance(expr, AxonExprInt):
        return str(expr.value)
    if isinstance(expr, AxonExprFloat):
        return _float_text(expr.value, expr.lexeme)
    if isinstance(expr, AxonExprBool):
        return "true" if expr.value else "false"
    if isinstance(expr, AxonExprNull):
        return "null"
    if isinstance(expr, AxonExprString):
        return json.dumps(expr.value)
    if isinstance(expr, AxonExprPath):
        return expr.to_source()
    if isinstance(expr, AxonExprList):
        return "[" + ", ".join(_expr_text(item) for item in expr.items) + "]"
    if isinstance(expr, AxonExprTuple):
        inner = ", ".join(_expr_text(item) for item in expr.items)
        if len(expr.items) == 1:
            inner += ","
        return "(" + inner + ")"
    if isinstance(expr, AxonExprCall):
        parts = [_expr_text(arg) for arg in expr.args]
        parts.extend(f"{key}={_kwarg_text(value)}" for key, value in expr.kwargs.items())
        if not parts:
            return expr.callee
        return f"{expr.callee} {' '.join(parts)}"
    if isinstance(expr, AxonExprPipe):
        parts = [_expr_text(expr.value), *(_expr_text(stage) for stage in expr.stages)]
        return " |> ".join(parts)
    if isinstance(expr, AxonExprBind):
        return f"{_expr_text(expr.value)} >>= \\\\{expr.var} -> {_expr_text(expr.body)}"
    if isinstance(expr, AxonExprIf):
        return f"if {_expr_text(expr.cond)} then {_expr_text(expr.true_expr)} else {_expr_text(expr.false_expr)}"
    if isinstance(expr, AxonExprTernary):
        cond_text = _expr_text(expr.cond)
        true_text = _expr_text(expr.true_expr)
        false_text = _expr_text(expr.false_expr)
        if "\n" not in true_text and "\n" not in false_text:
            return f"{cond_text} ? {true_text} : {false_text}"

        def _paren(text: str) -> str:
            parts = text.splitlines() or [text]
            if len(parts) == 1:
                return f"({parts[0]})"
            indented = "\n".join("  " + line for line in parts[1:])
            return f"({parts[0]}\n{indented}\n)"

        return f"{cond_text} ? {_paren(true_text)} : {_paren(false_text)}"
    if isinstance(expr, AxonExprBinary):
        return f"{_expr_text(expr.left)} {expr.op} {_expr_text(expr.right)}"
    if isinstance(expr, AxonExprLambda):
        return f"\\\\{expr.var} -> {_expr_text(expr.body)}"
    if isinstance(expr, AxonExprParen):
        return f"({_expr_text(expr.inner)})"
    if isinstance(expr, AxonExprDo):
        if expr.inline:
            return "do " + "; ".join(_statement_inline_text(stmt) for stmt in expr.body)
        lines = ["do"]
        lines.extend(_render_statements(expr.body, indent="  "))
        return "\n".join(lines)
    raise ValueError(f"Unsupported expression type for rendering: {type(expr).__name__}")


def _statement_inline_text(stmt: AxonStatement) -> str:
    if isinstance(stmt, AxonBind):
        lhs = ", ".join(stmt.targets)
        return f"{lhs} <- {_expr_text(stmt.expr)}"
    if isinstance(stmt, AxonReturn):
        return "return " + ", ".join(_expr_text(value) for value in stmt.values)
    raise ValueError(f"Unsupported inline statement: {type(stmt).__name__}")


def _render_statements(statements: tuple[AxonStatement, ...], *, indent: str) -> list[str]:
    lines: list[str] = []
    for stmt in statements:
        if isinstance(stmt, AxonBind):
            lhs = ", ".join(stmt.targets)
            expr_text = _expr_text(stmt.expr)
            expr_lines = expr_text.splitlines() or [expr_text]
            lines.append(f"{indent}{lhs} <- {expr_lines[0]}")
            for tail in expr_lines[1:]:
                lines.append(f"{indent}{tail}")
            continue
        if isinstance(stmt, AxonReturn):
            lines.append(f"{indent}return {', '.join(_expr_text(value) for value in stmt.values)}")
            continue
        if isinstance(stmt, AxonYield):
            lines.append(f"{indent}yield {', '.join(_expr_text(value) for value in stmt.values)}")
            continue
        if isinstance(stmt, AxonRepeat):
            name = f"@{stmt.name}" if stmt.name else ""
            step = _expr_text(stmt.step_expr)
            step_suffix = "" if step == "1" else f" step={step}"
            lines.append(
                f"{indent}for{name} {stmt.var} <- [{_expr_text(stmt.from_expr)}..{_expr_text(stmt.to_expr)}){step_suffix} do"
            )
            lines.extend(_render_statements(stmt.body, indent=indent + "  "))
            continue
        if isinstance(stmt, AxonScopeBind):
            lhs = ", ".join(stmt.targets)
            kw_parts = [f"{key}={_kwarg_text(value)}" for key, value in stmt.kwargs.items()]
            kw_suffix = "" if not kw_parts else " " + " ".join(kw_parts)
            lines.append(f"{indent}{lhs} <- scope@{stmt.prefix}{kw_suffix} do")
            lines.extend(_render_statements(stmt.body, indent=indent + "  "))
            continue
        raise ValueError(f"Unsupported statement type for rendering: {type(stmt).__name__}")
    return lines


def _render_signature(signature: ParsedSignature) -> str:
    type_sig = signature.type_signature
    parts = []
    for path_param in type_sig.path_params:
        if path_param.name:
            parts.append(f"{path_param.name}@{render_type(path_param.type_expr)}")
        else:
            parts.append(f"@{render_type(path_param.type_expr)}")
    parts.extend(render_type(arg_type) for arg_type in type_sig.arg_types)
    parts.append(render_type(type_sig.return_type))
    return f"{signature.module_decl} :: " + " -> ".join(parts)


def _is_simple_default_expr(expr: AxonExpr) -> bool:
    return isinstance(
        expr,
        AxonExprInt
        | AxonExprFloat
        | AxonExprBool
        | AxonExprNull
        | AxonExprString
        | AxonExprPath
        | AxonExprName,
    )


def _render_def_param(param: ParsedDefParam) -> str:
    if not isinstance(param.default_expr, AxonExpr):
        return param.name
    default_text = _expr_text(param.default_expr)
    if _is_simple_default_expr(param.default_expr):
        return f"?{param.name}={default_text}"
    return f"?{param.name}=({default_text})"


def _render_definition(
    module_decl: str, args: tuple[ParsedDefParam, ...], rhs: AxonExpr
) -> list[str]:
    head = module_decl
    if args:
        head += " " + " ".join(_render_def_param(arg) for arg in args)
    if isinstance(rhs, AxonExprDo):
        lines = [f"{head} = do"]
        lines.extend(_render_statements(rhs.body, indent="  "))
        return lines
    return [f"{head} = {_expr_text(rhs)}"]


def _render_program(
    parsed: ParsedProgramSource,
    *,
    constants: dict[str, AxonExpr],
    module_bodies: dict[str, AxonExpr],
    checkpoints: list[str],
) -> str:
    lines: list[str] = []
    pragmas = dict(parsed.pragmas)
    pragmas["checkpoints"] = checkpoints if len(checkpoints) != 1 else checkpoints[0]
    for name, value in pragmas.items():
        lines.append("{-# " + name.upper() + " " + json.dumps(value) + " #-}")
    if lines:
        lines.append("")

    uses_cfg = any(_expr_uses_config_or_params(expr) for expr in constants.values()) or any(
        _expr_uses_config_or_params(expr) for expr in module_bodies.values()
    )
    for namespace in parsed.imports:
        if namespace in {"Config", "Params"} and not uses_cfg:
            continue
        lines.append(f"import {namespace}")
        members = parsed.imported_members.get(namespace, ())
        if members:
            lines.append(f"import {namespace} ({', '.join(members)})")
    if parsed.imports:
        lines.append("")

    for name, expr in constants.items():
        lines.append(f"{name} = {_expr_text(expr)}")
    if constants:
        lines.append("")

    for index, module in enumerate(parsed.modules):
        lines.append(_render_signature(module.signature))
        lines.extend(
            _render_definition(
                module.definition.module_decl,
                module.definition.args,
                module_bodies[module.definition.module_decl],
            )
        )
        if index != len(parsed.modules) - 1:
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _materialize_program(
    parsed: ParsedProgramSource,
    *,
    config: dict[str, object],
    state_keys: set[str],
) -> str:
    env: dict[str, object] = {}
    constants_out: dict[str, AxonExpr] = dict(parsed.constants)

    # Resolve constants to a fixed point so earlier constants can depend on later ones.
    for _ in range(max(1, len(constants_out) * 2)):
        changed = False
        for name, expr in parsed.constants.items():
            materialized = _materialize_expr(
                constants_out.get(name, expr),
                env=env,
                config=config,
                state_keys=state_keys,
                resolve_names=True,
            )
            if materialized != constants_out.get(name):
                constants_out[name] = materialized
                changed = True
            value = _try_eval_expr(materialized, env=env, config=config, state_keys=state_keys)
            if value is not _NOT_EVALUABLE and (name not in env or env[name] != value):
                env[name] = value
                changed = True
        if not changed:
            break

    module_bodies: dict[str, AxonExpr] = {}
    for module in parsed.modules:
        module_bodies[module.definition.module_decl] = _materialize_expr(
            module.definition.rhs,
            env=env,
            config=config,
            state_keys=state_keys,
            resolve_names=False,
        )

    constants_out = _prune_unused_constants(constants_out, module_bodies=module_bodies)

    return _render_program(
        parsed,
        constants=constants_out,
        module_bodies=module_bodies,
        checkpoints=[],
    )


def run_axon_materialize(
    *,
    axon_path: Path,
    checkpoints: list[str] | None = None,
    models_root: Path = Path("models"),
) -> list[Path]:
    resolved_axon = axon_path.resolve()
    resolved_models_root = models_root.resolve()
    if not resolved_axon.exists():
        raise FileNotFoundError(f"Axon file not found: {resolved_axon}")

    parsed = parse_program_source(resolved_axon.read_text(encoding="utf-8"))
    declared = _checkpoint_pragma(parsed)
    requested = list(checkpoints or declared)
    if not requested:
        raise ValueError(f"No CHECKPOINTS pragma entries found in {resolved_axon}")

    grouped: dict[str, list[str]] = {}
    for checkpoint in requested:
        model_dir = resolved_models_root / checkpoint
        config = _load_json(model_dir / "config.json")
        state_keys = _checkpoint_state_keys(model_dir)
        rendered = _materialize_program(parsed, config=config, state_keys=state_keys)
        existing_checkpoints = grouped.get(rendered)
        if existing_checkpoints is None:
            grouped[rendered] = [checkpoint]
            continue
        existing_checkpoints.append(checkpoint)

    written: list[Path] = []
    expected: set[Path] = set()
    stale_candidates: set[Path] = set()
    for body, body_checkpoints in grouped.items():
        out_name = f"{_group_output_name(body_checkpoints)}.axon"
        out_path = resolved_axon.parent / out_name
        text = body.replace(
            "CHECKPOINTS []",
            f"CHECKPOINTS {json.dumps(body_checkpoints if len(body_checkpoints) != 1 else body_checkpoints[0])}",
        )
        out_path.write_text(text, encoding="utf-8")
        # Validate the generated source immediately.
        parse_program_source(text)
        expected.add(out_path.resolve())
        written.append(out_path)
        for checkpoint in body_checkpoints:
            stale_candidates.add(
                (resolved_axon.parent / f"{checkpoint.split('/')[-1]}.axon").resolve()
            )
            stale_candidates.add(
                (resolved_axon.parent / f"{_normalize_checkpoint_name(checkpoint)}.axon").resolve()
            )

    for stale_path in stale_candidates:
        if stale_path not in expected and stale_path.exists():
            stale_path.unlink()

    return written
