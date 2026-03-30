from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

_MODEL_TYPES_KEY = "types"
_BLOCK_IO_TYPES_KEY = "block_io"


def normalize_type_expr(type_expr: str, *, optional: bool = False) -> str:
    stripped = type_expr.strip()
    if not stripped:
        return "?Any" if optional else "Any"
    if optional and not stripped.startswith("?"):
        return f"?{stripped}"
    return stripped


def split_top_level_types(type_expr: str) -> list[str]:
    expr = type_expr.strip()
    if not expr:
        return []
    if expr.startswith("(") and expr.endswith(")"):
        expr = expr[1:-1].strip()
    if not expr:
        return []
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in expr:
        if ch == "," and depth == 0:
            token = "".join(current).strip()
            if token:
                parts.append(token)
            current = []
            continue
        if ch in "([<":
            depth += 1
        elif ch in ")]>":
            depth = max(0, depth - 1)
        current.append(ch)
    token = "".join(current).strip()
    if token:
        parts.append(token)
    return parts


def split_bracket_dims(type_expr: str) -> list[str] | None:
    expr = type_expr.strip()
    optional = expr.startswith("?")
    core = expr[1:] if optional else expr
    left = core.find("[")
    right = core.rfind("]")
    if left <= 0 or right <= left:
        return None
    inner = core[left + 1 : right].strip()
    if not inner:
        return []
    return [part.strip() for part in split_top_level_types(inner)]


def replace_last_dim(type_expr: str, new_dim: str) -> str | None:
    expr = type_expr.strip()
    optional = expr.startswith("?")
    core = expr[1:] if optional else expr
    left = core.find("[")
    right = core.rfind("]")
    if left <= 0 or right <= left:
        return None
    base = core[:left]
    dims = split_bracket_dims(type_expr)
    if dims is None:
        return None
    if not dims:
        dims = [new_dim]
    else:
        dims[-1] = new_dim
    rebuilt = f"{base}[{','.join(dims)}]"
    return f"?{rebuilt}" if optional else rebuilt


def append_dim(type_expr: str, *, out_base: str, new_dim: str) -> str | None:
    dims = split_bracket_dims(type_expr)
    if dims is None:
        return None
    return f"{out_base}[{','.join([*dims, new_dim])}]"


def literal_type(value: Any) -> str | None:
    if value is None:
        return "Null"
    if isinstance(value, bool):
        return "Bool"
    if isinstance(value, int):
        return "I"
    if isinstance(value, float):
        return "F"
    if isinstance(value, str):
        return "Str"
    return None


def infer_input_slot_types(
    *,
    input_slots: list[tuple[str, set[str]]],
    var_types: Mapping[str, str],
) -> dict[str, str]:
    inferred: dict[str, str] = {}
    for slot_name, refs in input_slots:
        ref_types: list[str] = []
        for ref in sorted(refs):
            value = var_types.get(ref)
            if isinstance(value, str):
                ref_types.append(value)
        if len(ref_types) == 1:
            inferred[slot_name] = ref_types[0]
        elif len(ref_types) > 1 and len(set(ref_types)) == 1:
            inferred[slot_name] = ref_types[0]
    return inferred


def infer_output_types_for_node(
    *,
    op_name: str,
    node_spec: Mapping[str, Any],
    input_slots: list[tuple[str, set[str]]],
    output_vars: list[str],
    var_types: Mapping[str, str],
) -> dict[str, str]:
    inferred: dict[str, str] = {}
    in_types_by_slot = infer_input_slot_types(input_slots=input_slots, var_types=var_types)
    first_type: str | None = None
    for _slot_name, refs in input_slots:
        slot_types: list[str] = []
        for ref in sorted(refs):
            value = var_types.get(ref)
            if isinstance(value, str):
                slot_types.append(value)
        if not slot_types:
            continue
        if len(set(slot_types)) == 1:
            first_type = slot_types[0]
            break

    same_as_first = {
        "_ir_alias",
        "activation",
        "add",
        "mul",
        "clamp",
        "layernorm",
        "rmsnorm",
        "softmax",
        "zeros_like",
        "merge_heads",
        "repeat",
        "reshape_heads",
    }
    if op_name in same_as_first and isinstance(first_type, str):
        for out_name in output_vars:
            inferred[out_name] = first_type

    if op_name == "linear":
        x_type = in_types_by_slot.get("x") or first_type
        dim_val = node_spec.get("dim")
        if isinstance(x_type, str) and (isinstance(dim_val, str) or isinstance(dim_val, int)):
            replaced = replace_last_dim(x_type, str(dim_val))
            if isinstance(replaced, str):
                for out_name in output_vars:
                    inferred[out_name] = replaced

    if op_name == "embedding":
        x_type = in_types_by_slot.get("x") or first_type
        dim_val = node_spec.get("dim")
        if isinstance(x_type, str) and (isinstance(dim_val, str) or isinstance(dim_val, int)):
            out_type = append_dim(x_type, out_base="Tensor", new_dim=str(dim_val))
            if isinstance(out_type, str):
                for out_name in output_vars:
                    inferred[out_name] = out_type

    if op_name == "topk":
        x_type = in_types_by_slot.get("x") or first_type
        k_val = node_spec.get("k")
        if isinstance(k_val, int) or isinstance(k_val, str):
            if isinstance(x_type, str):
                score_type = replace_last_dim(x_type, str(k_val))
                if isinstance(score_type, str) and output_vars:
                    inferred[output_vars[0]] = score_type
                idx_dims = split_bracket_dims(score_type) if isinstance(score_type, str) else None
                if isinstance(idx_dims, list) and len(output_vars) >= 2:
                    inferred[output_vars[1]] = f"IdxTensor[{','.join(idx_dims)}]"

    if op_name == "_ir_expr":
        literal = literal_type(node_spec.get("value"))
        if isinstance(literal, str):
            for out_name in output_vars:
                inferred[out_name] = literal

    if op_name == "list_init":
        for out_name in output_vars:
            inferred[out_name] = "List[Any]"

    if op_name == "list_append":
        xs_type = in_types_by_slot.get("xs")
        if isinstance(xs_type, str):
            for out_name in output_vars:
                inferred[out_name] = xs_type

    return inferred


def module_io_types(
    module: Any,
    *,
    input_names: list[str],
    output_names: list[str],
) -> dict[str, dict[str, str]]:
    input_types: dict[str, str] = {}
    for param in getattr(module, "params", ()):
        name = getattr(param, "name", None)
        if not isinstance(name, str) or name not in input_names:
            continue
        type_expr = getattr(param, "type_expr", None)
        optional = bool(getattr(param, "optional", False))
        if isinstance(type_expr, str) and type_expr:
            input_types[name] = normalize_type_expr(type_expr, optional=optional)
        else:
            input_types[name] = "?Any" if optional else "Any"

    ret_expr = getattr(module, "return_type_expr", None)
    out_type_tokens = split_top_level_types(ret_expr) if isinstance(ret_expr, str) else []
    output_types: dict[str, str] = {}
    if len(output_names) == 1:
        if out_type_tokens:
            output_types[output_names[0]] = normalize_type_expr(out_type_tokens[0])
    elif len(out_type_tokens) == len(output_names):
        for name, type_expr in zip(output_names, out_type_tokens, strict=False):
            output_types[name] = normalize_type_expr(type_expr)

    return {"inputs": input_types, "outputs": output_types}


def infer_block_io_types_from_modules(
    *,
    spec: Mapping[str, Any],
    modules: Sequence[Any],
    selected_main: str,
) -> dict[str, dict[str, dict[str, str]]]:
    selected_main_module = next(
        (module for module in modules if module.name == selected_main), None
    )
    if selected_main_module is None:
        return {}

    model_spec = spec.get("model", {})
    if not isinstance(model_spec, Mapping):
        return {}

    block_io_types: dict[str, dict[str, dict[str, str]]] = {}
    model_inputs = model_spec.get("inputs", {})
    model_outputs = model_spec.get("outputs", {})
    main_input_names = (
        [str(name) for name in model_inputs.keys()] if isinstance(model_inputs, Mapping) else []
    )
    main_output_names = (
        [str(name) for name in model_outputs.keys()] if isinstance(model_outputs, Mapping) else []
    )
    block_io_types["main"] = module_io_types(
        selected_main_module,
        input_names=[name for name in main_input_names if isinstance(name, str)],
        output_names=[name for name in main_output_names if isinstance(name, str)],
    )

    modules_by_name = {module.name: module for module in modules}
    blocks_spec = model_spec.get("blocks", {})
    if not isinstance(blocks_spec, Mapping):
        return block_io_types

    for block_name, block_spec in blocks_spec.items():
        if not isinstance(block_name, str) or not isinstance(block_spec, Mapping):
            continue
        module = modules_by_name.get(block_name)
        if module is None:
            continue
        block_inputs = block_spec.get("inputs", {})
        block_outputs = block_spec.get("outputs", {})
        input_names = (
            [str(name) for name in block_inputs.keys()] if isinstance(block_inputs, Mapping) else []
        )
        output_names = (
            [str(name) for name in block_outputs.keys()]
            if isinstance(block_outputs, Mapping)
            else []
        )
        block_io_types[block_name] = module_io_types(
            module,
            input_names=[name for name in input_names if isinstance(name, str)],
            output_names=[name for name in output_names if isinstance(name, str)],
        )

    return block_io_types


def annotate_spec_with_block_io_types(
    spec: dict[str, Any],
    *,
    block_io_types: Mapping[str, Mapping[str, Mapping[str, str]]],
) -> None:
    model = spec.get("model")
    if not isinstance(model, dict):
        return
    clean: dict[str, dict[str, dict[str, str]]] = {}
    for block_name, io_spec in block_io_types.items():
        if not isinstance(block_name, str) or not isinstance(io_spec, Mapping):
            continue
        inputs = io_spec.get("inputs", {})
        outputs = io_spec.get("outputs", {})
        clean_inputs = (
            {
                name: type_expr
                for name, type_expr in inputs.items()
                if isinstance(name, str) and isinstance(type_expr, str) and type_expr
            }
            if isinstance(inputs, Mapping)
            else {}
        )
        clean_outputs = (
            {
                name: type_expr
                for name, type_expr in outputs.items()
                if isinstance(name, str) and isinstance(type_expr, str) and type_expr
            }
            if isinstance(outputs, Mapping)
            else {}
        )
        clean[block_name] = {"inputs": clean_inputs, "outputs": clean_outputs}

    types = model.get(_MODEL_TYPES_KEY)
    types_dict: dict[str, Any]
    if isinstance(types, dict):
        types_dict = types
    else:
        types_dict = {}
        model[_MODEL_TYPES_KEY] = types_dict
    types_dict[_BLOCK_IO_TYPES_KEY] = clean


def extract_block_io_types_from_spec(
    spec: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, str]]]:
    model = spec.get("model")
    if not isinstance(model, Mapping):
        return {}
    types = model.get(_MODEL_TYPES_KEY)
    if not isinstance(types, Mapping):
        return {}
    raw_block_io = types.get(_BLOCK_IO_TYPES_KEY)
    if not isinstance(raw_block_io, Mapping):
        return {}
    out: dict[str, dict[str, dict[str, str]]] = {}
    for block_name, io_spec in raw_block_io.items():
        if not isinstance(block_name, str) or not isinstance(io_spec, Mapping):
            continue
        inputs = io_spec.get("inputs", {})
        outputs = io_spec.get("outputs", {})
        clean_inputs = (
            {
                name: type_expr
                for name, type_expr in inputs.items()
                if isinstance(name, str) and isinstance(type_expr, str) and type_expr
            }
            if isinstance(inputs, Mapping)
            else {}
        )
        clean_outputs = (
            {
                name: type_expr
                for name, type_expr in outputs.items()
                if isinstance(name, str) and isinstance(type_expr, str) and type_expr
            }
            if isinstance(outputs, Mapping)
            else {}
        )
        out[block_name] = {"inputs": clean_inputs, "outputs": clean_outputs}
    return out


__all__ = [
    "annotate_spec_with_block_io_types",
    "append_dim",
    "extract_block_io_types_from_spec",
    "infer_block_io_types_from_modules",
    "infer_input_slot_types",
    "infer_output_types_for_node",
    "literal_type",
    "module_io_types",
    "normalize_type_expr",
    "replace_last_dim",
    "split_bracket_dims",
    "split_top_level_types",
]
