from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PipelineStage:
    index: int
    device: str
    layer_start: int
    layer_stop: int

    @property
    def layer_count(self) -> int:
        return self.layer_stop - self.layer_start


@dataclass(frozen=True)
class PipelinePlan:
    devices: tuple[str, ...]
    layers_var: str
    layers_scope: str
    total_layers: int
    stages: tuple[PipelineStage, ...]


def available_pipeline_devices(requested_device: str | torch.device) -> tuple[str, ...]:
    device = torch.device(requested_device)
    if device.type != "cuda":
        raise ValueError("pipeline backend currently requires CUDA")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if device.index is not None:
        return (f"cuda:{device.index}",)
    count = torch.cuda.device_count()
    if count <= 0:
        raise ValueError("No CUDA devices available")
    return tuple(f"cuda:{idx}" for idx in range(count))


def partition_layer_ranges(total_layers: int, num_stages: int) -> tuple[tuple[int, int], ...]:
    if total_layers <= 0:
        raise ValueError("total_layers must be > 0")
    if num_stages <= 0:
        raise ValueError("num_stages must be > 0")
    active = min(total_layers, num_stages)
    base = total_layers // active
    remainder = total_layers % active
    out: list[tuple[int, int]] = []
    cursor = 0
    for idx in range(active):
        width = base + (1 if idx < remainder else 0)
        start = cursor
        stop = start + width
        out.append((start, stop))
        cursor = stop
    return tuple(out)


def _iter_node_specs(obj: Any) -> Any:
    if isinstance(obj, dict):
        if "_op" in obj:
            yield obj
        for value in obj.values():
            yield from _iter_node_specs(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _iter_node_specs(value)


def _find_bound_int(
    model_graph: list[Any],
    name: str,
    *,
    model_symbols: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
) -> int | None:
    def _coerce_int(value: Any) -> int | None:
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return None

    def _lookup_path(mapping: dict[str, Any] | None, path: str) -> Any | None:
        if not isinstance(mapping, dict):
            return None
        if path == "":
            return mapping
        current: Any = mapping
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        return current

    def _bound_node(bind_name: str, *, nodes: list[Any] | None = None) -> dict[str, Any] | None:
        search_nodes: Any = model_graph if nodes is None else nodes
        for node_spec in _iter_node_specs(search_nodes):
            if node_spec.get("_bind") == bind_name:
                return node_spec
        return None

    def _bound_node_block(bind_name: str, *, nodes: Any) -> dict[str, Any] | None:
        for node_spec in _iter_node_specs(nodes):
            if node_spec.get("_bind") == bind_name:
                return node_spec
        for node_spec in _iter_node_specs(model_graph):
            if node_spec.get("_bind") == bind_name:
                return node_spec
        return None

    def _resolve_block_bound_string(block: Any, bind_name: str) -> str | None:
        if not isinstance(block, list):
            return None
        for node_spec in _iter_node_specs(block):
            if node_spec.get("_bind") != bind_name:
                continue
            if node_spec.get("_op") != "_ir_expr":
                continue
            value = node_spec.get("value")
            if isinstance(value, dict) and value.get("_expr") == "string":
                out = value.get("value")
                if isinstance(out, str):
                    return out
        return None

    def _resolve_config_root(root: Any) -> str | None:
        if isinstance(root, str) and (root == "" or _lookup_path(model_config, root) is not None):
            return root
        if not isinstance(root, str):
            return None
        root_node = _bound_node(root)
        if not isinstance(root_node, dict):
            return None
        root_op = root_node.get("_op")
        if root_op == "_ir_expr":
            value = root_node.get("value")
            if isinstance(value, dict) and value.get("_expr") == "string":
                resolved = value.get("value")
                if isinstance(resolved, str):
                    return resolved
            return None
        if root_op != "select":
            return None
        cond_name = root_node.get("cond")
        if not isinstance(cond_name, str):
            return None
        cond_node = _bound_node(cond_name)
        if not isinstance(cond_node, dict) or cond_node.get("_op") != "config_has":
            return None
        cond_key = cond_node.get("_args")
        if not isinstance(cond_key, str):
            return None
        cond_root = _resolve_config_root(cond_node.get("root"))
        cond_base = _lookup_path(model_config, cond_root or "")
        cond_value = isinstance(cond_base, dict) and cond_key in cond_base
        then_bind = root_node.get("_then_bind")
        else_bind = root_node.get("_else_bind")
        if not isinstance(then_bind, str) or not isinstance(else_bind, str):
            return None
        if cond_value:
            return _resolve_block_bound_string(root_node.get("_then"), then_bind)
        return _resolve_block_bound_string(root_node.get("_else"), else_bind)

    def _lookup_config_int(
        model_config: dict[str, Any] | None,
        key: str,
        *,
        root: Any = None,
    ) -> int | None:
        base = _lookup_path(model_config, _resolve_config_root(root) or "")
        if not isinstance(base, dict):
            return None
        direct = _coerce_int(base.get(key))
        if direct is not None:
            return direct
        dotted = _lookup_path(base, key)
        return _coerce_int(dotted)

    def _scalar_const(
        value: Any,
        *,
        local_nodes: list[Any] | None,
        cache: dict[str, Any],
        visiting: set[str],
    ) -> Any | None:
        if isinstance(value, (int, float, bool)):
            return value
        if isinstance(value, str):
            resolved = _resolve_bind_scalar(
                value,
                local_nodes=local_nodes,
                cache=cache,
                visiting=visiting,
            )
            if resolved is not None:
                return resolved
            return value
        if isinstance(value, dict):
            expr = value.get("_expr")
            if expr == "name":
                ident = value.get("id")
                if isinstance(ident, str):
                    return _resolve_bind_scalar(
                        ident,
                        local_nodes=local_nodes,
                        cache=cache,
                        visiting=visiting,
                    )
            if expr in {"int", "float", "bool", "string"}:
                return value.get("value")
        return None

    def _resolve_bind_scalar(
        bind_name: str,
        *,
        local_nodes: list[Any] | None,
        cache: dict[str, Any],
        visiting: set[str],
    ) -> Any | None:
        if bind_name in cache:
            return cache[bind_name]
        if bind_name in visiting:
            return None

        if isinstance(model_symbols, dict) and bind_name in model_symbols:
            cache[bind_name] = model_symbols[bind_name]
            return cache[bind_name]

        node = (
            _bound_node(bind_name, nodes=local_nodes)
            if local_nodes is not None
            else _bound_node(bind_name)
        )
        if node is None:
            return None

        visiting.add(bind_name)
        try:
            op = node.get("_op")
            if op == "_ir_expr":
                value = node.get("value")
                if isinstance(value, dict):
                    expr = value.get("_expr")
                    if expr in {"int", "float", "bool", "string"}:
                        cache[bind_name] = value.get("value")
                        return cache[bind_name]
                return None

            if op == "config_int":
                key = node.get("_args")
                if isinstance(key, str):
                    cfg_value = _lookup_config_int(model_config, key, root=node.get("root"))
                    if cfg_value is not None:
                        cache[bind_name] = cfg_value
                        return cache[bind_name]
                default = node.get("default")
                if isinstance(default, int):
                    cache[bind_name] = default
                    return cache[bind_name]
                return None

            if op == "config_has":
                key = node.get("_args")
                if not isinstance(key, str):
                    return None
                root = _resolve_config_root(node.get("root"))
                base = _lookup_path(model_config, root or "")
                value = isinstance(base, dict) and key in base
                cache[bind_name] = value
                return value

            if op == "select":
                cond_name = node.get("cond")
                if not isinstance(cond_name, str):
                    return None
                cond_value = _resolve_bind_scalar(
                    cond_name,
                    local_nodes=local_nodes,
                    cache=cache,
                    visiting=visiting,
                )
                branch_nodes = node.get("_then") if bool(cond_value) else node.get("_else")
                branch_bind = node.get("_then_bind") if bool(cond_value) else node.get("_else_bind")
                if isinstance(branch_bind, str):
                    branch_node = _bound_node_block(branch_bind, nodes=branch_nodes)
                    if branch_node is not None:
                        branch_value = _resolve_bind_scalar(
                            branch_bind,
                            local_nodes=(
                                branch_nodes if isinstance(branch_nodes, list) else local_nodes
                            ),
                            cache=cache,
                            visiting=visiting,
                        )
                        cache[bind_name] = branch_value
                        return branch_value
                return None

            arg_raw = node.get("_args")
            args: list[Any]
            if isinstance(arg_raw, list):
                args = [
                    _scalar_const(arg, local_nodes=local_nodes, cache=cache, visiting=visiting)
                    for arg in arg_raw
                ]
            else:
                args = [
                    _scalar_const(arg_raw, local_nodes=local_nodes, cache=cache, visiting=visiting)
                ]
            if any(arg is None for arg in args):
                return None
            a0 = args[0]
            if op == "add" and len(args) == 2:
                cache[bind_name] = args[0] + args[1]
                return cache[bind_name]
            if op == "sub" and len(args) == 2:
                cache[bind_name] = args[0] - args[1]
                return cache[bind_name]
            if op == "mul" and len(args) == 2:
                cache[bind_name] = args[0] * args[1]
                return cache[bind_name]
            if op == "div" and len(args) == 2:
                cache[bind_name] = args[0] / args[1]
                return cache[bind_name]
            if op == "floor" and len(args) == 1 and isinstance(a0, (int, float)):
                cache[bind_name] = int(a0) if isinstance(a0, int) else int(a0 // 1)
                return cache[bind_name]
            if op == "sqrt" and len(args) == 1 and isinstance(a0, (int, float)) and a0 >= 0:
                cache[bind_name] = a0**0.5
                return cache[bind_name]
            return None
        finally:
            visiting.discard(bind_name)

    for node_spec in _iter_node_specs(model_graph):
        if node_spec.get("_bind") != name:
            continue
        if node_spec.get("_op") != "config_int":
            continue
        key = node_spec.get("_args")
        if isinstance(key, str):
            cfg_value = _lookup_config_int(model_config, key, root=node_spec.get("root"))
            if cfg_value is not None:
                return cfg_value
    if isinstance(model_symbols, dict):
        value = model_symbols.get(name)
        if isinstance(value, int):
            return value
    for node_spec in _iter_node_specs(model_graph):
        if node_spec.get("_bind") != name:
            continue
        if node_spec.get("_op") != "config_int":
            continue
        default = node_spec.get("default")
        if isinstance(default, int):
            return default
    resolved_value = _resolve_bind_scalar(name, local_nodes=None, cache={}, visiting=set())
    return _coerce_int(resolved_value)


def _resolve_int_expr(
    value: Any,
    model_graph: list[Any],
    *,
    model_symbols: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, dict) and value.get("_expr") == "name":
        symbol = value.get("id")
        if isinstance(symbol, str):
            return _find_bound_int(
                model_graph,
                symbol,
                model_symbols=model_symbols,
                model_config=model_config,
            )
    return None


def _iter_top_level_named_nodes(model_graph: list[Any]) -> Any:
    for index, item in enumerate(model_graph):
        if not isinstance(item, dict) or len(item) != 1:
            continue
        node_name, node_spec = next(iter(item.items()))
        if not isinstance(node_name, str) or not isinstance(node_spec, dict):
            continue
        yield index, node_name, node_spec


def _top_level_layer_loops(
    model_graph: list[Any],
    *,
    model_symbols: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
) -> list[tuple[int, str, dict[str, Any], str, str, int, int]]:
    def _is_layerish_scope(scope_name: str) -> bool:
        token = scope_name.rsplit(".", 1)[-1]
        return token in {"layers", "layer", "block", "h"}

    loops: list[tuple[int, str, dict[str, Any], str, str, int, int]] = []
    for index, node_name, node_spec in _iter_top_level_named_nodes(model_graph):
        if node_spec.get("_op") != "for":
            continue
        scope_name = node_spec.get("_scope")
        var_name = node_spec.get("_var")
        if not isinstance(scope_name, str) or not _is_layerish_scope(scope_name):
            continue
        if not isinstance(var_name, str) or not var_name:
            continue
        to_value = _resolve_int_expr(
            node_spec.get("_to"),
            model_graph,
            model_symbols=model_symbols,
            model_config=model_config,
        )
        if to_value is None:
            raise ValueError("pipeline backend requires layer loop _to to resolve to an int")
        from_value_raw = node_spec.get("_from", 0)
        from_value = _resolve_int_expr(
            from_value_raw,
            model_graph,
            model_symbols=model_symbols,
            model_config=model_config,
        )
        if from_value is None:
            raise ValueError("pipeline backend requires layer loop _from to resolve to an int")
        loops.append((index, node_name, node_spec, var_name, scope_name, from_value, to_value))
    return loops


def _find_primary_layer_loop(
    model_graph: list[Any],
    *,
    model_symbols: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
) -> tuple[str, str, int]:
    loops = _top_level_layer_loops(
        model_graph,
        model_symbols=model_symbols,
        model_config=model_config,
    )
    if not loops:
        raise ValueError("pipeline backend could not find a top-level for@*.layers loop")
    first_var = loops[0][3]
    total_layers = max(to_value for *_head, to_value in loops)
    return first_var, "layers", total_layers


def _find_primary_layer_loop_top_level(
    model_graph: list[Any],
    *,
    model_symbols: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
) -> tuple[int, str, dict[str, Any], int]:
    loops = _top_level_layer_loops(
        model_graph,
        model_symbols=model_symbols,
        model_config=model_config,
    )
    if not loops:
        raise ValueError(
            "pipeline backend requires a top-level for@*.layers loop for stage partitioning"
        )
    index, node_name, node_spec, _var, _scope, _from, _to = loops[0]
    total_layers = max(to_value for *_head, to_value in loops)
    return index, node_name, node_spec, total_layers


def _copy_item(item: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(item)


def _filter_prefix_graph(prefix_graph: list[Any], *, drop_binds: set[str]) -> list[Any]:
    filtered: list[Any] = []
    for item in prefix_graph:
        if not isinstance(item, dict) or len(item) != 1:
            filtered.append(copy.deepcopy(item))
            continue
        _node_name, node_spec = next(iter(item.items()))
        if not isinstance(node_spec, dict):
            filtered.append(copy.deepcopy(item))
            continue
        bind = node_spec.get("_bind")
        binds = tuple(bind) if isinstance(bind, list) else (bind,)
        if any(isinstance(name, str) and name in drop_binds for name in binds):
            continue
        filtered.append(copy.deepcopy(item))
    return filtered


def build_pipeline_stage_spec(spec: dict[str, Any], stage: PipelineStage) -> dict[str, Any]:
    model = spec.get("model")
    if not isinstance(model, dict):
        raise ValueError("spec.model must be a mapping")
    graph = model.get("graph")
    if not isinstance(graph, list):
        raise ValueError("spec.model.graph must be a list")
    model_symbols = model.get("symbols")
    if model_symbols is not None and not isinstance(model_symbols, dict):
        raise ValueError("spec.model.symbols must be a mapping when present")
    model_config = model.get("config")
    if model_config is not None and not isinstance(model_config, dict):
        raise ValueError("spec.model.config must be a mapping when present")
    layer_loops = _top_level_layer_loops(
        graph,
        model_symbols=model_symbols,
        model_config=model_config,
    )
    if not layer_loops:
        raise ValueError(
            "pipeline backend requires a top-level for@*.layers loop for stage partitioning"
        )
    first_var = layer_loops[0][3]
    first_scope = layer_loops[0][4]
    for _idx, _name, _spec, var_name, scope_name, _from, _to in layer_loops[1:]:
        if var_name != first_var or scope_name != first_scope:
            raise ValueError(
                "pipeline backend requires all top-level for@*.layers loops "
                "to share the same loop variable and scope"
            )
    total_layers = max(to_value for *_head, to_value in layer_loops)
    if (
        stage.layer_start < 0
        or stage.layer_stop > total_layers
        or stage.layer_start >= stage.layer_stop
    ):
        raise ValueError("pipeline stage range is out of bounds")

    out_spec = copy.deepcopy(spec)
    out_model = out_spec["model"]
    first_loop_index = min(index for index, *_rest in layer_loops)
    last_loop_index = max(index for index, *_rest in layer_loops)
    prefix_graph = graph[:first_loop_index]
    suffix_graph = graph[last_loop_index + 1 :]
    original_outputs = out_model.get("outputs", {})
    if not isinstance(original_outputs, dict):
        raise ValueError("spec.model.outputs must be a mapping")

    is_first = stage.index == 0
    is_last = stage.layer_stop == total_layers

    if is_first:
        stage_prefix = [_copy_item(item) for item in prefix_graph]
        stage_inputs = copy.deepcopy(out_model.get("inputs", {}))
    else:
        stage_prefix = _filter_prefix_graph(prefix_graph, drop_binds={"x"})
        stage_inputs = copy.deepcopy(out_model.get("inputs", {}))
        stage_inputs["x"] = {"optional": False}

    stage_layer_region: list[Any] = []
    layer_loop_index_set = {index for index, *_rest in layer_loops}
    for index in range(first_loop_index, last_loop_index + 1):
        item = graph[index]
        if index not in layer_loop_index_set:
            stage_layer_region.append(_copy_item(item))
            continue
        copied = _copy_item(item)
        node_spec = next(iter(copied.values()))
        orig_spec = graph[index]
        orig_node_spec = next(iter(orig_spec.values()))
        orig_from = _resolve_int_expr(
            orig_node_spec.get("_from", 0),
            graph,
            model_symbols=model_symbols,
            model_config=model_config,
        )
        orig_to = _resolve_int_expr(
            orig_node_spec.get("_to"),
            graph,
            model_symbols=model_symbols,
            model_config=model_config,
        )
        if orig_from is None or orig_to is None:
            raise ValueError("pipeline backend requires layer loop bounds to resolve to ints")
        new_from = max(orig_from, stage.layer_start)
        new_to = min(orig_to, stage.layer_stop)
        if new_from >= new_to:
            continue
        node_spec["_from"] = int(new_from)
        node_spec["_to"] = int(new_to)
        stage_layer_region.append(copied)

    if not stage_layer_region:
        raise ValueError(
            f"pipeline stage [{stage.layer_start},{stage.layer_stop}) has no executable layer loops"
        )

    if is_last:
        stage_suffix = [_copy_item(item) for item in suffix_graph]
        stage_outputs = copy.deepcopy(original_outputs)
    else:
        stage_suffix = []
        stage_outputs = {"x": "x"}
        if "new_kv" in original_outputs:
            stage_outputs["new_kv"] = "new_kv"

    out_model["inputs"] = stage_inputs
    out_model["graph"] = [*stage_prefix, *stage_layer_region, *stage_suffix]
    out_model["outputs"] = stage_outputs
    meta = out_model.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        out_model["meta"] = meta
    meta["pipeline_stage"] = {
        "index": stage.index,
        "device": stage.device,
        "layer_start": stage.layer_start,
        "layer_stop": stage.layer_stop,
        "total_layers": total_layers,
    }
    return out_spec


def build_pipeline_stage_specs(
    spec: dict[str, Any],
    *,
    requested_device: str | torch.device = "cuda",
) -> tuple[PipelinePlan, tuple[dict[str, Any], ...]]:
    plan = build_pipeline_plan(spec, requested_device=requested_device)
    return plan, tuple(build_pipeline_stage_spec(spec, stage) for stage in plan.stages)


def build_pipeline_plan(
    spec: dict[str, Any],
    *,
    requested_device: str | torch.device = "cuda",
) -> PipelinePlan:
    model = spec.get("model")
    if not isinstance(model, dict):
        raise ValueError("spec.model must be a mapping")
    graph = model.get("graph")
    if not isinstance(graph, list):
        raise ValueError("spec.model.graph must be a list")
    model_symbols = model.get("symbols")
    if model_symbols is not None and not isinstance(model_symbols, dict):
        raise ValueError("spec.model.symbols must be a mapping when present")
    model_config = model.get("config")
    if model_config is not None and not isinstance(model_config, dict):
        raise ValueError("spec.model.config must be a mapping when present")
    layers_var, layers_scope, total_layers = _find_primary_layer_loop(
        graph,
        model_symbols=model_symbols,
        model_config=model_config,
    )
    devices = available_pipeline_devices(requested_device)
    ranges = partition_layer_ranges(total_layers, len(devices))
    stages = tuple(
        PipelineStage(
            index=idx,
            device=devices[idx],
            layer_start=start,
            layer_stop=stop,
        )
        for idx, (start, stop) in enumerate(ranges)
    )
    used_devices = tuple(stage.device for stage in stages)
    return PipelinePlan(
        devices=used_devices,
        layers_var=layers_var,
        layers_scope=layers_scope,
        total_layers=total_layers,
        stages=stages,
    )


__all__ = [
    "PipelinePlan",
    "PipelineStage",
    "available_pipeline_devices",
    "build_pipeline_plan",
    "build_pipeline_stage_spec",
    "build_pipeline_stage_specs",
    "partition_layer_ranges",
]
