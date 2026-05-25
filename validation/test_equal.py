from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from brainsurgery.core import TransformControl, get_transform
from brainsurgery.engine import create_state_dict_provider
from brainsurgery.engine.runtime_flags_policy import (
    RuntimeFlagLifecycleScope,
    reset_runtime_flags_for_scope,
)

from validation import pytorch_example as pe

PLAN_PATH = PROJECT_ROOT / "validation" / "validation.yaml"
PYTORCH_OUTPUT_DIR = PROJECT_ROOT / "models" / "test" / "validation_pytorch"


def _load_plan(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("validation plan must be a YAML mapping")
    return data


def _resolve_input_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _clone_tensor_map(src: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in src.items()}


def _provider_snapshot(provider: Any) -> pe.AliasStore:
    store: pe.AliasStore = {}
    for alias, state_dict in provider.state_dicts.items():
        store[alias] = _clone_tensor_map(dict(state_dict.items()))
    return store


def _save_pytorch_output(store: pe.AliasStore, *, output_dir: Path, alias: str = "model") -> Path:
    if alias not in store:
        raise KeyError(f"missing alias in pytorch store: {alias}")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "model.pt"
    torch.save(store[alias], out_path)
    return out_path


def _assert_store_equal(tool_store: pe.AliasStore, pyt_store: pe.AliasStore, *, step: str) -> None:
    tool_aliases = set(tool_store)
    pyt_aliases = set(pyt_store)
    if tool_aliases != pyt_aliases:
        raise AssertionError(
            f"alias mismatch at {step}: tool={sorted(tool_aliases)} pytorch={sorted(pyt_aliases)}"
        )

    for alias in sorted(tool_aliases):
        tool_keys = set(tool_store[alias])
        pyt_keys = set(pyt_store[alias])
        if tool_keys != pyt_keys:
            raise AssertionError(
                f"key mismatch at {step} for {alias}: tool={sorted(tool_keys)} pytorch={sorted(pyt_keys)}"
            )
        for key in sorted(tool_keys):
            a = tool_store[alias][key]
            b = pyt_store[alias][key]
            if a.shape != b.shape:
                raise AssertionError(
                    f"shape mismatch at {step} for {alias}::{key}: {tuple(a.shape)} != {tuple(b.shape)}"
                )
            if a.dtype != b.dtype:
                raise AssertionError(
                    f"dtype mismatch at {step} for {alias}::{key}: {a.dtype} != {b.dtype}"
                )
            a_cpu = a.detach().cpu()
            b_cpu = b.detach().cpu()
            if a_cpu.dtype.is_floating_point or a_cpu.dtype.is_complex:
                same = bool(torch.allclose(a_cpu, b_cpu, atol=1e-6, rtol=1e-6))
            else:
                same = bool(torch.equal(a_cpu, b_cpu))
            if not same:
                raise AssertionError(f"value mismatch at {step} for {alias}::{key}")


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype in test dispatcher: {name}") from exc


def _pytorch_single_tensor_path(path_text: str) -> Path:
    path = Path(path_text)
    # Keep tool and plain-PyTorch save/load artifacts independent.
    if path.name == "brainsurgery_validation_x.safetensors":
        return path.with_name("brainsurgery_validation_x_pytorch.pt")
    return path


def _eval_assert_expr(store: pe.AliasStore, expr: dict[str, Any]) -> None:
    if "not" in expr:
        inner = expr["not"]
        if not isinstance(inner, dict):
            raise AssertionError("assert.not expects a mapping")
        try:
            _eval_assert_expr(store, inner)
        except AssertionError:
            return
        raise AssertionError("assert.not failed: inner expression evaluated to true")

    if "shape" in expr:
        payload = expr["shape"]
        pe.assert_shape(store, payload["of"], tuple(payload["is"]))
        return

    if "iszero" in expr:
        pe.assert_iszero(store, expr["iszero"])
        return

    if "exists" in expr:
        pe.assert_exists(store, expr["exists"])
        return

    if "dtype" in expr:
        payload = expr["dtype"]
        pe.assert_dtype(store, payload["of"], _dtype_from_name(payload["is"]))
        return

    if "equal" in expr:
        payload = expr["equal"]
        left = payload["left"]
        right = payload["right"]
        eps = payload.get("eps")
        if isinstance(right, str) and "\\1" in right:
            pe.assert_equal_regex_map(store, left, right, default_right_alias="model", eps=eps)
        else:
            pe.assert_equal(store, left, right, eps=eps)
        return

    raise AssertionError(f"unsupported assert payload in test dispatcher: {expr}")


def _apply_pytorch_transform(
    store: pe.AliasStore,
    raw_transform: dict[str, Any],
    *,
    flags: pe.RuntimeFlags,
    initial_model_snapshot: dict[str, torch.Tensor],
) -> bool:
    op_name, payload = next(iter(raw_transform.items()))

    if op_name == "set":
        if not payload:
            pe.transform_set(flags)
        else:
            pe.transform_set(
                flags,
                dry_run=payload.get("dry-run"),
                preview=payload.get("preview"),
                verbose=payload.get("verbose"),
            )
        return True

    if op_name == "help":
        _ = pe.transform_help_diff()
        return True

    if op_name == "dump":
        _ = pe.transform_dump_compact(store, alias="model")
        return True

    if op_name == "prefixes":
        mode = payload.get("mode", "list")
        if mode == "list":
            _ = pe.transform_prefixes_list(store)
        elif mode == "add":
            pe.transform_prefixes_add(store, payload["alias"])
        elif mode == "remove":
            pe.transform_prefixes_remove(store, payload["alias"])
        elif mode == "rename":
            pe.transform_prefixes_rename(store, payload["from"], payload["to"])
        else:
            raise ValueError(f"unsupported prefixes mode: {mode}")
        return True

    if op_name == "zeroes":
        pe.transform_zeroes(store, payload["target"], tuple(payload["shape"]))
        return True

    if op_name == "ones":
        pe.transform_ones(store, payload["target"], tuple(payload["shape"]))
        return True

    if op_name == "rand":
        pe.transform_rand(
            store,
            payload["target"],
            tuple(payload["shape"]),
            distribution=payload.get("distribution", "uniform"),
            low=payload.get("low", 0.0),
            high=payload.get("high", 1.0),
            mean=payload.get("mean", 0.0),
            std=payload.get("std", 1.0),
            seed=payload.get("seed"),
        )
        return True

    if op_name == "assert":
        if not isinstance(payload, dict):
            raise AssertionError("assert payload must be a mapping")
        _eval_assert_expr(store, payload)
        return True

    if op_name == "copy":
        pe.transform_copy(store, payload["from"], payload["to"])
        return True

    if op_name == "move":
        pe.transform_move(store, payload["from"], payload["to"])
        return True

    if op_name == "delete":
        _ = pe.transform_delete(store, payload["target"])
        return True

    if op_name == "assign":
        pe.transform_assign(store, payload["from"], payload["to"])
        return True

    if op_name == "add":
        pe.transform_add(store, payload["from_a"], payload["from_b"], payload["to"])
        return True

    if op_name == "subtract":
        pe.transform_subtract(store, payload["from_a"], payload["from_b"], payload["to"])
        return True

    if op_name == "multiply":
        pe.transform_multiply(store, payload["from_a"], payload["from_b"], payload["to"])
        return True

    if op_name == "add_":
        pe.transform_add_(store, payload["from"], payload["to"])
        return True

    if op_name == "subtract_":
        pe.transform_subtract_(store, payload["from"], payload["to"])
        return True

    if op_name == "scale":
        pe.transform_scale(store, payload["from"], payload["to"], float(payload["by"]))
        return True

    if op_name == "scale_":
        pe.transform_scale_(store, payload["target"], float(payload["by"]))
        return True

    if op_name == "fill":
        mode = payload["mode"]
        if mode == "constant":
            pe.transform_fill_constant(store, payload["from"], payload["to"], float(payload["value"]))
        elif mode == "rand":
            pe.transform_fill_rand(
                store,
                payload["from"],
                payload["to"],
                distribution=payload.get("distribution", "uniform"),
                seed=payload.get("seed"),
                low=payload.get("low", 0.0),
                high=payload.get("high", 1.0),
                mean=payload.get("mean", 0.0),
                std=payload.get("std", 1.0),
            )
        else:
            raise ValueError(f"unsupported fill mode: {mode}")
        return True

    if op_name == "fill_":
        pe.transform_fill_tensor_(store, payload["target"], payload["values"])
        return True

    if op_name == "clamp":
        pe.transform_clamp(
            store,
            payload["from"],
            payload["to"],
            min_value=float(payload["min"]),
            max_value=float(payload["max"]),
        )
        return True

    if op_name == "clamp_":
        pe.transform_clamp_(
            store,
            payload["target"],
            min_value=float(payload["min"]),
            max_value=float(payload["max"]),
        )
        return True

    if op_name == "cast":
        pe.transform_cast(store, payload["from"], payload["to"], dtype=_dtype_from_name(payload["dtype"]))
        return True

    if op_name == "cast_":
        pe.transform_cast_(store, payload["target"], dtype=_dtype_from_name(payload["to"]))
        return True

    if op_name == "split":
        pe.transform_split(store, payload["from"], payload["to"], payload["sizes"], dim=int(payload.get("dim", 0)))
        return True

    if op_name == "concat":
        pe.transform_concat(store, payload["from"], payload["to"], dim=int(payload.get("dim", 0)))
        return True

    if op_name == "reshape":
        pe.transform_reshape(store, payload["from"], payload["to"], tuple(payload["shape"]))
        return True

    if op_name == "reshape_":
        pe.transform_reshape_(store, payload["target"], tuple(payload["shape"]))
        return True

    if op_name == "permute":
        pe.transform_permute(store, payload["from"], payload["to"], tuple(payload["order"]))
        return True

    if op_name == "matmul":
        pe.transform_matmul(store, payload["from_a"], payload["from_b"], payload["to"])
        return True

    if op_name == "phlora_":
        pe.transform_phlora_(store, payload["target"], int(payload["rank"]))
        return True

    if op_name == "phlora":
        pe.transform_phlora(
            store,
            payload["target"],
            payload["target_a"],
            payload["target_b"],
            int(payload["rank"]),
            delete_original=bool(payload.get("delete_original", True)),
        )
        return True

    if op_name == "save":
        path = _pytorch_single_tensor_path(payload["path"])
        pe.transform_save_tensor(store, payload["target"], path)
        return True

    if op_name == "load":
        if "to" in payload:
            path = _pytorch_single_tensor_path(payload["path"])
            pe.transform_load_tensor(store, path, payload["to"])
            return True
        alias = payload["alias"]
        # In this validation plan alias-load source is the same model checkpoint loaded at start.
        store[alias] = _clone_tensor_map(initial_model_snapshot)
        return True

    if op_name == "diff":
        _ = pe.transform_diff_aliases(
            store,
            payload["left_alias"],
            payload["right_alias"],
            eps=payload.get("eps"),
        )
        return True

    if op_name == "execute":
        nested = payload.get("transforms", [])
        steps = [
            (lambda local_store, item=item: _apply_pytorch_transform(
                local_store,
                item,
                flags=flags,
                initial_model_snapshot=initial_model_snapshot,
            ))
            for item in nested
        ]
        pe.transform_execute(store, steps)
        return True

    if op_name == "exit":
        _ = pe.transform_exit()
        return False

    raise ValueError(f"unsupported transform in pytorch dispatcher: {op_name}")


def test_yaml_plan_matches_plain_pytorch_equivalent() -> None:
    if not PLAN_PATH.exists():
        raise FileNotFoundError(f"missing plan file: {PLAN_PATH}")

    plan = _load_plan(PLAN_PATH)
    raw_inputs = plan.get("inputs", [])
    if not isinstance(raw_inputs, list) or len(raw_inputs) != 1:
        raise AssertionError("this equivalence test expects exactly one input")
    input_path = _resolve_input_path(str(raw_inputs[0]))
    if not input_path.exists():
        raise FileNotFoundError(f"missing input checkpoint: {input_path}")

    transforms = plan.get("transforms", [])
    if not isinstance(transforms, list):
        raise AssertionError("plan.transforms must be a list")

    reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)

    provider = create_state_dict_provider(
        provider="inmemory",
        model_paths={"model": input_path},
        max_io_workers=8,
        arena_root=PROJECT_ROOT / ".brainsurgery",
        arena_segment_size="1GB",
    )

    try:
        tool_model_sd = provider.get_state_dict("model")
        initial_model_snapshot = _clone_tensor_map(dict(tool_model_sd.items()))
        pytorch_store: pe.AliasStore = {"model": _clone_tensor_map(initial_model_snapshot)}
        flags = pe.RuntimeFlags()

        print(f"[INFO] Number of transforms to execute: {len(transforms)}")

        for index, raw in enumerate(transforms):
            if not isinstance(raw, dict) or len(raw) != 1:
                raise AssertionError(f"invalid transform entry at index {index}: {raw!r}")

            op_name = next(iter(raw.keys()))
            payload = raw[op_name]
            step = f"#{index}:{op_name}"

            transform = get_transform(op_name)
            spec = transform.compile(payload, default_model="model")
            tool_result = transform.apply(spec, provider)

            continue_pyt = _apply_pytorch_transform(
                pytorch_store,
                raw,
                flags=flags,
                initial_model_snapshot=initial_model_snapshot,
            )

            tool_store = _provider_snapshot(provider)
            _assert_store_equal(tool_store, pytorch_store, step=step)

            tool_continue = tool_result.control == TransformControl.CONTINUE
            if continue_pyt != tool_continue:
                raise AssertionError(
                    f"control-flow mismatch at {step}: tool_continue={tool_continue}, "
                    f"pytorch_continue={continue_pyt}"
                )
            print(f"[PASS] {step}")
            if not tool_continue:
                break

        pytorch_output_path = _save_pytorch_output(
            pytorch_store,
            output_dir=PYTORCH_OUTPUT_DIR,
            alias="model",
        )
        print(f"[INFO] Saved plain-PyTorch output to: {pytorch_output_path}")
    finally:
        provider.close()
        reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)


if __name__ == "__main__":
    test_yaml_plan_matches_plain_pytorch_equivalent()
    print("YAML/tool and plain-PyTorch execution are equivalent for validation/validation.yaml")
