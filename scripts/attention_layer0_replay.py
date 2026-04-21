#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch

from brainsurgery.synapse.axon import (
    lower_axon_program_to_synapse_spec,
    parse_axon_program_from_path,
    tokenize_prompts,
)
from brainsurgery.synapse.axon_test import (
    _load_generated_class,
    _load_state_dict,
    _resolve_safetensors_paths,
)
from brainsurgery.synapse.codegen import emit_model_code_from_synapse_spec


def _attention_basic(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keep: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    scores = torch.matmul(q, k.transpose(2, 3))
    scores = scores * scale
    additive = torch.where(
        keep,
        torch.zeros_like(scores),
        torch.full_like(scores, torch.finfo(scores.dtype).min),
    )
    probs = torch.softmax(scores + additive, dim=-1)
    return torch.matmul(probs, v)


def _extract_layer0_attention_inputs(trace_ops: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    layer_items = [
        item
        for item in trace_ops
        if isinstance(item, dict) and "model.layers.0" in str(item.get("node_path", ""))
    ]
    attn_idx = -1
    for idx, item in enumerate(layer_items):
        if str(item.get("op", "")) == "attention":
            attn_idx = idx
            break
    if attn_idx < 0:
        raise ValueError("Could not find primitive attention op in layer 0 trace")

    prefix = layer_items[:attn_idx]
    attn_item = layer_items[attn_idx]
    attn_out = attn_item.get("tensor")
    if not torch.is_tensor(attn_out):
        raise ValueError("Layer 0 attention output tensor missing in trace")

    q: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None

    for item in reversed(prefix):
        bind = str(item.get("bind", ""))
        tensor = item.get("tensor")
        if not torch.is_tensor(tensor):
            continue
        if q is None and bind == "q":
            q = tensor
        if mask is None and bind == "mask":
            mask = tensor
        if q is not None and mask is not None:
            break

    repeats = [
        item
        for item in prefix
        if str(item.get("op", "")) == "repeat"
        and str(item.get("bind", "")) == "out_0"
        and torch.is_tensor(item.get("tensor"))
    ]
    if len(repeats) >= 2:
        k = repeats[-2]["tensor"]
        v = repeats[-1]["tensor"]
    else:
        for item in reversed(prefix):
            bind = str(item.get("bind", ""))
            tensor = item.get("tensor")
            if not torch.is_tensor(tensor):
                continue
            if k is None and bind == "k":
                k = tensor
            elif v is None and bind == "v":
                v = tensor
            if k is not None and v is not None:
                break

    if q is None or k is None or v is None or mask is None:
        raise ValueError("Failed to recover q/k/v/mask from layer 0 trace")

    return {
        "q": q,
        "k": k,
        "v": v,
        "mask": mask,
        "attn_out_primitive": attn_out,
    }


def _resolve_dtype(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key == "float32":
        return torch.float32
    if key == "bfloat16":
        return torch.bfloat16
    if key == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Replay layer-0 primitive attention inputs through attention_basic math and compare outputs."
    )
    ap.add_argument(
        "--axon-file",
        type=Path,
        default=Path("brainsurgery/synapse/models/llama3/generic-llama3.axon"),
    )
    ap.add_argument(
        "--weights",
        type=Path,
        default=Path("models/meta-llama/Llama-3.2-1B"),
    )
    ap.add_argument("--device", type=str, default="cuda:5")
    ap.add_argument("--dtype", type=str, default="float32")
    ap.add_argument("--max-len", type=int, default=32)
    ap.add_argument("--prompt", type=str, default="hello world")
    args = ap.parse_args()

    axon_file = args.axon_file.resolve()
    weights = args.weights.resolve()
    dtype = _resolve_dtype(args.dtype)
    device = torch.device(args.device)

    modules = parse_axon_program_from_path(axon_file)
    lowered_spec = lower_axon_program_to_synapse_spec(modules)

    with TemporaryDirectory(prefix="layer0_attn_replay_") as td:
        tmp = Path(td)
        generated_py = tmp / "generated_model.py"
        generated_py.write_text(
            emit_model_code_from_synapse_spec(lowered_spec, class_name="Layer0ReplayModel"),
            encoding="utf-8",
        )
        model_cls = _load_generated_class(generated_py, "Layer0ReplayModel")

        state = _load_state_dict(
            _resolve_safetensors_paths(weights),
            device=device,
            dtype=dtype,
        )
        model = model_cls.from_state_dict(state).to(device).eval()
        setattr(model, "_trace_enabled", True)
        if hasattr(model, "_reset_trace"):
            model._reset_trace()

        _, ids_cpu, attn_mask_cpu = tokenize_prompts(
            prompts=[args.prompt],
            tokenizer_source=str(weights),
            tokenizer_fallback=weights.name,
            device=torch.device("cpu"),
            max_len=int(args.max_len),
            lowered_spec=lowered_spec,
            trust_remote_code=False,
        )
        syn_inputs: dict[str, Any] = {
            "input_ids": ids_cpu.to(device),
            "past_kv": None,
            "use_cache": False,
        }
        if attn_mask_cpu is not None:
            syn_inputs["attn_mask"] = attn_mask_cpu.to(device)

        with torch.no_grad():
            _ = model(**syn_inputs)

        extracted = _extract_layer0_attention_inputs(list(getattr(model, "trace_ops", [])))
        q = extracted["q"]
        k = extracted["k"]
        v = extracted["v"]
        mask = extracted["mask"]
        primitive = extracted["attn_out_primitive"]

        keep = mask == 0
        scale = 1.0 / math.sqrt(float(q.shape[-1]))
        replay = _attention_basic(q=q, k=k, v=v, keep=keep, scale=scale)

        abs_diff = (primitive - replay).abs()
        denom = replay.abs().clamp_min(1e-12)
        rel_diff = abs_diff / denom
        top1_eq = torch.equal(
            torch.argmax(primitive.float(), dim=-1),
            torch.argmax(replay.float(), dim=-1),
        )

        print(f"axon_file={axon_file}")
        print(f"weights={weights}")
        print(f"device={device}")
        print(f"dtype={dtype}")
        print(f"layer0_q_shape={tuple(q.shape)}")
        print(f"layer0_k_shape={tuple(k.shape)}")
        print(f"layer0_v_shape={tuple(v.shape)}")
        print(f"layer0_mask_shape={tuple(mask.shape)}")
        print(f"masked_top1_eq={bool(top1_eq)}")
        print(f"max_abs_diff={float(abs_diff.max().item()):.12g}")
        print(f"max_rel_diff={float(rel_diff.max().item()):.12g}")


if __name__ == "__main__":
    main()
