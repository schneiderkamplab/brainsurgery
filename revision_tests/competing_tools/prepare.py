#!/usr/bin/env python3
"""Create deterministic fixtures and rendered tool specifications."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import save_file

try:
    from .oracle import PROTOCOL_ID, load_model, model_state, rename_state, state_manifest
except ImportError:
    from oracle import PROTOCOL_ID, load_model, model_state, rename_state, state_manifest


HERE = Path(__file__).resolve().parent
CONFIG = {
    "architectures": ["GPT2LMHeadModel"],
    "model_type": "gpt2",
    "n_layer": 2,
    "n_embd": 4,
    "n_head": 2,
    "n_inner": 16,
    "n_positions": 8,
    "n_ctx": 8,
    "vocab_size": 8,
    "tie_word_embeddings": False,
    "torch_dtype": "float32",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-model",
        type=Path,
        help="Optional real local GPT-2-family model used for reportable-size fixtures",
    )
    parser.add_argument(
        "--source-revision",
        help="Required pinned revision when --source-model is supplied",
    )
    parser.add_argument("--source-id", help="Upstream model identifier for provenance")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_yaml(path: Path, value: Any) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def brain_plan(case_id: str, fixture_root: Path, output: Path) -> dict[str, Any]:
    if case_id == "R01":
        return {
            "inputs": [f"work::{fixture_root / 'rename' / 'model.safetensors'}"],
            "transforms": [
                {
                    "move": {
                        "from": r"work::layer\.(\d+)\.(weight|bias)",
                        "to": r"work::block.\1.\2",
                    }
                }
            ],
            "output": str(output),
        }
    if case_id == "M01":
        models = fixture_root / "models"
        return {
            "inputs": [f"a::{models / 'model_a'}", f"b::{models / 'model_b'}"],
            "transforms": [
                {"scale_": {"target": r"a::(.+)", "by": 0.25}},
                {"scale": {"from": r"b::(.+)", "to": r"a::__delta__.\1", "by": 0.75}},
                {"add_": {"from": r"a::__delta__\.(.+)", "to": r"a::\1"}},
                {"delete": {"target": r"a::__delta__\.(.+)"}},
            ],
            "output": str(output),
        }
    if case_id == "M02":
        models = fixture_root / "models"
        return {
            "inputs": [
                f"base::{models / 'base'}",
                f"ft1::{models / 'finetune_1'}",
                f"ft2::{models / 'finetune_2'}",
            ],
            "transforms": [
                {"copy": {"from": r"ft1::(.+)", "to": r"base::__delta1__.\1"}},
                {
                    "subtract_": {
                        "from": r"base::(?!__delta)(.+)",
                        "to": r"base::__delta1__.\1",
                    }
                },
                {"scale_": {"target": r"base::__delta1__\.(.+)", "by": 0.5}},
                {"copy": {"from": r"ft2::(.+)", "to": r"base::__delta2__.\1"}},
                {
                    "subtract_": {
                        "from": r"base::(?!__delta)(.+)",
                        "to": r"base::__delta2__.\1",
                    }
                },
                {"scale_": {"target": r"base::__delta2__\.(.+)", "by": 0.25}},
                {"add_": {"from": r"base::__delta1__\.(.+)", "to": r"base::\1"}},
                {"add_": {"from": r"base::__delta2__\.(.+)", "to": r"base::\1"}},
                {"delete": {"target": r"base::__delta(1|2)__\.(.+)"}},
            ],
            "output": str(output),
        }
    raise ValueError(f"unsupported BrainSurgery case: {case_id}")


def mergekit_config(case_id: str, fixture_root: Path) -> dict[str, Any]:
    models = fixture_root / "models"
    if case_id == "M01":
        return {
            "models": [
                {"model": str(models / "model_a"), "parameters": {"weight": 0.25}},
                {"model": str(models / "model_b"), "parameters": {"weight": 0.75}},
            ],
            "merge_method": "linear",
            "parameters": {"normalize": False},
            "dtype": "float32",
        }
    if case_id == "M02":
        return {
            "models": [
                {"model": str(models / "finetune_1"), "parameters": {"weight": 0.5}},
                {"model": str(models / "finetune_2"), "parameters": {"weight": 0.25}},
            ],
            "base_model": str(models / "base"),
            "merge_method": "task_arithmetic",
            "parameters": {"normalize": False, "rescale": False},
            "dtype": "float32",
        }
    raise ValueError(f"unsupported MergeKit case: {case_id}")


def derived_state(source: dict[str, torch.Tensor], offset: float) -> dict[str, torch.Tensor]:
    result = {}
    for name, tensor in source.items():
        if not tensor.is_floating_point():
            raise ValueError(f"real source tensor is not floating point: {name} ({tensor.dtype})")
        result[name] = tensor.to(torch.float32) + offset
    return result


def gpt2_merge_weight_names(
    source: dict[str, torch.Tensor], config: dict[str, Any]
) -> set[str]:
    names = set()
    for choices in (
        ("transformer.wte.weight", "wte.weight"),
        ("transformer.wpe.weight", "wpe.weight"),
        ("transformer.ln_f.weight", "ln_f.weight"),
        ("transformer.ln_f.bias", "ln_f.bias"),
        ("lm_head.weight",),
    ):
        present = [name for name in choices if name in source]
        if len(present) > 1:
            raise ValueError(f"source contains conflicting GPT-2 aliases: {present}")
        names.update(present)
    suffixes = (
        "attn.c_attn.weight",
        "attn.c_attn.bias",
        "attn.c_proj.weight",
        "attn.c_proj.bias",
        "ln_1.weight",
        "ln_1.bias",
        "ln_2.weight",
        "ln_2.bias",
        "mlp.c_fc.weight",
        "mlp.c_fc.bias",
        "mlp.c_proj.weight",
        "mlp.c_proj.bias",
    )
    for layer in range(int(config["n_layer"])):
        for suffix in suffixes:
            choices = (f"transformer.h.{layer}.{suffix}", f"h.{layer}.{suffix}")
            present = [name for name in choices if name in source]
            if len(present) != 1:
                raise ValueError(
                    f"expected one GPT-2 alias for layer {layer} {suffix}, found {present}"
                )
            names.add(present[0])
    return names


def canonicalize_gpt2_merge_state(
    source: dict[str, torch.Tensor], names: set[str]
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Return the shared MergeKit/BrainSurgery contract in HF canonical names."""
    canonical = {}
    aliases = {}
    for source_name in sorted(names):
        if source_name.startswith("h.") or source_name.startswith(("wte.", "wpe.", "ln_f.")):
            target_name = f"transformer.{source_name}"
        else:
            target_name = source_name
        if target_name in canonical:
            raise ValueError(f"canonical GPT-2 name collision: {target_name}")
        canonical[target_name] = source[source_name]
        aliases[source_name] = target_name
    return canonical, aliases


def prepare(
    root: Path,
    *,
    source_model: Path | None = None,
    source_revision: str | None = None,
    source_id: str | None = None,
) -> dict[str, Any]:
    if root.exists():
        raise FileExistsError(f"refusing to overwrite fixture directory: {root}")
    if source_model is not None and not source_revision:
        raise ValueError("--source-revision is required with --source-model")
    (root / "rename").mkdir(parents=True)
    if source_model is None:
        rename = rename_state()
        source = None
        fixture_id = "deterministic_tiny_gpt2_v1"
        config = CONFIG
        source_record = None
    else:
        source_model = source_model.resolve()
        source = load_model(source_model)
        rename = {
            f"layer.{index}.weight": tensor.clone()
            for index, (_name, tensor) in enumerate(sorted(source.items()))
        }
        rename["metadata.version"] = torch.tensor(1, dtype=torch.int64)
        offsets = {
            "base": 0.0,
            "model_a": 0.125,
            "model_b": -0.25,
            "finetune_1": 0.5,
            "finetune_2": -0.375,
        }
        config_path = source_model / "config.json"
        if not config_path.is_file():
            raise ValueError(f"source model lacks config.json: {source_model}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("model_type") != "gpt2":
            raise ValueError(
                f"reported-size protocol requires model_type=gpt2, found {config.get('model_type')!r}"
            )
        merge_names = gpt2_merge_weight_names(source, config)
        merge_source, canonical_name_mapping = canonicalize_gpt2_merge_state(
            source, merge_names
        )
        excluded_source = {name: source[name] for name in sorted(set(source) - merge_names)}
        checkpoint_paths = sorted(source_model.glob("*.safetensors"))
        index_path = source_model / "model.safetensors.index.json"
        if index_path.is_file():
            checkpoint_paths.append(index_path)
        source_record = {
            "model_id": source_id,
            "path": str(source_model),
            "revision": source_revision,
            "config_sha256": sha256_file(config_path),
            "checkpoint_files": {
                str(path.relative_to(source_model)): sha256_file(path)
                for path in checkpoint_paths
            },
            "tensor_manifest": state_manifest(source),
            "merge_tensor_manifest": state_manifest(merge_source),
            "canonical_name_mapping": canonical_name_mapping,
            "excluded_tensor_manifest": state_manifest(excluded_source),
            "exclusion_reason": (
                "Tensors absent from MergeKit's pinned GPT-2 architecture definition are "
                "excluded from arithmetic cases for a common input contract; R01 still "
                "covers all tensors."
            ),
        }
        fixture_id = f"real_shape_derived_v1_{source_revision}"
    save_file(rename, str(root / "rename" / "model.safetensors"))
    rename_manifest = state_manifest(rename)
    del rename
    model_manifests = {}
    variants = ("base", "model_a", "model_b", "finetune_1", "finetune_2")
    for variant in variants:
        state = (
            model_state(variant)
            if source is None
            else derived_state(merge_source, offsets[variant])
        )
        model_dir = root / "models" / variant
        model_dir.mkdir(parents=True)
        save_file(state, str(model_dir / "model.safetensors"))
        (model_dir / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        model_manifests[variant] = state_manifest(state)
        del state
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "fixture_id": fixture_id,
        "source_model": source_record,
        "rename": rename_manifest,
        "models": model_manifests,
        "files": {
            str(path.relative_to(root)): sha256_file(path)
            for path in sorted(root.rglob("*"))
            if path.is_file()
        },
    }
    (root / "fixture_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    args = parse_args()
    if (args.source_model is None) != (args.source_revision is None):
        raise SystemExit("--source-model and --source-revision must be supplied together")
    manifest = prepare(
        args.output.resolve(),
        source_model=args.source_model,
        source_revision=args.source_revision,
        source_id=args.source_id,
    )
    print(f"wrote fixture {manifest['fixture_id']} to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
