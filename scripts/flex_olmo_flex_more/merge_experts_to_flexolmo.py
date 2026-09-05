import argparse
from collections import defaultdict
import json
import logging
import os
from pathlib import Path
import tempfile

import torch
from transformers import AutoConfig, AutoModelForCausalLM
try:
    from transformers import FlexOlmoConfig, FlexOlmoForCausalLM
except ImportError:
    FlexOlmoConfig = None
    FlexOlmoForCausalLM = None
try:
    from transformers import OlmoeConfig, OlmoeForCausalLM
except ImportError:
    try:
        from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
        from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM
    except ImportError:
        OlmoeConfig = None
        OlmoeForCausalLM = None

try:
    from olmo_core.utils import prepare_cli_environment
except ImportError:
    def prepare_cli_environment(*args, **kwargs):
        return None


log = logging.getLogger(__name__)


def dtype_from_string(s):
    match s.lower():
        case "float32" | "fp32":
            return torch.float32
        case "float16" | "fp16":
            return torch.float16
        case "bfloat16" | "bf16":
            return torch.bfloat16
        case _:
            raise ValueError(f"Unsupported dtype string: {s}")


def load_raw_config_dict(path: str) -> dict:
    with open(Path(path) / "config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def can_use_native_flex_olmo() -> bool:
    return FlexOlmoConfig is not None and FlexOlmoForCausalLM is not None


def can_use_olmoe_fallback() -> bool:
    return OlmoeConfig is not None and OlmoeForCausalLM is not None


def prepare_olmoe_compat_dir(model_path: str) -> str:
    src = Path(model_path)
    config = load_raw_config_dict(model_path)
    if config.get("model_type") != "flex_olmo":
        return model_path

    compat_root = Path(tempfile.mkdtemp(prefix="flex_olmo_compat_", dir="/tmp"))
    for name in src.iterdir():
        if name.name == "config.json":
            continue
        os.symlink(name, compat_root / name.name)

    compat_config = dict(config)
    compat_config["model_type"] = "olmoe"
    compat_config["architectures"] = ["OlmoeForCausalLM"]
    with open(compat_root / "config.json", "w", encoding="utf-8") as f:
        json.dump(compat_config, f, indent=2, sort_keys=True)
        f.write("\n")

    return str(compat_root)


def load_config(path: str):
    config_dict = load_raw_config_dict(path)
    if config_dict.get("model_type") == "flex_olmo" and can_use_native_flex_olmo():
        log.info("Loading native FlexOlmo config from %s", path)
        return FlexOlmoConfig.from_dict(config_dict)
    if config_dict.get("model_type") == "flex_olmo" and can_use_olmoe_fallback():
        log.info("Loading direct Olmoe fallback config from %s", path)
        compat_config = dict(config_dict)
        compat_config["model_type"] = "olmoe"
        compat_config["architectures"] = ["OlmoeForCausalLM"]
        return OlmoeConfig.from_dict(compat_config)
    log.info("Loading compat AutoConfig from %s", path)
    return AutoConfig.from_pretrained(prepare_olmoe_compat_dir(path), trust_remote_code=True)


def load_model(path: str, dtype):
    config_dict = load_raw_config_dict(path)
    if config_dict.get("model_type") == "flex_olmo" and can_use_native_flex_olmo():
        log.info("Loading native FlexOlmo checkpoint from %s", path)
        try:
            return FlexOlmoForCausalLM.from_pretrained(
                path,
                torch_dtype=dtype,
            )
        except TypeError:
            return FlexOlmoForCausalLM.from_pretrained(
                path,
                dtype=dtype,
            )
    if config_dict.get("model_type") == "flex_olmo" and can_use_olmoe_fallback():
        log.info("Loading direct Olmoe fallback checkpoint from %s", path)
        compat_path = prepare_olmoe_compat_dir(path)
        try:
            return OlmoeForCausalLM.from_pretrained(
                compat_path,
                torch_dtype=dtype,
            )
        except TypeError:
            return OlmoeForCausalLM.from_pretrained(
                compat_path,
                dtype=dtype,
            )
    log.info("Loading compat AutoModel checkpoint from %s", path)
    compat_path = prepare_olmoe_compat_dir(path)
    try:
        return AutoModelForCausalLM.from_pretrained(
            compat_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            compat_path,
            dtype=dtype,
            trust_remote_code=True,
        )


def save_target_config(target_path: str, source_path: str, num_experts: int, dtype) -> None:
    config = load_raw_config_dict(source_path)
    config["num_experts"] = num_experts
    config["torch_dtype"] = str(dtype).replace("torch.", "")
    if "dtype" in config:
        config["dtype"] = str(dtype).replace("torch.", "")
    with open(Path(target_path) / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")


def raise_shared_key_mismatch(
    expert_index: int,
    model_path: str,
    moe_key: str,
    expected: torch.Tensor,
    actual: torch.Tensor,
):
    same_shape = tuple(expected.shape) == tuple(actual.shape)
    same_dtype = expected.dtype == actual.dtype
    max_abs_diff = None
    mean_abs_diff = None
    if same_shape:
        diff = (expected.to(dtype=torch.float64) - actual.to(dtype=torch.float64)).abs()
        max_abs_diff = float(diff.max().item()) if diff.numel() > 0 else 0.0
        mean_abs_diff = float(diff.mean().item()) if diff.numel() > 0 else 0.0

    raise AssertionError(
        "Shared key mismatch detected during expert merge: "
        f"expert={expert_index}, "
        f"path={model_path}, "
        f"moe_key={moe_key}, "
        f"expected_shape={tuple(expected.shape)}, "
        f"actual_shape={tuple(actual.shape)}, "
        f"expected_dtype={expected.dtype}, "
        f"actual_dtype={actual.dtype}, "
        f"same_shape={same_shape}, "
        f"same_dtype={same_dtype}, "
        f"max_abs_diff={max_abs_diff}, "
        f"mean_abs_diff={mean_abs_diff}"
    )


def copy_or_validate_tensor(
    moe_state_dict: dict[str, torch.Tensor],
    filled_keys: defaultdict[str, int],
    moe_key: str,
    source_tensor: torch.Tensor,
    expert_index: int,
    model_path: str,
):
    if expert_index:
        if not torch.equal(moe_state_dict[moe_key], source_tensor):
            raise_shared_key_mismatch(
                expert_index,
                model_path,
                moe_key,
                moe_state_dict[moe_key],
                source_tensor,
            )
        return False

    moe_state_dict[moe_key] = source_tensor
    filled_keys[moe_key] += 1
    return True


def copy_packed_expert_tensors(
    expert_key: str,
    weights: torch.Tensor,
    expert_index: int,
    model_path: str,
    moe_state_dict: dict[str, torch.Tensor],
    filled_keys: defaultdict[str, int],
) -> list[str]:
    processed_keys: list[str] = []

    if "gate_up_proj" in expert_key:
        if expert_index:
            if not torch.equal(moe_state_dict[expert_key][0], weights[0]):
                raise_shared_key_mismatch(
                    expert_index,
                    model_path,
                    expert_key,
                    moe_state_dict[expert_key][0],
                    weights[0],
                )
            moe_state_dict[expert_key][expert_index] = weights[1]
        else:
            moe_state_dict[expert_key][0] = weights[0]
        filled_keys[expert_key] += 1
        processed_keys.append(expert_key)
    elif "down_proj" in expert_key:
        if expert_index:
            if not torch.equal(moe_state_dict[expert_key][0], weights[0]):
                raise_shared_key_mismatch(
                    expert_index,
                    model_path,
                    expert_key,
                    moe_state_dict[expert_key][0],
                    weights[0],
                )
            moe_state_dict[expert_key][expert_index] = weights[1]
        else:
            moe_state_dict[expert_key][0] = weights[0]
        filled_keys[expert_key] += 1
        processed_keys.append(expert_key)
    else:
        raise AssertionError(f"Unexpected packed expert key {expert_key}")

    return processed_keys


def parse_args():
    parser = argparse.ArgumentParser(description="Merge ranked 2x7B experts into one FlexOlmo-style MoE model")
    parser.add_argument("target", help="Target path to save the merged model")
    parser.add_argument("models", nargs="+", help="List of expert model paths to merge")
    parser.add_argument("--device", default="cpu", help="Device to load the models on")
    parser.add_argument("--dtype", default="bfloat16", help="Data type to load the models with")
    return parser.parse_args()


def main():
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    prepare_cli_environment()
    args = parse_args()
    expert_paths = args.models
    target_path = args.target
    device = torch.device(args.device)
    dtype = dtype_from_string(args.dtype)
    log.info(f"Building model config from {expert_paths[0]} with {len(expert_paths)} experts")
    model_config = load_config(expert_paths[0])
    model_config.num_experts = len(expert_paths)
    if hasattr(model_config, "dtype"):
        model_config.dtype = str(dtype).replace("torch.", "")
    setattr(model_config, "torch_dtype", dtype)
    log.info(f"Building the MoE model on {device} with dtype {dtype}")
    with torch.device(device):
        if getattr(model_config, "model_type", None) == "flex_olmo" and can_use_native_flex_olmo():
            model = FlexOlmoForCausalLM(model_config)
        elif getattr(model_config, "model_type", None) == "olmoe" and can_use_olmoe_fallback():
            model = OlmoeForCausalLM(model_config)
        else:
            model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)
    log.info(f"Model loaded on {device} with dtype {dtype}")
    log.info(model)
    moe_state_dict = model.state_dict()
    filled_keys = defaultdict(int)

    for expert, path in enumerate(expert_paths):
        log.info(f"Loading model from {path} as expert {expert} on {device} with dtype {dtype}")
        with torch.device(device):
            expert_model = load_model(path, dtype=dtype)
        log.info(expert_model)
        assert expert_model.config.num_experts == 2, (
            f"Expert model at {path} has num_experts={expert_model.config.num_experts}, expected 2"
        )
        expert_state_dict = expert_model.state_dict()
        log.info(f"Expert {expert} model loaded")

        for expert_key, weights in expert_state_dict.items():
            if ".experts." in expert_key and (
                ".experts.gate_up_proj" in expert_key or ".experts.down_proj" in expert_key
            ):
                processed_keys = copy_packed_expert_tensors(
                    expert_key,
                    weights,
                    expert,
                    path,
                    moe_state_dict,
                    filled_keys,
                )
                for processed_key in processed_keys:
                    log.info(f"Packed expert key {expert_key} populated MoE model key {processed_key}")
                continue

            moe_key = expert_key
            if ".experts.0." in expert_key:
                if expert:
                    if not torch.equal(moe_state_dict[moe_key], weights):
                        raise_shared_key_mismatch(
                            expert,
                            path,
                            moe_key,
                            moe_state_dict[moe_key],
                            weights,
                        )
                    moe_key = None
            elif ".experts.1." in expert_key:
                if expert:
                    moe_key = expert_key.replace(".experts.1.", f".experts.{expert}.")
                else:
                    moe_key = None
            elif ".mlp.gate." in expert_key:
                if expert:
                    if not torch.equal(moe_state_dict[moe_key][:1, :], weights[:1, :]):
                        raise AssertionError(
                            f"Gate weights for expert 0 are different for expert and MoE model: {moe_key}"
                        )
                    moe_state_dict[moe_key][expert:expert + 1, :] = weights[1:2, :]
                else:
                    moe_state_dict[moe_key][:1, :] = weights[:1, :]
                filled_keys[moe_key] += 1
                moe_key = None
            elif expert:
                if not torch.equal(moe_state_dict[moe_key], weights):
                    raise_shared_key_mismatch(
                        expert,
                        path,
                        moe_key,
                        moe_state_dict[moe_key],
                        weights,
                    )
                moe_key = None

            if moe_key:
                moe_state_dict[moe_key] = weights
                filled_keys[moe_key] += 1
                log.info(f"Key {expert_key} copied from expert {expert} to MoE model key {moe_key}")
            else:
                log.info(f"Key {expert_key} has not been copied from expert {expert}")

        del expert_state_dict

    assert set(moe_state_dict.keys()) == set(filled_keys.keys()), (
        f"Not all keys have been filled: missing {set(moe_state_dict.keys()) - set(filled_keys.keys())}"
    )
    multi_fill_patterns = (".mlp.gate.", ".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")
    assert all(
        count == 1 for key, count in filled_keys.items() if not any(pattern in key for pattern in multi_fill_patterns)
    ), (
        "Some single-fill keys have been filled multiple times: "
        f"{ {key: count for key, count in filled_keys.items() if not any(pattern in key for pattern in multi_fill_patterns) and count != 1} }"
    )
    assert all(
        count == len(expert_paths) for key, count in filled_keys.items() if any(pattern in key for pattern in multi_fill_patterns)
    ), (
        "Not all multi-fill expert/router keys have been filled correctly: "
        f"{ {key: count for key, count in filled_keys.items() if any(pattern in key for pattern in multi_fill_patterns) and count != len(expert_paths)} }"
    )
    log.info(f"Saving the merged model to {target_path}")
    model.save_pretrained(target_path, state_dict=moe_state_dict)
    save_target_config(target_path, expert_paths[0], len(expert_paths), dtype)
    log.info(f"Model saved to {target_path}")


if __name__ == "__main__":
    main()
