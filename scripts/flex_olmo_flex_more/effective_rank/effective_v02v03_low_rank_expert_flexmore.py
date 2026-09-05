import json
import logging
import math
import os
import pickle

import torch
import typer
from transformers import FlexMoREConfig, FlexMoREForCausalLM, FlexOlmoConfig, FlexOlmoForCausalLM

from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)


MODULE_WEIGHTS = {
    "down_proj": 1.00,
    "gate_proj": 1.25,
    "up_proj": 1.10,
}


def compute_effective_rank(
    singular_values: torch.Tensor,
    probability_threshold: float = 0.0,
) -> tuple[int, int, int]:
    singular_values = singular_values.to(dtype=torch.float64)

    norm = singular_values.sum()
    if norm <= 0:
        return 1, 0, 0

    probabilities = singular_values / norm
    nonzero_probabilities = probabilities[probabilities > probability_threshold]
    probability_norm = nonzero_probabilities.sum()
    if probability_norm <= 0:
        return 1, singular_values.numel(), 0

    nonzero_probabilities = nonzero_probabilities / probability_norm
    entropy = -(nonzero_probabilities * nonzero_probabilities.log()).sum()
    effective_rank = torch.exp(entropy).item()
    rank = max(1, min(singular_values.numel(), math.ceil(effective_rank)))
    return rank, singular_values.numel(), nonzero_probabilities.numel()


def infer_layer_index(model_key: str) -> int:
    return int(model_key.split(".")[2])


def infer_module_name(model_key: str) -> str:
    return model_key.split(".")[-1]


def flatten_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.to(dtype=torch.float64).reshape(-1)
    b_flat = b.to(dtype=torch.float64).reshape(-1)
    denom = torch.linalg.norm(a_flat) * torch.linalg.norm(b_flat)
    if denom <= 0:
        return 0.0
    return float((torch.dot(a_flat, b_flat) / denom).item())


def normalize_feature(value: float, min_value: float, max_value: float) -> float:
    if math.isclose(min_value, max_value):
        return 0.5
    normalized = (value - min_value) / (max_value - min_value)
    return max(0.0, min(1.0, float(normalized)))


def compute_layer_weight(
    layer_index: int,
    num_layers: int,
    early_weight: float,
    mid_weight: float,
    late_weight: float,
    last_weight: float,
) -> float:
    last_layer = num_layers - 1
    if layer_index == last_layer:
        return last_weight

    early_end = max(0, int(math.floor(0.25 * last_layer)))
    mid_end = max(early_end + 1, int(math.floor(0.75 * last_layer)))

    if layer_index <= early_end:
        return early_weight
    if layer_index <= mid_end:
        return mid_weight
    return late_weight


def build_feature_row(
    model_key: str,
    base_expert: torch.Tensor,
    expert: torch.Tensor,
    base_rank: int,
    significant_probabilities: int,
) -> dict:
    delta = expert - base_expert
    public_norm = torch.linalg.norm(base_expert.to(dtype=torch.float64)).item()
    delta_norm = torch.linalg.norm(delta.to(dtype=torch.float64)).item()
    relative_delta_norm = delta_norm / public_norm if public_norm > 0 else 0.0
    public_expert_cosine = flatten_cosine_similarity(base_expert, expert)
    return {
        "model_key": model_key,
        "layer": infer_layer_index(model_key),
        "module": infer_module_name(model_key),
        "base_rank": base_rank,
        "significant_probabilities": significant_probabilities,
        "public_expert_cosine": public_expert_cosine,
        "relative_delta_norm": relative_delta_norm,
    }


def resolve_weighted_ranks(
    feature_rows: list[dict],
    num_layers: int,
    early_weight: float,
    mid_weight: float,
    late_weight: float,
    last_weight: float,
) -> dict[str, int]:
    cosine_distance_values = [1.0 - row["public_expert_cosine"] for row in feature_rows]
    relative_delta_values = [row["relative_delta_norm"] for row in feature_rows]

    cosine_min, cosine_max = min(cosine_distance_values), max(cosine_distance_values)
    delta_min, delta_max = min(relative_delta_values), max(relative_delta_values)

    resolved_ranks = {}
    for row in feature_rows:
        module_weight = MODULE_WEIGHTS.get(row["module"], 1.0)
        layer_weight = compute_layer_weight(
            row["layer"],
            num_layers,
            early_weight,
            mid_weight,
            late_weight,
            last_weight,
        )
        cosine_score = normalize_feature(1.0 - row["public_expert_cosine"], cosine_min, cosine_max)
        delta_score = normalize_feature(row["relative_delta_norm"], delta_min, delta_max)
        similarity_weight = 0.75 + 0.25 * cosine_score + 0.25 * delta_score
        adjusted_rank = max(1, math.ceil(row["base_rank"] * module_weight * layer_weight * similarity_weight))
        resolved_ranks[row["model_key"]] = adjusted_rank

        log.info(
            "Adjusted rank for %s: rank=%s (base=%s, probs=%s, cos=%.4f, rel_delta=%.4f, module_w=%.3f, layer_w=%.3f, sim_w=%.3f)",
            row["model_key"],
            adjusted_rank,
            row["base_rank"],
            row["significant_probabilities"],
            row["public_expert_cosine"],
            row["relative_delta_norm"],
            module_weight,
            layer_weight,
            similarity_weight,
        )

    return resolved_ranks


def main(
    model_path: str = typer.Argument(..., help="Path to the FlexOLMo model in HF format"),
    rank: list[int] = typer.Option(
        [0,1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        help="Rank for the low-rank adapters to be applied to each linear layer",
    ),
    processes: int = typer.Option(1, help="Number of processes for SVD computation"),
    lora_modules: list[str] = typer.Option(
        ["gate_proj", "down_proj", "up_proj"],
        help="List of modules to apply LoRA to",
    ),
    effective_rank_probability_threshold: float = typer.Option(
        1e-3,
        help=(
            "When rank is 0, ignore normalized singular-value probabilities at or below "
            "this value before computing entropy effective rank."
        ),
    ),
    early_layer_weight: float = typer.Option(0.90, help="Importance multiplier for early layers"),
    mid_layer_weight: float = typer.Option(1.00, help="Importance multiplier for middle layers"),
    late_layer_weight: float = typer.Option(1.15, help="Importance multiplier for late layers"),
    last_layer_weight: float = typer.Option(1.25, help="Importance multiplier for the final layer"),
):
    prepare_cli_environment()
    if effective_rank_probability_threshold < 0:
        raise ValueError("effective_rank_probability_threshold must be non-negative")
    torch.set_num_threads(processes)

    log.info(f"Loading config from {model_path}")
    model_config = json.load(open(f"{model_path}/config.json", "r"))
    model_config["model_type"] = "flexmore"
    model_config["architectures"] = ["FlexMoREForCausalLM"]
    model_config = FlexMoREConfig.from_dict(model_config)

    log.info(f"Loading model from {model_path}")
    expert = FlexOlmoForCausalLM.from_pretrained(model_path)
    expert_state_dict = expert.state_dict()
    num_layers = int(getattr(expert.config, "n_layers", getattr(expert.config, "num_hidden_layers", 32)))

    key2usvh = {} if not os.path.exists("svd_cache.pkl") else pickle.load(open("svd_cache.pkl", "rb"))
    for r in rank:
        resolved_ranks = {}
        if r == 0:
            feature_rows = []
            for key in list(expert_state_dict.keys()):
                weights = expert_state_dict[key]
                if ".experts." not in key or not any(lora_module in key for lora_module in lora_modules):
                    continue
                if "gate_up_proj" in key:
                    base_experts = list(weights[0].chunk(2, dim=0))
                    experts = list(weights[1].chunk(2, dim=0))
                    model_keys = [
                        key.replace(".experts.gate_up_proj", ".experts.1.gate_proj"),
                        key.replace(".experts.gate_up_proj", ".experts.1.up_proj"),
                    ]
                elif "down_proj" in key:
                    base_experts = [weights[0]]
                    experts = [weights[1]]
                    model_keys = [key.replace(".experts.down_proj", ".experts.1.down_proj")]
                else:
                    raise AssertionError(f"Unexpected expert key {key}")

                for base_expert, expert_tensor, model_key in zip(base_experts, experts, model_keys):
                    delta_expert = expert_tensor - base_expert
                    if model_key not in key2usvh:
                        key2usvh[model_key] = torch.linalg.svd(delta_expert, full_matrices=False)
                    _, s, _ = key2usvh[model_key]
                    base_rank, _, significant_probabilities = compute_effective_rank(
                        s,
                        probability_threshold=effective_rank_probability_threshold,
                    )
                    feature_rows.append(
                        build_feature_row(
                            model_key=model_key,
                            base_expert=base_expert,
                            expert=expert_tensor,
                            base_rank=base_rank,
                            significant_probabilities=significant_probabilities,
                        )
                    )

            resolved_ranks = resolve_weighted_ranks(
                feature_rows,
                num_layers,
                early_layer_weight,
                mid_layer_weight,
                late_layer_weight,
                last_layer_weight,
            )
            assert resolved_ranks, "No LoRA-enabled expert layers found for effective-rank computation"
            model_rank = max(resolved_ranks.values())
            log.info(f"Using model rank {model_rank} to accommodate per-layer effective ranks")
        else:
            model_rank = r

        model_config.expert_ranks = [0, model_rank]
        model = FlexMoREForCausalLM(config=model_config)
        model_state_dict = model.state_dict()
        processed_keys = []
        for key in list(expert_state_dict.keys()):
            weights = expert_state_dict[key]
            if ".experts." in key and any(lora_module in key for lora_module in lora_modules):
                if "gate_up_proj" in key:
                    base_experts = list(weights[0].chunk(2, dim=0))
                    experts = list(weights[1].chunk(2, dim=0))
                    model_keys = [
                        key.replace(".experts.gate_up_proj", ".experts.1.gate_proj"),
                        key.replace(".experts.gate_up_proj", ".experts.1.up_proj"),
                    ]
                elif "down_proj" in key:
                    base_experts = [weights[0]]
                    experts = [weights[1]]
                    model_keys = [key.replace(".experts.down_proj", ".experts.1.down_proj")]
                else:
                    raise AssertionError(f"Unexpected expert key {key}")
                for base_expert, expert_tensor, model_key in zip(base_experts, experts, model_keys):
                    base_key = model_key.replace(".experts.1.", ".experts.0.").replace("_proj", "_proj.weight")
                    model_state_dict[base_key] = base_expert
                    processed_keys.append(base_key)
                    delta_expert = expert_tensor - base_expert
                    if model_key not in key2usvh:
                        key2usvh[model_key] = torch.linalg.svd(delta_expert, full_matrices=False)
                    u, s, vh = key2usvh[model_key]
                    layer_rank = resolved_ranks.get(model_key, r)
                    lora_u = u[:, :layer_rank]
                    lora_s = s[:layer_rank]
                    lora_vh = vh[:layer_rank, :]
                    sqrt_s = lora_s.sqrt()
                    lora_a = sqrt_s[:, None] * lora_vh
                    lora_b = lora_u * sqrt_s
                    a_key = model_key.replace("_proj", "_proj_a.weight")
                    b_key = model_key.replace("_proj", "_proj_b.weight")
                    padded_lora_a = torch.zeros_like(model_state_dict[a_key])
                    padded_lora_b = torch.zeros_like(model_state_dict[b_key])
                    padded_lora_a[:layer_rank, :] = lora_a
                    padded_lora_b[:, :layer_rank] = lora_b
                    model_state_dict[a_key] = padded_lora_a
                    model_state_dict[b_key] = padded_lora_b
                    processed_keys.extend([a_key, b_key])
            else:
                model_state_dict[key] = weights
                processed_keys.append(key)

        pickle.dump(key2usvh, open("svd_cache.pkl", "wb"))
        assert set(processed_keys) == set(model_state_dict.keys())
        assert len(processed_keys) == len(model_state_dict)
        model.config.expert_ranks = [0, model_rank]
        save_path = f"{model_path}-r{r}" if r != 0 else f"{model_path}-erank"
        model.save_pretrained(save_path, state_dict=model_state_dict)


if __name__ == "__main__":
    typer.run(main)
