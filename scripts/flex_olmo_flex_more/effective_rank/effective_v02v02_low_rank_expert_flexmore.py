import json
import logging
import math
import os
import pickle
import torch
from transformers import FlexMoREConfig, FlexMoREForCausalLM, FlexOlmoConfig, FlexOlmoForCausalLM
import typer

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


def adjust_rank_with_importance(
    model_key: str,
    base_rank: int,
    num_layers: int,
    early_weight: float,
    mid_weight: float,
    late_weight: float,
    last_weight: float,
) -> tuple[int, float, float]:
    layer_index = infer_layer_index(model_key)
    module_name = infer_module_name(model_key)
    module_weight = MODULE_WEIGHTS.get(module_name, 1.0)
    layer_weight = compute_layer_weight(
        layer_index,
        num_layers,
        early_weight,
        mid_weight,
        late_weight,
        last_weight,
    )
    adjusted_rank = max(1, math.ceil(base_rank * module_weight * layer_weight))
    return adjusted_rank, module_weight, layer_weight

def main(
    model_path: str = typer.Argument(..., help="Path to the FlexOLMo model in HF format"),
    rank: list[int] = typer.Option(
        [0,1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384],
        help="Rank for the low-rank adapters to be applied to each linear layer",
    ),
    processes: int = typer.Option(
        1,
        help="Number of processes for SVD computation",
    ),
    lora_modules: list[str] = typer.Option(
        [
            "gate_proj",
            "down_proj",
            "up_proj",
        ],
        help="List of modules to apply LoRA to",
    ),
    effective_rank_probability_threshold: float = typer.Option(
        0.0,
        help=(
            "When rank is 0, ignore normalized singular-value probabilities at or below "
            "this value before computing entropy effective rank, then renormalize the "
            "remaining probabilities."
        ),
    ),
    early_layer_weight: float = typer.Option(
        0.90,
        help="Weight multiplier for early layers when aggregating to one expert-wide rank.",
    ),
    mid_layer_weight: float = typer.Option(
        1.00,
        help="Weight multiplier for middle layers when aggregating to one expert-wide rank.",
    ),
    late_layer_weight: float = typer.Option(
        1.15,
        help="Weight multiplier for late layers when aggregating to one expert-wide rank.",
    ),
    last_layer_weight: float = typer.Option(
        1.25,
        help="Weight multiplier for the final layer when aggregating to one expert-wide rank.",
    ),
):
    prepare_cli_environment()
    if effective_rank_probability_threshold < 0:
        raise ValueError("effective_rank_probability_threshold must be non-negative")
    log.info(f"Setting number of threads for SVD computation to {processes}")
    torch.set_num_threads(processes)

    log.info(f"Loading config from {model_path}")
    model_config = json.load(open(f"{model_path}/config.json", "r"))
    model_config['model_type'] = 'flexmore'
    model_config['architectures'] = ['FlexMoREForCausalLM']
    model_config = FlexMoREConfig.from_dict(model_config)
    log.info(model_config)

    log.info(f"Loading model from {model_path}")
    expert = FlexOlmoForCausalLM.from_pretrained(model_path)
    log.info(expert.config)
    log.info(expert)
    expert_state_dict = expert.state_dict()
    num_layers = int(getattr(expert.config, "n_layers", getattr(expert.config, "num_hidden_layers", 32)))

    key2usvh = {} if not os.path.exists("svd_cache.pkl") else pickle.load(open("svd_cache.pkl", "rb"))
    for r in rank:
        resolved_ranks = {}
        if r == 0:
            log.info("Computing effective ranks for LoRA-enabled expert layers")
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
                    assert False, f"Unexpected expert key {key}"

                for base_expert, expert, model_key in zip(base_experts, experts, model_keys):
                    delta_expert = expert - base_expert
                    if model_key not in key2usvh:
                        log.info(f"Computing SVD for key {model_key} with shape {delta_expert.shape}")
                        key2usvh[model_key] = torch.linalg.svd(delta_expert, full_matrices=False)
                    _, s, _ = key2usvh[model_key]
                    (
                        base_rank,
                        num_significant_singular_values,
                        num_significant_probabilities,
                    ) = compute_effective_rank(
                        s,
                        probability_threshold=effective_rank_probability_threshold,
                    )
                    (
                        resolved_ranks[model_key],
                        module_weight,
                        layer_weight,
                    ) = adjust_rank_with_importance(
                        model_key=model_key,
                        base_rank=base_rank,
                        num_layers=num_layers,
                        early_weight=early_layer_weight,
                        mid_weight=mid_layer_weight,
                        late_weight=late_layer_weight,
                        last_weight=last_layer_weight,
                    )
                    log.info(
                        f"Effective rank for key {model_key}: {resolved_ranks[model_key]} "
                        f"(base rank {base_rank}, from {num_significant_singular_values} singular values and "
                        f"{num_significant_probabilities} significant probabilities "
                        f"out of {s.numel()}, probability threshold "
                        f"{effective_rank_probability_threshold}, module weight "
                        f"{module_weight}, layer weight {layer_weight})"
                    )
            assert resolved_ranks, "No LoRA-enabled expert layers found for effective-rank computation"
            model_rank = max(resolved_ranks.values())
            log.info(f"Using model rank {model_rank} to accommodate per-layer effective ranks")
        else:
            model_rank = r

        log.info("Initializing empty model")
        model_config.expert_ranks = [0, model_rank]
        model = FlexMoREForCausalLM(config=model_config)
        model_state_dict = model.state_dict()
        log.info(model)
        processed_keys = []
        for key in list(expert_state_dict.keys()):
            weights = expert_state_dict[key]
            if ".experts." in key and any(lora_module in key for lora_module in lora_modules):
                log.info(f"Processing key {key}")
                if "gate_up_proj" in key:
                    print(f"Shape of weights for key {key}: {weights.shape}")
                    base_experts = list(weights[0].chunk(2, dim=0))
                    print(f"Shapes of base experts for key {key}: {[be.shape for be in base_experts]}")
                    experts = list(weights[1].chunk(2, dim=0))
                    model_keys = [
                        key.replace(".experts.gate_up_proj", ".experts.1.gate_proj"),
                        key.replace(".experts.gate_up_proj", ".experts.1.up_proj"),
                    ]
                elif "down_proj" in key:
                    base_experts = [weights[0]]
                    print(f"Shape of base expert for key {key}: {base_experts[0].shape}")
                    experts = [weights[1]]
                    print(f"Shape of expert for key {key}: {experts[0].shape}")
                    model_keys = [key.replace(".experts.down_proj", ".experts.1.down_proj")]
                else:
                    assert False, f"Unexpected expert key {key}"
                for base_expert, expert, model_key in zip(base_experts, experts, model_keys):
                    base_key = model_key.replace(".experts.1.", ".experts.0.")
                    base_key = base_key.replace("_proj", "_proj.weight")
                    assert base_key in model_state_dict, f"Base key {base_key} not found in model state dict: {list(model_state_dict.keys())}"
                    assert base_expert.shape == model_state_dict[base_key].shape, f"Shape mismatch for base key {base_key}: expert shape {base_expert.shape}, model shape {model_state_dict[base_key].shape}"
                    model_state_dict[base_key] = base_expert
                    processed_keys.append(base_key)
                    delta_expert = expert - base_expert
                    # compute the low-rank adaptation
                    if model_key not in key2usvh:
                        log.info(f"Computing SVD for key {model_key} with shape {delta_expert.shape}")
                        key2usvh[model_key] = torch.linalg.svd(delta_expert, full_matrices=False)
                    u, s, vh = key2usvh[model_key]
                    layer_rank = resolved_ranks.get(model_key, r)
                    lora_u = u[:, :layer_rank]
                    lora_s = s[:layer_rank]
                    lora_vh = vh[:layer_rank, :]
                    print(f"Shapes for key {model_key}: u {u.shape}, s {s.shape}, vh {vh.shape}")
                    print(f"Shapes for LoRA key {model_key}: lora_u {lora_u.shape}, lora_s {lora_s.shape}, lora_vh {lora_vh.shape}")
                    sqrt_s = lora_s.sqrt()
                    lora_a = sqrt_s[:, None] * lora_vh
                    lora_b = lora_u * sqrt_s
                    dummy = (lora_b @ lora_a)
                    print(f"Reconstructed delta shape for key {model_key}: {dummy.shape}")
                    log.info(f"Storing LoRA adapters for key {model_key} with shapes {lora_a.shape}, {lora_b.shape}")
                    a_key = model_key.replace("_proj", f"_proj_a.weight")
                    b_key = model_key.replace("_proj", f"_proj_b.weight")
                    assert a_key in model_state_dict, f"Key {a_key} not found in model state dict: {list(model_state_dict.keys())}"
                    assert b_key in model_state_dict, f"Key {b_key} not found in model state dict: {list(model_state_dict.keys())}"
                    padded_lora_a = torch.zeros_like(model_state_dict[a_key])
                    padded_lora_b = torch.zeros_like(model_state_dict[b_key])
                    padded_lora_a[:layer_rank, :] = lora_a
                    padded_lora_b[:, :layer_rank] = lora_b
                    model_state_dict[a_key] = padded_lora_a
                    model_state_dict[b_key] = padded_lora_b
                    processed_keys.extend([a_key, b_key])
            else:
                assert key in model_state_dict, f"Key {key} not found in model state dict: {list(model_state_dict.keys())}"
                assert weights.shape == model_state_dict[key].shape, f"Shape mismatch for key {key}: expert shape {weights.shape}, model shape {model_state_dict[key].shape}"
                model_state_dict[key] = weights
                processed_keys.append(key)
        pickle.dump(key2usvh, open(f"svd_cache.pkl", "wb"))
        assert set(processed_keys) == set(model_state_dict.keys()), f"Not all keys were processed: processed {processed_keys}, model keys {list(model_state_dict.keys())}, missing {set(model_state_dict.keys()) - set(processed_keys)}"
        assert len(processed_keys) == len(model_state_dict), "Some keys were processed multiple times"
        # adapt the config
        model.config.expert_ranks = [0, model_rank]
        log.info(f"Model config after adaptation: {model.config}")
        # save the final_state_dict for the MoE in a format that the olmo_core trainer likes
        save_path = f"{model_path}-r{r}" if r != 0 else f"{model_path}-erank"
        log.info(f"Saving model to {save_path}")
        print(f"Final model state dict keys: {list(model_state_dict.keys())}")
        model.save_pretrained(save_path, state_dict=model_state_dict)
    log.info("Done")

if __name__ == "__main__":
    typer.run(main)
