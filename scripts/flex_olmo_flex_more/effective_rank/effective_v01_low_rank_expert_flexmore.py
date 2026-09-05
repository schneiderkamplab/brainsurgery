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


def compute_effective_rank(
    singular_values: torch.Tensor,
    relative_threshold: float = 1e-3,
    threshold: float = 0.0,
) -> tuple[int, int, float]:
    singular_values = singular_values.to(dtype=torch.float64)
    max_singular_value = singular_values.max().item()

    if relative_threshold > 0:
        threshold = max(threshold, max_singular_value * relative_threshold)
    if threshold > 0:
        singular_values = singular_values[singular_values >= threshold]

    norm = singular_values.sum()
    if norm <= 0:
        return 1, 0, threshold

    probabilities = singular_values / norm
    nonzero_probabilities = probabilities[probabilities > 0]
    entropy = -(nonzero_probabilities * nonzero_probabilities.log()).sum()
    effective_rank = torch.exp(entropy).item()
    rank = max(1, min(singular_values.numel(), math.ceil(effective_rank)))
    return rank, singular_values.numel(), threshold

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
    effective_rank_relative_threshold: float = typer.Option(
        1e-3,
        help=(
            "When rank is 0, ignore singular values smaller than this fraction "
            "of the largest singular value before computing entropy effective rank. "
            "Use 0 to recover the unthresholded paper definition."
        ),
    ),
    effective_rank_absolute_threshold: float = typer.Option(
        1e-8,
        help=(
            "When rank is 0, ignore singular values smaller than this absolute value "
            "before computing entropy effective rank. This is combined with the relative "
            "threshold by using the stricter cutoff."
        ),
    ),
):
    prepare_cli_environment()
    if effective_rank_relative_threshold < 0:
        raise ValueError("effective_rank_relative_threshold must be non-negative")
    if effective_rank_absolute_threshold < 0:
        raise ValueError("effective_rank_absolute_threshold must be non-negative")
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
                    resolved_ranks[model_key], num_significant, effective_rank_threshold = compute_effective_rank(
                        s,
                        relative_threshold=effective_rank_relative_threshold,
                        threshold=effective_rank_absolute_threshold,
                    )
                    log.info(
                        f"Effective rank for key {model_key}: {resolved_ranks[model_key]} "
                        f"(from {num_significant} significant singular values "
                        f"out of {s.numel()}, threshold {effective_rank_threshold}, "
                        f"relative threshold {effective_rank_relative_threshold}, "
                        f"absolute threshold {effective_rank_absolute_threshold})"
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
