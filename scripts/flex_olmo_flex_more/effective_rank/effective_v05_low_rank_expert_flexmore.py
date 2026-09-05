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


MODULE_BONUS = {
    "down_proj": 0.00,
    "gate_proj": 0.08,
    "up_proj": 0.04,
}


def compute_energy_rank(
    singular_values: torch.Tensor,
    energy_tau: float = 0.95,
) -> tuple[int, int]:
    singular_values = torch.sort(singular_values.to(dtype=torch.float64), descending=True).values
    if singular_values.numel() == 0:
        return 1, 0

    energy = singular_values.square()
    total_energy = energy.sum()
    if total_energy <= 0:
        return 1, singular_values.numel()

    explained_energy = energy / total_energy
    cumulative_energy = explained_energy.cumsum(dim=0)
    rank = int(torch.searchsorted(cumulative_energy, torch.tensor(energy_tau, dtype=torch.float64)).item()) + 1
    rank = max(1, min(rank, singular_values.numel()))
    return rank, singular_values.numel()


def compute_topk_energy_share(singular_values: torch.Tensor, k: int) -> float:
    singular_values = torch.sort(singular_values.to(dtype=torch.float64), descending=True).values
    if singular_values.numel() == 0:
        return 0.0

    energy = singular_values.square()
    total_energy = energy.sum()
    if total_energy <= 0:
        return 0.0

    topk = energy[: min(k, energy.numel())].sum()
    return float((topk / total_energy).item())


def compute_probability_entropy(singular_values: torch.Tensor) -> float:
    singular_values = singular_values.to(dtype=torch.float64)
    total = singular_values.sum()
    if total <= 0 or singular_values.numel() == 0:
        return 0.0

    probabilities = singular_values / total
    probabilities = probabilities[probabilities > 0]
    if probabilities.numel() == 0:
        return 0.0

    entropy = -(probabilities * probabilities.log()).sum().item()
    max_entropy = math.log(float(probabilities.numel()))
    if max_entropy <= 0:
        return 0.0
    return float(entropy / max_entropy)


def compute_gini_coefficient(singular_values: torch.Tensor) -> float:
    values = torch.sort(singular_values.to(dtype=torch.float64).flatten(), descending=False).values
    if values.numel() == 0:
        return 0.0

    total = values.sum().item()
    if total <= 0:
        return 0.0

    n = values.numel()
    indices = torch.arange(1, n + 1, dtype=torch.float64)
    gini = ((2 * indices - n - 1) * values).sum().item() / (n * total)
    return max(0.0, min(1.0, float(gini)))


def compute_subspace_novelty(delta_u: torch.Tensor, public_u: torch.Tensor, top_k: int) -> float:
    if delta_u.numel() == 0 or public_u.numel() == 0:
        return 0.0

    k = min(top_k, delta_u.shape[1], public_u.shape[1])
    if k <= 0:
        return 0.0

    delta_basis = delta_u[:, :k].to(dtype=torch.float64)
    public_basis = public_u[:, :k].to(dtype=torch.float64)
    overlap = delta_basis.T @ public_basis
    normalized_overlap = torch.linalg.norm(overlap, ord="fro").item() / math.sqrt(k)
    similarity = max(0.0, min(1.0, float(normalized_overlap)))
    return 1.0 - similarity


def flatten_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.to(dtype=torch.float64).reshape(-1)
    b_flat = b.to(dtype=torch.float64).reshape(-1)
    denom = torch.linalg.norm(a_flat) * torch.linalg.norm(b_flat)
    if denom <= 0:
        return 0.0
    return float((torch.dot(a_flat, b_flat) / denom).item())


def infer_layer_index(model_key: str) -> int:
    parts = model_key.split(".")
    return int(parts[2])


def infer_module_name(model_key: str) -> str:
    return model_key.split(".")[-1]


def normalize_feature(value: float, min_value: float, max_value: float) -> float:
    if math.isclose(min_value, max_value):
        return 0.5
    normalized = (value - min_value) / (max_value - min_value)
    return max(0.0, min(1.0, float(normalized)))


def select_candidate_pool(candidate_ranks: list[int]) -> list[int]:
    compact_pool = [rank for rank in candidate_ranks if 8 <= rank <= 1024]
    if compact_pool:
        return compact_pool
    positive_candidates = [rank for rank in candidate_ranks if rank > 0]
    if positive_candidates:
        return positive_candidates
    return [1]


def assign_rank_from_score(score: float, candidate_pool: list[int]) -> int:
    if len(candidate_pool) == 1:
        return candidate_pool[0]
    index = int(round(score * (len(candidate_pool) - 1)))
    index = max(0, min(index, len(candidate_pool) - 1))
    return candidate_pool[index]


def build_feature_row(
    model_key: str,
    base_expert: torch.Tensor,
    expert: torch.Tensor,
    delta_u: torch.Tensor,
    delta_s: torch.Tensor,
    public_u: torch.Tensor,
    overlap_top_k: int,
) -> dict:
    delta_expert = expert - base_expert
    public_norm = torch.linalg.norm(base_expert.to(dtype=torch.float64)).item()
    delta_norm = torch.linalg.norm(delta_expert.to(dtype=torch.float64)).item()
    public_expert_cosine = flatten_cosine_similarity(base_expert, expert)
    relative_delta_norm = delta_norm / public_norm if public_norm > 0 else 0.0
    top50_energy_share = compute_topk_energy_share(delta_s, 50)
    entropy = compute_probability_entropy(delta_s)
    gini = compute_gini_coefficient(delta_s)
    subspace_novelty = compute_subspace_novelty(delta_u, public_u, overlap_top_k)

    return {
        "model_key": model_key,
        "layer": infer_layer_index(model_key),
        "module": infer_module_name(model_key),
        "public_expert_cosine": public_expert_cosine,
        "relative_delta_norm": relative_delta_norm,
        "top50_energy_share": top50_energy_share,
        "entropy": entropy,
        "gini": gini,
        "subspace_novelty": subspace_novelty,
        "num_singular_values": int(delta_s.numel()),
    }


def resolve_hybrid_ranks(feature_rows: list[dict], candidate_ranks: list[int]) -> dict[str, int]:
    cosine_distance_values = [1.0 - row["public_expert_cosine"] for row in feature_rows]
    relative_delta_values = [row["relative_delta_norm"] for row in feature_rows]
    inverse_top50_values = [1.0 - row["top50_energy_share"] for row in feature_rows]
    entropy_values = [row["entropy"] for row in feature_rows]
    inverse_gini_values = [1.0 - row["gini"] for row in feature_rows]
    novelty_values = [row["subspace_novelty"] for row in feature_rows]
    max_layer = max(row["layer"] for row in feature_rows) if feature_rows else 1
    candidate_pool = select_candidate_pool(sorted(set(candidate_ranks)))

    cosine_min, cosine_max = min(cosine_distance_values), max(cosine_distance_values)
    delta_min, delta_max = min(relative_delta_values), max(relative_delta_values)
    top50_min, top50_max = min(inverse_top50_values), max(inverse_top50_values)
    entropy_min, entropy_max = min(entropy_values), max(entropy_values)
    gini_min, gini_max = min(inverse_gini_values), max(inverse_gini_values)
    novelty_min, novelty_max = min(novelty_values), max(novelty_values)

    resolved_ranks = {}
    for row in feature_rows:
        cosine_score = normalize_feature(1.0 - row["public_expert_cosine"], cosine_min, cosine_max)
        delta_score = normalize_feature(row["relative_delta_norm"], delta_min, delta_max)
        top50_score = normalize_feature(1.0 - row["top50_energy_share"], top50_min, top50_max)
        entropy_score = normalize_feature(row["entropy"], entropy_min, entropy_max)
        gini_score = normalize_feature(1.0 - row["gini"], gini_min, gini_max)
        novelty_score = normalize_feature(row["subspace_novelty"], novelty_min, novelty_max)
        layer_bonus = 0.05 * (row["layer"] / max_layer if max_layer > 0 else 0.0)
        module_bonus = MODULE_BONUS.get(row["module"], 0.0)

        score = (
            0.20 * cosine_score
            + 0.20 * delta_score
            + 0.15 * top50_score
            + 0.15 * entropy_score
            + 0.10 * gini_score
            + 0.20 * novelty_score
            + layer_bonus
            + module_bonus
        )
        score = max(0.0, min(1.0, score))
        resolved_rank = assign_rank_from_score(score, candidate_pool)
        resolved_ranks[row["model_key"]] = resolved_rank

        log.info(
            "Hybrid rank for %s: rank=%s score=%.4f "
            "(cos=%.4f, rel_delta=%.4f, top50=%.4f, entropy=%.4f, gini=%.4f, novelty=%.4f, module=%s, layer=%s)",
            row["model_key"],
            resolved_rank,
            score,
            row["public_expert_cosine"],
            row["relative_delta_norm"],
            row["top50_energy_share"],
            row["entropy"],
            row["gini"],
            row["subspace_novelty"],
            row["module"],
            row["layer"],
        )

    return resolved_ranks

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
    energy_tau: float = typer.Option(
        0.95,
        help=(
            "Unused compatibility option inherited from v04."
        ),
    ),
    overlap_top_k: int = typer.Option(
        16,
        help="Number of leading left-singular vectors used for public-vs-delta subspace overlap.",
    ),
):
    prepare_cli_environment()
    if not (0.0 < energy_tau <= 1.0):
        raise ValueError("energy_tau must be in the interval (0, 1]")
    if overlap_top_k <= 0:
        raise ValueError("overlap_top_k must be positive")
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
    public_key2usvh = {} if not os.path.exists("public_svd_cache.pkl") else pickle.load(open("public_svd_cache.pkl", "rb"))
    for r in rank:
        resolved_ranks = {}
        if r == 0:
            log.info("Computing redundancy-aware hybrid effective ranks for LoRA-enabled expert layers")
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
                    assert False, f"Unexpected expert key {key}"

                for base_expert, expert, model_key in zip(base_experts, experts, model_keys):
                    delta_expert = expert - base_expert
                    if model_key not in key2usvh:
                        log.info(f"Computing SVD for key {model_key} with shape {delta_expert.shape}")
                        key2usvh[model_key] = torch.linalg.svd(delta_expert, full_matrices=False)
                    base_key = f"{model_key}::public"
                    if base_key not in public_key2usvh:
                        log.info(f"Computing public SVD for key {model_key} with shape {base_expert.shape}")
                        public_key2usvh[base_key] = torch.linalg.svd(base_expert, full_matrices=False)
                    delta_u, delta_s, _ = key2usvh[model_key]
                    public_u, _, _ = public_key2usvh[base_key]
                    feature_rows.append(
                        build_feature_row(
                            model_key=model_key,
                            base_expert=base_expert,
                            expert=expert,
                            delta_u=delta_u,
                            delta_s=delta_s,
                            public_u=public_u,
                            overlap_top_k=overlap_top_k,
                        )
                    )
            resolved_ranks = resolve_hybrid_ranks(feature_rows, rank)
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
        pickle.dump(public_key2usvh, open(f"public_svd_cache.pkl", "wb"))
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
