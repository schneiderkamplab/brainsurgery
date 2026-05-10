#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def _deepseek_v4_classes():
    from transformers.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM

    return DeepseekV4Config, DeepseekV4ForCausalLM


def _test_config() -> object:
    DeepseekV4Config, _ = _deepseek_v4_classes()
    # Feature-complete miniature V4: it includes sliding attention, HCA, CSA,
    # hash-MoE bootstrap layers, routed MoE layers, mHC residual streams, grouped
    # output projection metadata, and compressed-RoPE settings. The vocab stays
    # GPT-2-tokenizer-safe while being far smaller than the real checkpoints.
    return DeepseekV4Config(
        architectures=["DeepseekV4ForCausalLM"],
        model_type="deepseek_v4",
        vocab_size=65536,
        hidden_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        q_lora_rank=32,
        hc_mult=2,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        moe_intermediate_size=64,
        scoring_func="sqrtsoftplus",
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        max_position_embeddings=512,
        rope_theta=10000.0,
        layer_types=[
            "sliding_attention",
            "heavily_compressed_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
            "compressed_sparse_attention",
            "sliding_attention",
        ],
        compress_rates={
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 128,
        },
        compress_rope_theta=160000.0,
        hc_sinkhorn_iters=4,
        hc_eps=1.0e-6,
        mlp_layer_types=["hash_moe", "hash_moe", "hash_moe", "moe", "moe", "moe"],
        swiglu_limit=10.0,
        sliding_window=64,
        o_groups=2,
        o_lora_rank=32,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=8,
        num_nextn_predict_layers=0,
        qk_rope_head_dim=8,
        rms_norm_eps=1.0e-6,
        attention_dropout=0.0,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=1,
        tie_word_embeddings=False,
        use_cache=True,
    )


def _build_config(variant: str) -> object:
    if variant == "test":
        return _test_config()
    raise ValueError(f"unsupported variant: {variant}")


def _dtype_from_name(name: str) -> torch.dtype:
    normalized = name.strip().lower()
    if normalized in {"float32", "fp32"}:
        return torch.float32
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float16", "fp16"}:
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def create_deepseek_v4_test_checkpoint(
    *,
    output: Path,
    dtype: torch.dtype,
    seed: int,
    max_shard_size: str,
    tokenizer: str | None = "openai-community/gpt2",
) -> dict[str, object]:
    torch.manual_seed(int(seed))
    config = _build_config("test")
    _, model_cls = _deepseek_v4_classes()
    model = model_cls(config)
    model.eval()

    if dtype != torch.float32:
        model = model.to(dtype=dtype)

    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        str(output),
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    # DeepSeek V4's save hooks may serialize expert tensors in a transformed
    # raw layout. Axon/codegen2-torch intentionally consumes the checkpoint
    # state_dict keys directly, so store the canonical HF state_dict layout that
    # the loaded module exposes.
    for path in output.glob("*.safetensors"):
        path.unlink()
    for path in output.glob("*.safetensors.index.json"):
        path.unlink()
    canonical_state = {
        name: tensor.detach().contiguous().cpu() for name, tensor in model.state_dict().items()
    }
    save_file(canonical_state, output / "model.safetensors", metadata={"format": "pt"})
    if tokenizer:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(tokenizer).save_pretrained(str(output))

    summary: dict[str, object] = {
        "variant": "test",
        "parameter_count": int(sum(param.numel() for param in model.parameters())),
        "storage_dtype": str(dtype).removeprefix("torch."),
        "seed": int(seed),
        "tokenizer_for_tests": tokenizer,
        "num_hidden_layers": int(getattr(config, "num_hidden_layers", 0)),
        "layer_types": list(getattr(config, "layer_types", [])),
        "mlp_layer_types": list(getattr(config, "mlp_layer_types", [])),
        "compress_rates": dict(getattr(config, "compress_rates", {})),
    }
    (output / "README.md").write_text(
        f"# {output.name}\n\n"
        "Deterministic synthetic DeepSeek V4 checkpoint for local Axon/HF parity tests.\n"
        "Weights are randomly initialized; this is not a trained model.\n\n"
        f"```json\n{json.dumps(summary, indent=2, sort_keys=True)}\n```\n",
        encoding="utf-8",
    )
    (output / "random_checkpoint_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"output": str(output), **summary}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a deterministic DeepSeek V4 test checkpoint for Axon/HF parity tests."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output model directory.",
    )
    parser.add_argument(
        "--variant",
        choices=("test",),
        default="test",
        help="Checkpoint variant to create.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default="bfloat16",
        help="Storage dtype. Benchmarks can still load with --dtype float32.",
    )
    parser.add_argument("--max-shard-size", default="2GB")
    parser.add_argument(
        "--tokenizer",
        default="openai-community/gpt2",
        help="Tokenizer to save into the random checkpoint directory for local benchmarks.",
    )
    args = parser.parse_args()

    dtype = _dtype_from_name(args.dtype)
    default_output = Path("models/test/DeepSeek-V4-Test")
    output = args.output or default_output
    summary = create_deepseek_v4_test_checkpoint(
        output=output,
        dtype=dtype,
        seed=int(args.seed),
        max_shard_size=args.max_shard_size,
        tokenizer=args.tokenizer,
    )
    summary["storage_dtype"] = args.dtype
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
