#!/usr/bin/env python3
"""Throughput benchmark: native vLLM vs axon-generated vLLM model.

Phase 1 (run in `bs` conda env):
    Generate vLLM model code from an Axon file and write to a .py file.

Phase 2 (run in `g4vllm` conda env):
    Load the generated model, register with vLLM ModelRegistry,
    create dummy safetensors weights, run both native and generated
    models through vLLM's LLM API, and report throughput.

Usage:
    # Phase 1: generate code (run in bs env)
    python scripts/benchmark_vllm_throughput.py generate \
        --axon brainsurgery/synapse/models/gemma3/gemma-3-270m.axon \
        --output /tmp/opencode/generated_vllm_model.py

    # Phase 2: run benchmark (run in g4vllm env)
    python scripts/benchmark_vllm_throughput.py benchmark \
        --generated-model /tmp/opencode/generated_vllm_model.py \
        --model-type gemma3_text \
        --batch-sizes 1 4 16 \
        --seq-len 128
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Phase 1: Generate vLLM model code from Axon file
# ---------------------------------------------------------------------------

def run_generate(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from brainsurgery.synapse.axon import (
        elaborate_closed_axon_file,
        flatten_closed_axon_file,
        lower_axon_program_to_graph_ir,
        normalize_closed_axon_file,
        optimize_graph_program,
        resolve_axon_program_from_path,
        typecheck2_flat_axon_file,
        GraphOptimizeConfig,
    )
    from brainsurgery.synapse.axon.codegen2_vllm import emit_model_code_from_graph_ir

    axon_file = Path(args.axon)
    resolved = resolve_axon_program_from_path(axon_file).ast
    normalized = normalize_closed_axon_file(resolved)
    elaborated = elaborate_closed_axon_file(normalized)
    flat = flatten_closed_axon_file(elaborated)
    typed = typecheck2_flat_axon_file(flat)
    graph_program = lower_axon_program_to_graph_ir(typed)

    # Extract embedding scale from pre-optimization graph (lost by optimizer)
    from brainsurgery.synapse.axon.graph_ir.core import GraphLiteral, GraphValueRef

    def _resolve_literal(gp, ref):
        """Follow GraphValueRef chain to find a literal value."""
        if isinstance(ref, GraphLiteral):
            return float(ref.value)
        if isinstance(ref, GraphValueRef):
            for m in gp.modules:
                for n in m.nodes:
                    for out in n.outputs:
                        if hasattr(out, 'name') and out.name == ref.name:
                            if n.op.name in ("_sqrt", "Math.sqrt"):
                                inner = _resolve_literal(gp, n.inputs[0])
                                return inner ** 0.5 if inner is not None else None
                            if n.op.name in ("_div", "Math.div"):
                                a = _resolve_literal(gp, n.inputs[0])
                                b = _resolve_literal(gp, n.inputs[1])
                                return a / b if a is not None and b is not None else None
                            for inp in n.inputs:
                                val = _resolve_literal(gp, inp)
                                if val is not None:
                                    return val
        return None

    _emb_scale = None
    _has_emb_scale = False
    for mod in graph_program.modules:
        for node in mod.nodes:
            if node.op.name == "NN.embedding" and len(node.inputs) >= 4:
                scale_ref = node.inputs[3]
                _has_emb_scale = True
                _emb_scale = _resolve_literal(graph_program, scale_ref)
                if _emb_scale is not None:
                    break
        if _emb_scale is not None:
            break

    if _emb_scale is None and _has_emb_scale and hasattr(args, 'hidden_size'):
        _emb_scale = float(args.hidden_size) ** 0.5

    # Extract attention scale from pre-optimization graph (lost by optimizer)
    _attn_scale = None
    for mod in graph_program.modules:
        for node in mod.nodes:
            if node.op.name == "Attention.attention_scaled" and len(node.inputs) >= 5:
                scale_input = node.inputs[4]
                if isinstance(scale_input, GraphLiteral):
                    _attn_scale = float(scale_input.value)

    graph_program = optimize_graph_program(
        graph_program, config=GraphOptimizeConfig(backend_intrinsics="codegen2-vllm")
    )

    model_config: dict[str, object] = {}
    for field in ("model_type", "hidden_size", "vocab_size", "num_hidden_layers",
                  "num_attention_heads", "num_key_value_heads", "head_dim",
                  "intermediate_size", "rms_norm_eps", "query_pre_attn_scalar"):
        if hasattr(args, field) and getattr(args, field) is not None:
            model_config[field] = getattr(args, field)
    if _emb_scale is not None:
        model_config["embedding_scale"] = _emb_scale
    if _attn_scale is not None:
        model_config["attention_scale"] = _attn_scale

    if args.extra_config:
        extra = json.loads(args.extra_config)
        model_config.update(extra)

    code = emit_model_code_from_graph_ir(
        graph_program,
        class_name=args.class_name,
        model_config=model_config,
    )
    compile(code, "<generated>", "exec")
    Path(args.output).write_text(code)
    print(f"Generated model code written to {args.output} ({len(code)} chars)")

    config_path = str(args.output) + ".config.json"
    Path(config_path).write_text(json.dumps(model_config, indent=2))
    print(f"Model config written to {config_path}")


# ---------------------------------------------------------------------------
# Phase 2: Benchmark throughput
# ---------------------------------------------------------------------------

def _create_dummy_weights(model: "torch.nn.Module", out_dir: Path) -> Path:
    """Save model state_dict as safetensors with random weights."""
    from safetensors.torch import save_file

    sd = {}
    for name, param in model.named_parameters():
        sd[name] = param.data.clone().cpu()
    for name, buf in model.named_buffers():
        if buf.numel() > 0:
            sd[name] = buf.clone().cpu()

    weight_file = out_dir / "model.safetensors"
    save_file(sd, str(weight_file))
    return weight_file


def _make_config_json(model_type: str, config: dict) -> dict:
    archs = config.get("architectures", "GeneratedVLLMModel")
    if isinstance(archs, str):
        archs = [archs]
    # Start with all fields from the input config, then override/add required ones.
    base = dict(config)
    base["model_type"] = model_type
    base["architectures"] = archs
    # Map model-specific field aliases to standard vLLM names.
    base.setdefault("hidden_size", base.get("n_embd", base.get("d_model", 640)))
    base.setdefault("vocab_size", 262144)
    base.setdefault("num_hidden_layers", base.get("n_layer", base.get("num_layers", 18)))
    base.setdefault("num_attention_heads", base.get("n_head", base.get("num_heads", 4)))
    base.setdefault("num_key_value_heads", base["num_attention_heads"])
    base.setdefault("head_dim", base["hidden_size"] // base["num_attention_heads"])
    base.setdefault("intermediate_size", base.get("n_inner", base.get("d_ff", base["hidden_size"] * 4)))
    base.setdefault("rms_norm_eps", base.get("layer_norm_epsilon", base.get("layer_norm_eps", 1e-6)))
    base.setdefault("query_pre_attn_scalar", int(base.get("head_dim", base["hidden_size"])))
    base.setdefault("torch_dtype", "bfloat16")
    base.setdefault("max_position_embeddings", base.get("n_positions", 4096))
    base.setdefault("bos_token_id", 2)
    base.setdefault("eos_token_id", 1)
    base.setdefault("pad_token_id", 0)
    return base


def run_benchmark(args: argparse.Namespace) -> None:
    import torch

    config = json.loads(Path(args.config).read_text())
    model_type = args.model_type

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seq_len = args.seq_len
    num_warmup = args.warmup
    num_iters = args.iterations

    results: list[dict] = []

    # --- Native vLLM model ---
    print("\n=== Native vLLM model ===")
    native_dir = Path(tempfile.mkdtemp(prefix="native_vllm_"))
    native_arch = args.native_arch
    native_config = _make_config_json(model_type, {**config, "architectures": [native_arch]})
    (native_dir / "config.json").write_text(json.dumps(native_config, indent=2))

    # Create a tokenizer-less config (we'll skip tokenizer)
    # Write a minimal tokenizer_config.json
    (native_dir / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": 4096,
    }))

    try:
        from vllm import LLM, SamplingParams

        print("Loading native vLLM model (random weights)...")
        native_llm = LLM(
            model=str(native_dir),
            enforce_eager=args.eager,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_mem,
            max_model_len=seq_len + 16,
            skip_tokenizer_init=True,
            load_format="dummy",
            tensor_parallel_size=args.tensor_parallel_size,
        )
        print("Native model loaded!")

        sp = SamplingParams(max_tokens=args.max_tokens, temperature=0, detokenize=False)
        prompt_tokens = list(range(1, seq_len + 1))

        for bs in batch_sizes:
            prompts = [{"prompt_token_ids": prompt_tokens} for _ in range(bs)]
            for _ in range(num_warmup):
                native_llm.generate(prompts, sp, use_tqdm=False)

            t0 = time.perf_counter()
            for _ in range(num_iters):
                native_llm.generate(prompts, sp, use_tqdm=False)
            elapsed = time.perf_counter() - t0
            total_tokens = bs * (seq_len + args.max_tokens) * num_iters
            throughput = total_tokens / elapsed
            print(f"  bs={bs}, seq={seq_len}: {throughput:.1f} tok/s ({elapsed:.3f}s for {num_iters} iters)")
            results.append({"model": "native", "batch_size": bs, "seq_len": seq_len,
                            "throughput_tok_s": throughput, "elapsed_s": elapsed})

        del native_llm
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Native model benchmark failed: {e}")

    # --- Generated vLLM model ---
    print("\n=== Generated vLLM model ===")
    generated_code = Path(args.generated_model).read_text()
    generated_ns: dict = {}
    exec(compile(generated_code, "<generated>", "exec", dont_inherit=True), generated_ns)
    GeneratedModelClass = generated_ns[args.class_name]

    from vllm.model_executor.models.registry import ModelRegistry
    arch_name = "GeneratedVLLMModel"
    ModelRegistry.register_model(arch_name, GeneratedModelClass)
    print(f"Registered {arch_name} with ModelRegistry")

    gen_dir = Path(tempfile.mkdtemp(prefix="generated_vllm_"))
    gen_config = _make_config_json(model_type, {**config, "architectures": [arch_name]})
    (gen_dir / "config.json").write_text(json.dumps(gen_config, indent=2))
    (gen_dir / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": 4096,
    }))

    try:
        # Detect heterogeneous head dims (e.g. Gemma4 E2B) and force
        # TRITON_ATTN backend, matching native vLLM's Gemma4Config logic.
        _hd = config.get("head_dim")
        _ghd = config.get("global_head_dim")
        _attn_backend = None
        if _hd is not None and _ghd is not None and _hd != _ghd and max(_hd, _ghd) > 256:
            _attn_backend = "TRITON_ATTN"
            print(f"Heterogeneous head dims detected (head_dim={_hd}, global_head_dim={_ghd}), "
                  f"forcing TRITON_ATTN backend")
 
        gen_llm = LLM(
            model=str(gen_dir),
            enforce_eager=args.eager,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_mem,
            max_model_len=seq_len + 16,
            skip_tokenizer_init=True,
            load_format="dummy",
            tensor_parallel_size=args.tensor_parallel_size,
            attention_backend=_attn_backend,
        )
        print("Generated model loaded!")

        sp = SamplingParams(max_tokens=args.max_tokens, temperature=0, detokenize=False)
        prompt_tokens = list(range(1, seq_len + 1))

        for bs in batch_sizes:
            prompts = [{"prompt_token_ids": prompt_tokens} for _ in range(bs)]
            for _ in range(num_warmup):
                gen_llm.generate(prompts, sp, use_tqdm=False)

            t0 = time.perf_counter()
            for _ in range(num_iters):
                gen_llm.generate(prompts, sp, use_tqdm=False)
            elapsed = time.perf_counter() - t0
            total_tokens = bs * (seq_len + args.max_tokens) * num_iters
            throughput = total_tokens / elapsed
            print(f"  bs={bs}, seq={seq_len}: {throughput:.1f} tok/s ({elapsed:.3f}s for {num_iters} iters)")
            results.append({"model": "generated", "batch_size": bs, "seq_len": seq_len,
                            "throughput_tok_s": throughput, "elapsed_s": elapsed})

        del gen_llm
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Generated model benchmark failed: {e}")
        import traceback
        traceback.print_exc()

    # --- Summary ---
    print("\n=== Summary ===")
    print(f"{'Model':<12} {'BS':<6} {'SeqLen':<8} {'Throughput (tok/s)':<20}")
    print("-" * 46)
    for r in results:
        print(f"{r['model']:<12} {r['batch_size']:<6} {r['seq_len']:<8} {r['throughput_tok_s']:<20.1f}")

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {args.output_json}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)

    gen = sub.add_parser("generate", help="Phase 1: generate vLLM model code from Axon")
    gen.add_argument("--axon", required=True, help="Path to .axon file")
    gen.add_argument("--output", required=True, help="Output .py file path")
    gen.add_argument("--class-name", default="GeneratedVLLMModel")
    gen.add_argument("--model-type", default="gemma3_text")
    gen.add_argument("--hidden-size", type=int, default=640)
    gen.add_argument("--vocab-size", type=int, default=262144)
    gen.add_argument("--num-hidden-layers", type=int, default=18)
    gen.add_argument("--num-attention-heads", type=int, default=4)
    gen.add_argument("--num-key-value-heads", type=int, default=1)
    gen.add_argument("--head-dim", type=int, default=256)
    gen.add_argument("--intermediate-size", type=int, default=2048)
    gen.add_argument("--rms-norm-eps", type=float, default=1e-6)
    gen.add_argument("--query-pre-attn-scalar", type=int, default=256)
    gen.add_argument("--extra-config", default=None,
                     help="JSON string of extra config fields to include in .config.json "
                          "(e.g. '{\"use_double_wide_mlp\": true, \"num_kv_shared_layers\": 20}')")

    bench = sub.add_parser("benchmark", help="Phase 2: run throughput benchmark")
    bench.add_argument("--generated-model", required=True, help="Path to generated .py file")
    bench.add_argument("--config", required=True, help="Path to .config.json from generate phase")
    bench.add_argument("--model-type", default="gemma3_text")
    bench.add_argument("--class-name", default="GeneratedVLLMModel")
    bench.add_argument("--batch-sizes", default="1,4,16")
    bench.add_argument("--seq-len", type=int, default=128)
    bench.add_argument("--max-tokens", type=int, default=1,
                        help="Output tokens to generate per prompt (1=prefill only)")
    bench.add_argument("--warmup", type=int, default=3)
    bench.add_argument("--iterations", type=int, default=10)
    bench.add_argument("--gpu-mem", type=float, default=0.3)
    bench.add_argument("--dtype", default="bfloat16")
    bench.add_argument("--native-arch", default="Gemma3ForCausalLM",
                       help="Native vLLM architecture name for the native model")
    bench.add_argument("--tensor-parallel-size", type=int, default=1,
                       help="Tensor parallel size for vLLM")
    bench.add_argument("--eager", action="store_true", default=False,
                        help="Use eager mode (no CUDA graphs / torch.compile)")
    bench.add_argument("--output-json", default=None)

    args = parser.parse_args()
    if args.phase == "generate":
        run_generate(args)
    elif args.phase == "benchmark":
        run_benchmark(args)


if __name__ == "__main__":
    main()
