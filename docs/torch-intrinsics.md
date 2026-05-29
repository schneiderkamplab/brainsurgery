# Torch Graph Intrinsics

Backend-specific Graph IR intrinsics are opt-in Torch lowering targets. They are
not Axon definitions and must be introduced only by typed/provenance-based graph
rewrites, never by model-family names.

## CLI Selector

Enable all Torch intrinsics:

```bash
--graph-backend-intrinsics codegen2-torch
```

Enable only selected intrinsics:

```bash
--graph-backend-intrinsics codegen2-torch:selected_expert_packed_gegelu_ffn
--graph-backend-intrinsics codegen2-torch:__torch_sdpa,__torch_rope_pair_apply_factors
```

Selectors accept either the full `__torch_*` name or the suffix without
`__torch_`. Unknown names are rejected.

## Intrinsics To Keep

| Intrinsic | Purpose | Direct isolated evidence? | Current evidence |
|---|---|---:|---|
| `__torch_rope_pair_apply_factors` | Fuse q/k pair RoPE application with shared precomputed factors. | Yes | Isolated GPT-OSS-20B pp=2 selector dump introduced only this intrinsic. Axon improved from no-intrinsics `14.3188s` to `12.9042s` with identical fidelity. |
| Dense gate/up packed-parameter rewrite | Backend-neutral graph rewrite: paired gate/up dense linears become packed-parameter `_linear` plus `_chunk`. | Yes | Formerly tested as `__torch_gate_up_linear_pair`: isolated Llama-3.2-1B selector dump improved from no-intrinsics `1.0860s` to `0.8282s` in skip-HF timing. This is now normal graph optimization, not a `__torch_*` selector. |

## Intrinsics To Assess

| Intrinsic | Purpose | Candidate checkpoints | Current evidence |
|---|---|---|---|
| `__torch_sdpa` | Lower proven attention score/mask/softmax/value to PyTorch SDPA. | `mistralai/Magistral-Small-2509` fired; `mistralai/Codestral-22B-v0.1` fired. | No win in first isolated runs. Magistral pp=2 selector-only timing was slower than the matched no-intrinsics baseline: `16.0807s` vs `14.7219s`, both fidelity-clean. Codestral with `pipeline2-torch`, pp=2, skip-HF also regressed: `26.2292s` vs `22.8585s`. |
| `__torch_rope_apply_factors` | Fuse single-tensor RoPE application with precomputed factors. | `google/gemma-4-E4B` fired; `test/Llama4-Test` and `test/Gemma4-Dense-Test` did not fire. | Neutral first timing. Gemma-4-E4B selector-only timing was `3.4035s` vs no-intrinsics `3.4050s` in skip-HF timing. |
| `_assign_unit_slice` | Backend-neutral graph rewrite for proven unit-slice update/scatter shape. | `state-spaces/mamba-2.8b-hf`, `ai21labs/AI21-Jamba-Reasoning-3B`, and `Zyphra/BlackMamba-2.8B` all fired in earlier selector dumps. | This is now normal graph optimization, not a `__torch_*` selector. Earlier timing was blocked by an unrelated baseline failure with `NoneType` cache/past-state subscripting on both baseline and optimized sides. |
| `__torch_swiglu_ffn` | Fuse dense gate/up/down SwiGLU FFN. | `meta-llama/Llama-3.2-1B` and `Qwen/Qwen2.5-0.5B` fired only when `gate_up_linear_pair` was also enabled; `microsoft/Phi-3-mini-4k-instruct` did not fire. | Mixed. Against a gate-only baseline, Qwen2.5-0.5B improved from `1.1446s` to `1.0806s`, while Llama-3.2-1B regressed from `0.8177s` to `0.8410s`. |
| `__torch_expert_swiglu_ffn` | Fuse per-expert separate gate/up/down SwiGLU. | `ibm-research/PowerMoE-3b` and `test/Phi-MoE-Test` did not fire. | No usable candidate in this pass. |
| `__torch_expert_packed_swiglu_ffn` | Fuse per-expert packed gate-up SwiGLU. | `allenai/OLMoE-1B-7B-0924` fired; `test/FlexOlmo-Test` did not fire. | No win in first isolated run. OLMoE selector-only timing was slower than matched no-intrinsics baseline: `4.2944s` vs `4.0956s` in skip-HF timing. |
| `__torch_selected_expert_swiglu_ffn` | Fuse top-k selected-expert separate gate/up SwiGLU. | `test/Phi-MoE-Test` fired; `ibm-research/PowerMoE-3b` did not fire. | No win in first isolated run. `test/Phi-MoE-Test` selector-only timing was approximately neutral/slower than matched no-intrinsics baseline: `1.9374s` vs `1.9233s` in skip-HF timing. Real-checkpoint evidence still needed. |
| `__torch_selected_expert_packed_swiglu_ffn` | Fuse top-k selected-expert packed gate-up SwiGLU. | `test/Qwen3-MoE-Test` fired; `allenai/OLMoE-1B-7B-0924` did not fire. | Small first-pass win. Qwen3-MoE-Test selector-only timing improved from no-intrinsics `0.9149s` to `0.9005s` in skip-HF timing. This is not yet strong enough to move to “keep”. |
| `__torch_selected_expert_clamped_packed_swiglu_ffn` | Fuse top-k selected-expert packed SwiGLU with finite clamp. | `test/DeepSeek-V2-Test` and `deepseek-ai/DeepSeek-V2-Lite` did not fire. | No usable candidate in this pass. |
| `__torch_selected_expert_packed_gegelu_ffn` | Fuse GPT-OSS-style top-k selected-expert packed GeGLU with explicit `limit` and `alpha`. | `openai/gpt-oss-20b` fired. | No measurable win yet. GPT-OSS-20B pp=2 selector-only timing was `14.3359s` vs no-intrinsics `14.3188s` with identical fidelity; the previous all-intrinsics win was not isolated evidence for this intrinsic. |
| `__torch_selected_expert_relu2_ffn` | Fuse top-k selected-expert ReLU² FFN. | `test/Llama4-Test` did not fire. | No usable candidate in this pass. |
| `__torch_weighted_topk_sum` | Fuse `sum(values * unsqueeze(topk_scores, -1), dim=2)`. | `allenai/OLMoE-1B-7B-0924` and `test/Qwen3-MoE-Test` did not fire, even with likely prerequisite expert intrinsics enabled. | No usable candidate in this pass. |
| `__torch_topk_normalize` | Fuse router top-k weight normalization/cast pattern. | `allenai/OLMoE-1B-7B-0924` and `test/Qwen3-MoE-Test` did not fire, even with likely prerequisite expert intrinsics enabled. | No usable candidate in this pass. |

## Experiment Rule

An A/B run counts as isolated evidence only if the optimized graph dump for the
experimental side contains the candidate intrinsic and no other `__torch_*`
intrinsics that are absent from the baseline.
