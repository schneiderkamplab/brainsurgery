# Next Models

This document combines high-priority current gaps and classic baseline families that are good candidates for new `axon` coverage.

| Priority | Target family | Example checkpoint(s) | Why it's worth adding |
|---|---|---|---|
| 1 | Phi-4 | `microsoft/Phi-4-mini-instruct`, `microsoft/Phi-4-mini-reasoning` | Current Microsoft gap; strong small-model target. |
| 2 | Aya Expanse / Cohere | `CohereLabs/aya-expanse-8b` | Major multilingual family not covered locally. |
| 3 | Granite | `ibm-granite/granite-3.3-8b-instruct` | High-profile IBM family with strong enterprise relevance. |
| 4 | EXAONE 4 | `LGAI-EXAONE/EXAONE-4.0-1.2B` | Important Korean/English family supported in Transformers. |
| 5 | StarCoder2 | `bigcode/starcoder2-7b` | Clear missing code-model baseline. |
| 6 | BLOOM | `bigscience/bloom-560m`, `bigscience/bloom-1b7`, `bigscience/bloom-7b1` | Foundational multilingual causal LM family; still a useful historical and compatibility baseline. |
| 7 | OPT | `facebook/opt-1.3b`, `facebook/opt-6.7b`, `facebook/opt-13b` | Standard historical decoder-only baseline family, still widely referenced. |
| 8 | GPT-J / GPT-Neo | `EleutherAI/gpt-j-6b`, `EleutherAI/gpt-neo-1.3B`, `EleutherAI/gpt-neo-2.7B` | Older, but still useful for broad HF compatibility and regression coverage. |
| 9 | XGLM | `facebook/xglm-2.9B`, `facebook/xglm-7.5B` | Useful classic multilingual decoder baseline if broader language-family coverage is desired. |
| 10 | Phi-3.5-MoE / PhiMoE | `microsoft/Phi-3.5-MoE-instruct` | Extends existing Phi coverage into the MoE branch. |
| 11 | GraniteMoe | `ibm-granite/granitemoe-*`, `ibm-research/PowerMoE-3B` | Adds a modern sparse-MoE family outside the Mixtral/Qwen/Gemma cluster. |
| 12 | LFM2 | `LiquidAI/LFM2-1.2B`, `LiquidAI/LFM2-2.6B` | Good architecture-diversity target with on-device relevance. |
| 13 | BitNet | `microsoft/bitnet-b1.58-2B-4T` | High-profile architecture target, though less representative in plain Transformers benchmarking. |

## Suggested First Wave

If only a small number of families should be added next, start with:

1. `phi4`
2. `cohere` or `aya-expanse`
3. `granite`
4. `exaone4`
5. `starcoder2`
6. `bloom`

## Notes

- The current local model coverage already includes families such as Gemma, Llama, Qwen, Mistral, Mixtral, OLMo, Falcon, Phi-3, GPT-OSS, T5, RoBERTa, BERT, and related variants under `brainsurgery/synapse/models`.
- The first five additions above emphasize prominent current gaps in the modern Hugging Face / Transformers ecosystem.
- The classic baseline families improve historical benchmark coverage, compatibility testing, and regression tracking.

## Axon / Synapse TODO

- For GPT-2-style `position_ids_basic`, the immediate primitive-op path is to add `_eq` and `_masked_fill` rather than broadening Axon expressions first.
- Full expression-level elementwise tensor comparisons such as `attn_mask == 0` should remain a separate language/runtime task.
- Once that larger task exists, tensor comparison expressions could eventually replace some dedicated comparison primitives in Axon modules.
