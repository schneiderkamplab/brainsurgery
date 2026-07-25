---
status: active
last-confirmed: 2026-07-24
owners: agents
confidence: medium
---

# Model Family Gap Plan

Validated-by: the Transformers `main` auto-model mappings on 2026-07-23 and
repository inspection of `brainsurgery/synapse/models/**/*.axon`.

Depends-on: [benchmarks.md](benchmarks.md), the model-special-casing policy in
[../AGENTS.md](../AGENTS.md), and the Axon semantic-attribution rules in
[AGENTS.md](AGENTS.md).

## Goal

Close model-family gaps in this order:

1. encoder and masked-LM families;
2. causal-LM families;
3. encoder-decoder and seq2seq families;
4. multimodal, audio, OCR, and other specialized generation families.

The encoder/masked-LM, causal-LM, and seq2seq mappings were fully reconciled on
2026-07-24. Specialized-generation mappings are exhaustively classified below
under their current validation contract.

## Change Boundary

During this gap-closing work:

- Model implementations remain limited to newly created Axon files below
  `brainsurgery/synapse/models/`.
- Existing model files, builtins, parser, resolver, typecheck, lowering, graph
  optimizer, code generators, runtimes, pipelines, and benchmark scripts are
  not changed unless a generic missing capability is separately reviewed and
  approved.
- Approved generic extensions on 2026-07-24 are explicit keyed random tensors,
  stable sorting, scatter reduction, and elementwise `acos`; none infer model
  semantics from definition names.
- A family requiring compiler, builtin, or existing-model changes is recorded
  as blocked with the exact missing capability; work then continues with the
  next family.
- Benchmark and diagnostic artifacts go below `log/`.
- Ordinary Axon definition names are never treated as semantic evidence.

Small test checkpoints may be generated when no real checkpoint of at most
4B parameters exists. They are test artifacts, not source changes, and their
matching Axon implementations must still be new files.

## Per-Family Procedure

For every candidate:

1. Confirm the Transformers `model_type`, model class, configuration, and
   masked-LM forward contract from the current installed or upstream source.
2. Search existing Axon files by checkpoint, architecture, config keys, and
   weight layout. If an existing generic Axon file already covers the family,
   record `covered-by` and do not duplicate it.
3. Find the smallest practical real checkpoint at or below 4B parameters.
   Prefer safetensors metadata (`safetensors.total`) or cached
   `._param_count.json`; otherwise calculate from the checkpoint index.
4. If no such real checkpoint exists, create a feature-complete small random
   test checkpoint and a new Axon implementation.
5. Run `brainsurgery synapse axon-benchmark` using the repository-standard
   backend and fidelity fields. Record the exact log path, top-1 result, and
   masked maximum absolute difference.
6. Finish with exactly one status:
   `covered`, `implemented-real`, `implemented-test`, or `blocked`.

## Encoder / Masked-LM Ledger

Known existing coverage to reconfirm:

| Transformers family | Expected existing Axon coverage | Status |
|---|---|---|
| `albert` | `bert/generic-albert.axon` | covered |
| `bart` | `bart/generic-bart.axon` | covered |
| `bert` | `bert/generic-bert.axon` | covered |
| `camembert` | `roberta/generic-roberta.axon` | covered |
| `deberta-v2` | `bert/generic-deberta-v2.axon` | covered |
| `distilbert` | `bert/generic-distilbert.axon` | covered |
| `electra` | `bert/generic-electra.axon` | covered |
| `longformer` | `longformer/generic-longformer.axon` | covered |
| `mbart` | `bart/generic-mbart.axon` | covered |
| `modernbert` | `bert/generic-modernbert.axon` | covered |
| `roberta` | `roberta/generic-roberta.axon` | covered |
| `xlm-roberta` | `roberta/generic-roberta.axon` | covered |

Disposition of the remaining entries in the current masked-LM mapping:

| `model_type` | Evidence checkpoint | Status |
|---|---|---|
| `big_bird` | `google/bigbird-roberta-base` | implemented-real |
| `convbert` | `Finnish-NLP/convbert-base-generator-finnish` | implemented-real |
| `data2vec-text` | `facebook/data2vec-text-base` | implemented-real |
| `deberta` | `microsoft/deberta-base` | implemented-real |
| `ernie` | `nghuyong/ernie-1.0-base-zh`, ERNIE 3.0 base/xbase | implemented-real |
| `esm` | `facebook/esm2_t6_8M_UR50D` | implemented-real |
| `eurobert` | `EuroBERT/EuroBERT-210m` | implemented-real |
| `flaubert` | `flaubert/flaubert_base_cased` | implemented-real |
| `fnet` | `google/fnet-base` | implemented-real |
| `funnel` | `windowsartes/funnel` | implemented-real |
| `ibert` | `kssteven/ibert-roberta-base` | implemented-real |
| `jina_embeddings_v3` | `jinaai/jina-embeddings-v3` | out-of-scope |
| `layoutlm` | `test/LayoutLM-Test` | implemented-test |
| `luke` | `studio-ousia/luke-base`, `studio-ousia/luke-large` | implemented-real |
| `megatron-bert` | `IDEA-CCNL/Erlangshen-MegatronBert-1.3B` | implemented-real |
| `mobilebert` | `google/mobilebert-uncased` | implemented-real |
| `modernvbert` | smallest upstream checkpoint | out-of-scope |
| `mpnet` | `microsoft/mpnet-base` | implemented-real |
| `mra` | `uw-madison/mra-base-512-4` | implemented-real |
| `mvp` | existing seq2seq MVP coverage | covered |
| `nomic_bert` | `nomic-ai/nomic-bert-2048` | implemented-real |
| `nystromformer` | `uw-madison/nystromformer-512` | implemented-real |
| `perceiver` | `deepmind/language-perceiver` | implemented-real |
| `reformer` | `robingeibel/reformer-finetuned` | implemented-real |
| `rembert` | `ibraheemmoosa/xlmindic-rembert-{uni,multi}script` | implemented-real |
| `roberta-prelayernorm` | smallest upstream checkpoint | implemented-real |
| `roc_bert` | smallest upstream checkpoint | out-of-scope |
| `roformer` | `junnyu/roformer_chinese_small` | implemented-real |
| `squeezebert` | `squeezebert/squeezebert-uncased` | implemented-real |
| `tapas` | all six `google/tapas-{tiny,mini,small,medium,base,large}-masklm` variants | implemented-real |
| `xlm` | `FacebookAI/xlm-mlm-en-2048` | implemented-real |
| `xlm-roberta-xl` | smallest upstream checkpoint | implemented-real |
| `xmod` | `facebook/xmod-base` | out-of-scope |
| `yoso` | `uw-madison/yoso-4096` | implemented-real |

### Current Evidence

| Family | Status | Fidelity / blocker |
|---|---|---|
| `data2vec-text` | implemented-real | `top1=True`, masked max abs `0`; `log/masked-lm-gaps-20260723-batch1` |
| `roberta-prelayernorm` | implemented-real | `top1=True`, masked max abs `0`; `log/masked-lm-gaps-20260723-batch2` |
| `xlm` | implemented-real | `top1=True`, masked max abs `1.43e-4`; `log/masked-lm-gaps-20260723-batch5b` |
| `flaubert` | implemented-real | `top1=True`, masked max abs `0`; `log/masked-lm-gaps-20260723-batch5b` |
| `fnet` | implemented-real | explicit Axon DFT; raw `top1=True`, max abs `2.71e-3`; `log/masked-lm-gaps-20260723-batch4` |
| `esm` | implemented-real | `top1=True`, masked max abs `7.96e-4`; `log/masked-lm-gaps-20260723-batch5b` |
| `roformer` | implemented-real | `top1=True`, masked max abs `1.24e-5`; `log/masked-lm-gaps-20260723-batch7` |
| `squeezebert` | implemented-real | grouped convolution expressed by reshape plus batched matmul; `top1=True`, masked max abs `1.07e-2`; `log/masked-lm-gaps-20260723-batch7` |
| `eurobert` | implemented-real | current Transformers required explicit checkpoint restoration and rebuilding its nonpersistent RoPE frequencies after `from_pretrained`; CPU `top1=True`, masked max abs `4.24e-5`; `log/eurobert-fixed-cpu-20260724-r2` |
| `mpnet` | implemented-real | typecheck2 now refines inferred helper interfaces from consistent callsites without changing declared polymorphic interfaces; `top1=True`, masked max abs `0`; `log/mpnet-typecheck2-fix-20260724-b` |
| `mra` | implemented-real | portable dense evaluation of MRA's selected high-resolution blocks plus low-resolution correction; native 512-token CPU `top1=True`, masked max abs `0`; `log/mra-cpu-fidelity-512-20260724` |
| `xlm-roberta-xl` | implemented-real | `top1=True`, masked max abs `1.01e-4`; `log/masked-lm-xlm-roberta-xl-20260724` |
| `mobilebert` | implemented-real | full trigram, bottleneck, no-norm, repeated-inner-FFN, and split MLM projection path; `top1=True`, masked max abs `4.12e-5`; `log/masked-lm-mobilebert-20260724` |
| `ibert` | implemented-real | the public checkpoint has `quant_mode=false`, for which the quantization wrappers reduce exactly to ordinary float operations; `top1=True`, masked max abs `3.77e-5`; `log/masked-lm-ibert-20260724` |
| `convbert` | implemented-real | the real Finnish generator exercises trigram embeddings, bottleneck projection, dynamic convolution, depthwise value convolution, and the MLM head; `top1=True`, masked max abs `2.19e-5`; `log/real-checkpoint-evidence-cpu-20260724-rerun5` |
| `ernie` | implemented-real | ERNIE 1.0 and ERNIE 3.0 select ReLU/GELU and optional task embeddings from config. The 1.0/base/xbase rows are all top-1 equal with max abs `3.43e-5`, `1.12e-4`, and `2.22e-4`; `log/real-checkpoint-evidence-cpu-20260724-ernie-all-final`. The nominal 3.0 medium repository was excluded because its checkpoint has no MLM-head tensors. |
| `layoutlm` | implemented-test | persisted two-layer MLM checkpoint; `top1=True`, masked max abs `0`; `log/masked-lm-head-tests-20260724-b` |
| `luke` | implemented-real | the official base and large checkpoints use a tied token decoder while retaining their distinct entity decoder; both rows are exact (`top1=True`, max abs `0`); `log/real-checkpoint-evidence-cpu-20260724-rerun5`, `log/real-checkpoint-evidence-cpu-20260724-masked-variants` |
| `rembert` | implemented-real | both public uniscript and multiscript checkpoints exercise distinct input/output embedding dimensions; top-1 equal with max abs `3.24e-5` and `3.81e-5`; `log/real-checkpoint-evidence-cpu-20260724-batch1`, `log/real-checkpoint-evidence-cpu-20260724-masked-variants` |
| `tapas` | implemented-real | all six official mask-LM sizes (tiny through large) exercise all seven token-type embedding tables and are exact (`top1=True`, max abs `0`); `log/real-checkpoint-evidence-cpu-20260724-batch1`, `log/real-checkpoint-evidence-cpu-20260724-masked-variants` |
| `nystromformer` | implemented-real | the published 512 checkpoint takes its exact ordinary-attention branch because `num_landmarks == segment_means_seq_len`; `top1=True`, masked max abs `2.19e-5`; `log/masked-lm-nystromformer-20260724-b` |
| `perceiver` | implemented-real | text preprocessor, latent encoder, decoder query cross-attention, and tied byte decoder; `top1=True`, masked max abs `3.92e-3`; `log/masked-lm-perceiver-2048-20260724-b` |
| `nomic_bert` | implemented-real | generic HF embedding-device discovery handles remote models that omit `get_input_embeddings`; `top1=True`, masked max abs `2.47e-4`; `log/masked-lm-hf-integration-fixes-20260724` |
| `funnel` | implemented-real | the real third-party checkpoint exercises all three encoder blocks, pooled-query attention, relative-shift positions, odd-length ceil pooling, and the decoder; top-1 is equal, with a provisional masked max abs `1.64e-2`; `log/real-checkpoint-evidence-cpu-20260724-funnel-trace` |
| `yoso` | implemented-real | the public checkpoint uses its exact dense angular-expectation path (`use_expectation=true`); `top1=True`, masked max abs `5.72e-5`; `log/masked-lm-yoso-20260724-e` |
| `deberta` | implemented-real | DeBERTa-v1 interleaved per-head QKV packing, q/v bias, and c2p+p2c relative attention; `top1=True`, masked max abs `2.57e-5`; `log/masked-lm-deberta-megatron-final-20260724` |
| `megatron-bert` | implemented-real | pre-norm layers, final encoder normalization, and the checkpoint's actual decoder-bias path; `top1=True`, masked max abs `4.48e-5`; `log/masked-lm-deberta-megatron-final-20260724` |
| `big_bird` | implemented-real | exact short-sequence full attention and long-sequence eval block sparsity; eval-time repeated block-0 keys are represented by their equivalent `log(1 + num_random_blocks)` score bias; standard `top1=True`, max abs `1.17e-4`, and 713-token sparse `top1=True`, max abs `3.15e-5`; `log/masked-lm-big-bird-20260724-b`, `log/masked-lm-big-bird-sparse-20260724-c` |
| `reformer` | implemented-real | the real checkpoint exercises axial positions, local and factored-bucket LSH layers, seeded rotations, stable rank construction, cross-hash previous-chunk adjacency, duplicate key occurrences, padding, and reversible two-stream state; `top1=True`, masked max abs `1.53e-5`; `log/real-checkpoint-evidence-cpu-20260724-masked-variants` |

### Audited Dispositions

These are blockers in the currently exported model-level Axon surface, not
claims that the algorithms are intrinsically unsuitable for Axon:

| Family | Exact missing capability found during source audit |
|---|---|
| `eurobert` | **Closed:** current Transformers can leave most remote-model matrices randomly initialized and can materialize the nonpersistent `inv_freq` buffer from meta storage without initializing it. The HF loading integration now validates a representative matrix against safetensors, restores the checkpoint explicitly when necessary, and rebuilds rotary frequencies through the module's own `rope_init_fn`; CPU benchmark top-1 is equal with masked max abs `4.24e-5` |
| `jina_embeddings_v3` | **Out of scope:** the public checkpoint stores LoRA-parametrized embedding weights and has no trained masked-LM head, so it is unsuitable as direct masked-LM evidence |
| `modernvbert` | **Out of scope:** the masked model is multimodal: its forward contract includes pixel tensors or image hidden states and a SigLIP vision path, which the text-only masked-LM benchmark does not supply |
| `mra` | **Closed:** the Axon source expresses block scoring, top-k block selection, sampled high-resolution attention, low-resolution attention, and their normalizer correction without a custom primitive. Because upstream Transformers returns zeros when its out-of-tree CUDA extension is absent, the allowed HF integration layer installs a direct portable implementation of the same equations for reference execution; native 512-token CPU fidelity is exact |
| `roc_bert` | **Out of scope:** the tokenizer supplies learned pronunciation and glyph-shape IDs, but the current masked-LM benchmark deliberately forwards only token IDs and attention masks |
| `xmod` | **Out of scope:** `facebook/xmod-base` has no default language, while adapter selection requires a language/domain input absent from the benchmark forward contract |

The `nystromformer-512.axon` evidence covers the published checkpoint's exact
ordinary-attention configuration. It does not claim coverage of the alternate
iterative Nyström inverse branch. Perceiver emits a fixed 2048-position output,
so its evidence run uses a 2048-token input; shorter benchmark masks are not
shape-compatible with that public model contract.

## Causal-LM Gap Ledger

Last mapping snapshot: the installed Transformers
`MODEL_FOR_CAUSAL_LM_MAPPING_NAMES` on 2026-07-24, containing 160 entries.
Names from the older provisional queue that are absent from this mapping are
not retained as current gaps.

### New measured coverage

Every row below reached HF and Axon execution in float32 without fallback.

| Family | New Axon source | Evidence checkpoint | Fidelity evidence |
|---|---|---|---|
| `codegen` | `models/codegen/generic-codegen.axon` | `Salesforce/codegen-350M-mono` | top-1 equal, max abs `2.34e-5`; `log/causal-gaps-20260724-codegen` |
| `biogpt` | `models/biogpt/generic-biogpt.axon` | `microsoft/biogpt` | top-1 equal, max abs `3.15e-5`; `log/causal-gaps-20260724-biogpt` |
| `gpt_neo` | `models/gpt_neo/generic-gpt-neo.axon` | `EleutherAI/gpt-neo-125m` | top-1 equal, max abs `1.97e-4`; `log/causal-gaps-20260724-gpt-neo-batch-fix` |
| `gpt_bigcode` | `models/gpt_bigcode/generic-gpt-bigcode.axon` | `bigcode/tiny_starcoder_py` | top-1 equal, max abs `1.53e-5`; `log/causal-gaps-20260724-gpt-bigcode-batch-fix` |
| `phi` | `models/phi/generic-phi.axon` | `microsoft/phi-1_5` | top-1 equal, max abs `2.29e-5`; `log/causal-gaps-20260724-phi` |
| `stablelm` | `models/stablelm/generic-stablelm.axon` | `stabilityai/stablelm-2-1_6b` | top-1 equal, max abs `1.44e-4`; `log/causal-gaps-20260724-stablelm` |
| `ctrl` | `models/ctrl/generic-ctrl.axon` | `Salesforce/ctrl` | top-1 equal, max abs `8.58e-6`; `log/causal-gaps-20260724-ctrl` |
| `olmo` | `models/olmo/generic-olmo.axon` | `allenai/OLMo-1B-hf` | top-1 equal, max abs `4.63e-5`; `log/causal-gaps-20260724-olmo-final` |
| `lfm2` | `models/lfm2/generic-lfm2.axon` | `LiquidAI/LFM2-350M` | top-1 equal, max abs `3.43e-5`; `log/causal-gaps-20260724-lfm2-rerun` |
| `rwkv` | `models/rwkv/generic-rwkv.axon` | `RWKV/rwkv-4-169m-pile` | exact, max abs `0`; `log/causal-gaps-20260724-rwkv-rescale` |
| `falcon_mamba` | `models/falcon_mamba/generic-falcon-mamba.axon` | `tiiuae/falcon-mamba-tiny-dev` | top-1 equal, max abs `2.15e-6`; `log/causal-gaps-20260724-falcon-mamba-tied` |
| `gpt_neox_japanese` | `models/gpt_neox_japanese/generic-gpt-neox-japanese.axon` | `abeja/gpt-neox-japanese-2.7b` | top-1 equal, max abs `2.10e-5`; native tokenizer loaded explicitly; `log/causal-gaps-20260724-gpt-neox-japanese-tokenizer` |
| `doge` dense | `models/doge/generic-doge-dense.axon` | `SmallDoge/Doge-20M` | top-1 equal, max abs `1.72e-5`; `log/causal-gaps-20260724-doge-heads` |
| `modernbert-decoder` | `models/modernbert_decoder/generic-modernbert-decoder.axon` | `onnx-internal-testing/tiny-random-ModernBertDecoderForCausalLM` | top-1 equal, max abs `3.62e-5`; `log/causal-gaps-20260724-modernbert-decoder` |
| `mpt` | `models/mpt/generic-mpt.axon` | `test/MPT-Test` | exact masked logits; `log/causal-gaps-20260724-mpt-test-vocab` |
| `persimmon` | `models/persimmon/generic-persimmon.axon` | `test/Persimmon-Test` | top-1 equal, max abs `4.77e-7`; `log/causal-gaps-20260724-persimmon-test` |
| `arcee` | `models/arcee/generic-arcee.axon` | `onnx-internal-testing/tiny-random-ArceeForCausalLM` | top-1 equal, max abs `7.72e-5`; `log/causal-gaps-20260724-arcee-tiny` |
| `falcon_h1` | `models/falcon_h1/generic-falcon-h1.axon` | `tiiuae/Falcon-H1-0.5B-Base` | cached generation is implemented entirely in Axon with a per-layer attention/recurrent-state cache; generated completions match HF with top-1 equality and max abs `1.53e-5`; `log/causal-close-falcon-h1-cache-r2-20260724.txt` |
| `cohere2` | `models/cohere2/generic-cohere2.axon` | `test/Cohere2-Test` | top-1 equal, max abs `4.77e-7`; mixed sliding/full attention is exercised; `log/causal-gaps-20260724-cohere2-test-vocab` |
| `jais2` | `models/jais2/generic-jais2.axon` | `test/Jais2-Test` | top-1 equal, max abs `4.77e-7`; `log/causal-gaps-20260724-jais2-test` |
| `qwen2_moe` | `models/qwen2_moe/generic-qwen2-moe.axon` | `test/Qwen2-MoE-Test` | top-1 equal, max abs `1.19e-7`; mixed sliding/full attention, sparse routing, unnormalized top-k, and the shared expert are exercised; `log/causal-gaps-20260724-qwen2-moe-test-separate` |
| `solar_open` | `models/solar_open/generic-solar-open.axon` | `test/SolarOpen-Test` | top-1 equal, max abs `1.49e-7`; grouped correction-biased sigmoid routing and the shared expert are exercised; `log/causal-gaps-20260724-solar-open-test` |
| `granitemoeshared` | `models/granitemoeshared/generic-granitemoe-shared.axon` | `test/GraniteMoeShared-Test` | top-1 equal, max abs `1.49e-7`; routed and sibling shared SwiGLU branches are both exercised; `log/causal-gaps-20260724-granitemoe-shared-scope` |
| `aria_text` | `models/aria_text/generic-aria-text.axon` | `test/AriaText-Test` | top-1 equal, max abs `1.49e-7`; transposed grouped experts and the parallel shared expert are exercised; `log/causal-gaps-20260724-aria-text-test` |
| `cohere2_moe` | `models/cohere2_moe/generic-cohere2-moe.axon` | `test/Cohere2-MoE-Test` | top-1 equal, max abs `8.94e-8`; dense/sparse MLP layers, sliding/full attention, sigmoid routing, and shared-expert averaging are exercised; `log/causal-gaps-20260724-cohere2-moe-test` |
| `cwm` | `models/cwm/generic-cwm.axon` | `test/Cwm-Test` | top-1 equal, max abs `1.19e-7`; full/sliding attention and Llama-3 frequency-scaled RoPE are exercised; `log/causal-gaps-20260724-cwm-test` |
| `hunyuan_v1_dense` | `models/hunyuan_v1_dense/generic-hunyuan-v1-dense.axon` | `test/HunYuan-Dense-V1-Test` | top-1 equal, max abs `1.49e-7`; post-RoPE query/key normalization and biased projections are exercised; `log/causal-gaps-20260724-hunyuan-v1-dense-test` |
| `seed_oss` | `models/seed_oss/generic-seed-oss.axon` | `test/Seed-OSS-Test` | top-1 equal, max abs `1.49e-7`; independent attention-input, attention-output, and MLP bias flags are exercised; `log/causal-gaps-20260724-seed-oss-test` |
| `hyperclovax` | `models/hyperclovax/generic-hyperclovax.axon` | `test/HyperCLOVAX-Test` | top-1 equal, max abs `8.94e-8`; post-sublayer norms, embedding/logit/residual multipliers, and custom attention scaling are exercised; `log/causal-gaps-20260724-hyperclovax-test`. The real 1.5B repository is gated (`401`), so public-checkpoint confirmation is unavailable. |
| `hunyuan_v1_moe` | `models/hunyuan_v1_moe/generic-hunyuan-v1-moe.axon` | `test/HunYuan-MoE-V1-Test` | top-1 equal, max abs `1.49e-7`; dense/sparse layers, routed experts, and the shared MLP are exercised; `log/causal-gaps-20260724-hunyuan-v1-moe-test-rerun` |
| `afmoe` | `models/afmoe/generic-afmoe.axon` | `test/AFMoE-Test` | top-1 equal, max abs `2.24e-7`; dense/sparse layers, mixed attention windows, correction-biased routing, attention gating, and dual post norms are exercised; `log/causal-gaps-20260724-afmoe-test` |
| `exaone_moe` | `models/exaone_moe/generic-exaone-moe.axon` | `test/EXAONE-MoE-Test` | top-1 equal, max abs `1.79e-7`; dense/sparse layers, mixed NoPE/RoPE attention, correction-biased grouped routing, and the shared expert are exercised; `log/causal-gaps-20260724-exaone-moe-test-rerun2` |
| `hy_v3` | `models/hy_v3/generic-hy-v3.axon` | `test/HY-V3-Test` | top-1 equal, max abs `3.73e-8`; dense/sparse layers, correction-biased sigmoid routing, shared experts, and fp32 combine are exercised; `log/causal-gaps-20260724-hy-v3-test-rerun` |
| `laguna` | `models/laguna/generic-laguna.axon` | `test/Laguna-Test` | top-1 equal, max abs `1.19e-7`; per-layer head counts, mixed partial/full RoPE, softplus attention gating, grouped routing, and the shared expert are exercised; `log/causal-gaps-20260724-laguna-test-rerun3` |
| `dots1` | `models/dots1/generic-dots1.axon` | `test/Dots1-Test` | top-1 equal, max abs `1.49e-7`; dense/sparse layers, grouped correction-biased routing, mixed attention windows, and the shared expert are exercised; `log/causal-gaps-20260724-dots1-test` |
| `minimax_m2` | `models/minimax_m2/generic-minimax-m2.axon` | `test/MiniMax-M2-Test` | top-1 equal, max abs `1.49e-7`; whole-projection query/key normalization, correction-biased sigmoid routing, and every-layer MoE execution are exercised; `log/causal-gaps-20260724-minimax-m2-test-rerun` |
| `ernie4_5_moe` | `models/ernie4_5_moe/generic-ernie4-5-moe.axon` | `test/Ernie4.5-MoE-Test` | top-1 equal, max abs `1.73e-3`; dense/sparse layer selection, fp32 correction-biased softmax routing, shared experts, and tied output embeddings are exercised; `log/causal-gaps-20260724-ernie4-5-moe-test` |
| `lfm2_moe` | `models/lfm2_moe/generic-lfm2-moe.axon` | `test/LFM2-MoE-Test` | top-1 equal, max abs `1.19e-7`; convolution and attention layers plus dense and sigmoid-routed sparse FFNs are exercised; `log/causal-close-lfm2-moe-r4` |
| `dbrx` | `models/dbrx/generic-dbrx.axon` | `test/DBRX-Test` | top-1 equal, max abs `1.34e-7`; clipped packed GQA plus flattened DBRX expert tensors and L1-normalized top-k routing are exercised; `log/causal-close-dbrx-r5` |
| `jetmoe` | `models/jetmoe/generic-jetmoe.axon` | `test/JetMoE-Test` | top-1 equal, max abs `5.96e-8`; top-k-routed query/output attention projections, whole-head-block KV repetition, and routed SwiGLU FFNs are exercised; `log/causal-close-jetmoe-r3` |
| `bamba` | `models/bamba/generic-bamba.axon` | `test/Bamba-Test` | top-1 equal, max abs `8.94e-8`; one Mamba-2 layer, one GQA layer, and dense SwiGLU FFNs are exercised with exact full-prefix generation semantics; `log/causal-close-bamba-r1` |
| `granitemoehybrid` | `models/granitemoehybrid/generic-granitemoehybrid.axon` | `test/GraniteMoeHybrid-Test` | top-1 equal, max abs `8.20e-8`; Mamba-2 and GQA layers, packed routed experts, the shared MLP, optional RoPE, and Granite scaling factors are exercised; `log/causal-close-granitemoehybrid-r3` |
| `youtu` | `models/youtu/generic-youtu.axon` | `test/Youtu-Test` | top-1 equal, max abs `3.81e-6`; query LoRA, compressed latent KV, unequal QK/value head dimensions, and interleaved partial RoPE are exercised; `log/causal-close-youtu-r2` |
| `recurrent_gemma` | `models/recurrent_gemma/generic-recurrent-gemma.axon` | `test/RecurrentGemma-Test` | top-1 equal, max abs `9.83e-4`; grouped RG-LRU gates, reset-aware recurrence, depthwise causal convolution, partial-RoPE sliding attention, and unit-offset RMSNorm are exercised; `log/causal-close-recurrent-gemma-r2` |
| `qwen3_next` | `models/qwen3_next/generic-qwen3-next.axon` | `test/Qwen3-Next-Test` | top-1 equal, max abs `1.49e-7`; gated-delta recurrence, depthwise causal convolution, partial-RoPE attention, separate routed experts, and the shared expert are exercised; `log/causal-close-qwen3-next-r7` |
| `qwen3_5` | `models/qwen3_5/generic-qwen3-5.axon` | `test/Qwen3.5-Test` | top-1 equal, max abs `1.19e-7`; separate-projection gated-delta recurrence, depthwise causal convolution, partial-RoPE attention, and dense SwiGLU are exercised; `log/causal-close-qwen3-5-r1` |
| `qwen3_5_moe` | `models/qwen3_5/generic-qwen3-5.axon` | `test/Qwen3.5-MoE-Test` | top-1 equal, max abs `1.49e-7`; the shared Qwen3.5 token mixers plus separate routed experts and the gated shared expert are exercised; `log/causal-close-qwen3-5-both-r1` |
| `olmo_hybrid` | `models/olmo_hybrid/generic-olmo-hybrid.axon` | `test/OLMoHybrid-Test` | top-1 equal, max abs `1.79e-7`; linear-attention and full-attention layers, gated-delta recurrence, depthwise causal convolution, optional RoPE, and dense SwiGLU are exercised; `log/causal-close-olmo-hybrid-r3` |
| `zamba` | `models/zamba/generic-zamba.axon` | `test/Zamba-Test` | top-1 equal, max abs `1.04e-3`; interleaved multi-head Mamba recurrence, shared transformer injection, tied transformer parameters, and hybrid/plain layer paths are exercised; `log/causal-close-zamba-r4` |
| `zamba2` | `models/zamba2/generic-zamba2.axon` | `test/Zamba2-Test` | top-1 equal, max abs `8.14e-5`; grouped Mamba-2, two shared-memory source blocks and their reuse, per-use attention/MLP adapters, memory RoPE, and a plain Mamba layer are exercised; `log/causal-close-zamba2-r2` |
| `cpmant` | `models/cpmant/generic-cpmant.axon` | `test/CPMAnt-Test` | fully optimized generation is exact when the HF reference uses its correct uncached path: completions match, top-1 is equal, and max abs is `1.60e-5`; `log/causal-close-cpmant-reference-20260724.txt`. HF cached generation remains an upstream defect because it grows keys to width 7 while recomputing a width-3 position bias. |
| `xlstm` | `models/xlstm/generic-xlstm.axon` | `test/xLSTM-Test` | fully optimized generation is exact when the HF reference uses its correct uncached path: completions match, top-1 is equal, and max abs is `8.34e-7`; `log/causal-close-xlstm-reference-20260724.txt`. HF cached generation remains an upstream defect because it allocates a recurrent state with the q/k width where the unequal value width is required. |
| `glm_moe_dsa` | `models/glm_moe_dsa/generic-glm-moe-dsa.axon` | `test/GLM-MoE-DSA-Test` | cached generation now carries both attention KV and the indexer's independent key history in each layer cache; completions match HF, top-1 is equal, and max abs remains the accepted family value `7.89e-2`; `log/causal-close-glm-dsa-cache-r2-20260724.txt`. |
| `minimax` | `models/minimax/generic-minimax.axon` | `test/MiniMax-Test` | fully optimized forward preserves top-1 with max abs `1.91e-2`, accepted for this family; the two-layer test exercises blockwise Lightning Attention, RoPE GQA, packed top-2 experts, and non-unit residual scaling; `log/causal-close-minimax-forward-r5`. The isolated Lightning recurrence agrees with HF to `1.49e-8`. |
| `openai-gpt` | `models/openai_gpt/generic-openai-gpt.axon` | `openai-community/openai-gpt` | top-1 equal, max abs `1.34e-5`; Transformers' legacy OpenAI GPT maps config `"gelu"` to `gelu_new`, now matched by the Axon source; `log/causal-fixes-20260724-openai-gpt-r2`. |
| `mamba2` | `models/mamba2/generic-mamba2.axon` | `AntonV/mamba2-130m-hf` | top-1 equal, max abs `1.37e-4`; shared Python literal serialization now handles nested non-finite config values, and the source handles tied output embeddings without requiring `lm_head.weight`; `log/causal-fixes-20260724-mamba2-cpu-r2`. |
| `diffllama` | `models/diffllama/generic-diffllama.axon` | `kajuma/DiffLlama-0.3B-handcut` | top-1 equal, masked max abs `4.77e-6` on an unequal left-padded batch; the source now reads top-level `rope_theta` and applies the checkpoint's Llama-3 frequency scaling; `log/causal-fixes-20260724-diffllama-batch`. |
| `helium` | `models/helium/generic-helium.axon` | `kyutai/helium-1-preview-2b` | top-1 equal, max abs `1.45e-5`; the source now uses Helium's interleaved even/odd RoPE convention; `log/causal-fixes-20260724-helium-r2`. |
| `vaultgemma` | `models/vaultgemma/generic-vaultgemma.axon` | `test/VaultGemma-Test` | top-1 equal, max abs `2.98e-7`; explicit grouping preserves the intended final softcap `(tanh(logits / cap)) * cap`; `log/causal-fixes-20260724-vaultgemma-r2`. |
| `nanochat` | `models/nanochat/generic-nanochat.axon` | `nanochat-students/nanochat-d20` | top-1 equal, masked max abs `1.48e-5` on an unequal right-padded batch; the padding-side pragma survives the complete frontend, and the source now matches NanoChat's inverse half-split RoPE rotation; `log/causal-fixes-20260724-nanochat-cpu-r2`. |

### Real-checkpoint expansion

The 39 causal ledger rows whose original fidelity evidence used a
feature-complete `test/*` checkpoint now each declare at least one canonical
real checkpoint. The declarations include all architecture-defining public,
official, or author-published base/instruct variants found during the
2026-07-24 audit; they do not attempt to enumerate unrelated community
fine-tunes.

`bert_generation` is the only causal generic that remains test-only. No trained
standalone `BertGenerationDecoder` causal checkpoint was found; available uses
are encoder-decoder compositions rather than independently trained causal
models.

Declaration and frontend evidence:

- `log/causal-real-checkpoints-cpu-20260724/checkpoint-config-audit.csv`
  audits 178 real declarations across 71 test-backed causal generics. Of these,
  147 configs are publicly readable, 29 require repository authorization, and
  two are maintained local checkpoint aliases.
- `log/causal-real-checkpoints-cpu-20260724/test-backed-real-size-audit.csv`
  records parameter metadata for the 103 real declarations attached
  specifically to the 39 formerly test-evidenced ledger rows. It confirms that
  nearly all remaining unmeasured public families start above 6B parameters;
  sub-4B candidates were measured or are authorization-gated.
- Public configs resolve to the intended Transformers architecture family.
  `meituan-longcat/LongCat-Flash-Chat` is the metadata exception: its public
  config omits `model_type` but declares `LongcatFlashForCausalLM`.
- All 72 causal generics containing test checkpoints pass the complete
  parse/resolve/normalize/flatten/typecheck frontend, with zero failures;
  validated-by
  `log/causal-real-checkpoints-cpu-20260724/frontend-validation-2/status.csv`.

Real-weight CPU float32 evidence obtained during the expansion:

| Outcome | Families / evidence |
|---|---|
| healthy | Cohere2 public tiny `8.38e-9`; Youtu `4.39e-5`; GraniteMoeShared `8.77e-5`; OLMo Hybrid `1.43e-5`; Zamba `1.29e-5`; xLSTM `3.26e-4`; Persimmon `2.19e-5`; LFM2-MoE `2.57e-5` |
| top-1 equal, above `1e-2` | Qwen3.5 0.8B `1.94e-2`; JetMoE 8B `2.03e-2` |
| substantial numerical mismatch | Hunyuan 0.5B `2.08`; GraniteMoeHybrid 350M `2.70`; Zamba2 1.2B `3.24`; causal Reformer `3.07`; BLT 1B `11.08` |
| HF reference/access blocked | PhoGPT remote code has an unresolved external dependency; Cohere checkpoints are gated; Trinity Nano's remote HF config omits a required `pad_token_id` |
| not run in this CPU tranche | checkpoints whose size or access requirements make CPU float32 evidence impractical; declarations and frontend validation are not treated as runtime fidelity evidence |

The raw rows and per-checkpoint logs are under
`log/causal-real-checkpoints-cpu-20260724/`. Historical failed attempts in that
directory are superseded by the latest successful row for a checkpoint and are
not counted as current outcomes.

### Implemented but not accepted

No causal-LM implementation in this tranche is waiting on acceptance of its
feature-complete test evidence. The real-checkpoint mismatches listed above are
open fidelity follow-ups and must not be represented as healthy real-weight
evidence.

### Exact aliases and architectural coverage

These do not warrant copied Axon implementations:

| Mapping | Covered by | Evidence |
|---|---|---|
| `gpt-sw3` | `gpt2` | Transformers maps both to `GPT2LMHeadModel` |
| `llama4` | `llama4_text` | both map to `Llama4ForCausalLM`; the conditional-generation wrapper remains in the multimodal tranche |
| `gemma4` | existing `models/gemma4/generic-gemma-4-{dense,e,moe}.axon` | the mapping queue name is already represented by the repository's Gemma 4 implementations |
| `qwen3_5_text` | `qwen3_5` | both mapping keys resolve to the exact `Qwen3_5ForCausalLM` class |
| `qwen3_5_moe_text` | `qwen3_5_moe` | both mapping keys resolve to the exact `Qwen3_5MoeForCausalLM` class |
| public `JetBrains/Mellum-4b-*` checkpoints | `llama` | their published config declares `model_type: llama` and `LlamaForCausalLM`, not the newer `MellumForCausalLM` layout |
| dense `ernie4_5` | `qwen2` architecture | source audit shows the same separate GQA projections, RoPE, pre-norm SwiGLU, config dimensions, and parameter layout; the MoE mapping is separate |

### Task-specific mappings

The causal auto mapping also exposes encoder and encoder-decoder classes such
as `bert`, `big_bird`, `bigbird_pegasus`, `camembert`, `data2vec-text`,
`electra`, `megatron-bert`, `rembert`, `roberta`,
`roberta-prelayernorm`, `roformer`, `xlm`, `xlm-roberta`,
`xlm-roberta-xl`, `bart`, `blenderbot`, `blenderbot-small`, `marian`,
`mbart`, `mvp`, `pegasus`, and `plbart`. Existing masked-LM or seq2seq
sources are not automatically counted as causal fidelity evidence. A causal
checkpoint/configuration must still be tested when one exists.

`bert-generation` is now covered by
`models/bert_generation/generic-bert-generation.axon` and the feature-complete
`test/BertGeneration-Test` decoder checkpoint. The three-layer cached decoder
matches HF generation exactly at masked max abs `1.79e-7`; validated-by
`log/causal-bert-generation-20260724`.

### Deferred non-text wrappers

`emu3`, `fuyu`, `gemma3`, `gemma3n`, `git`, `got_ocr2`, `mllama`,
`moshi`, `musicgen`, `musicgen_melody`, `phi4_multimodal`, `trocr`, and
`whisper` belong to the multimodal/audio/OCR tranche. Their text components are
not silently treated as full wrapper coverage.

### Causal mappings with non-standalone input contracts

| Family | Concrete blocker under the current causal benchmark contract |
|---|---|
| `gemma4_assistant` | **Out of scope.** `Gemma4AssistantForCausalLM.forward` rejects ordinary standalone token input and requires `inputs_embeds` plus `shared_kv_states` produced by a parent Gemma 4 model; it is an assisted-decoding component, not an independently benchmarkable token-ID LM. |
| `bitnet` | **Closed.** `models/bitnet/generic-bitnet.axon` expresses online per-token int8 activation quantization and ternary weight quantization in Axon and uses the generic ties-to-even `Math.round` primitive. Matching Transformers' single flat weight-mean reduction removes the current-snapshot regression: the real 2B checkpoint preserves top-1 and produces identical completions at max abs `0.557`; validated-by `log/causal-bitnet-flat-mean-20260724` and `log/causal-bitnet-flat-mean-generate-20260724`. |
| `reformer` | **Closed.** `models/reformer/generic-reformer-causal.axon` expresses causal local and LSH attention, stable bucket sorting, axial positions, and uncached generation with existing Axon operations. The feature-complete five-layer test is top-1 equal with max abs `1.94e-7`, and generated completions match HF. Validated-by `log/causal-reformer-cpu-20260724-c` and `log/causal-reformer-cpu-20260724-d`. |
| `xmod` | **Out of scope.** Adapter selection requires a language/domain identifier that is not part of the causal benchmark’s token-ID forward contract. |
| `roc_bert` | **Out of scope.** The model consumes pronunciation and glyph-shape IDs alongside token IDs. Those tokenizer-derived tensors are not present in the causal benchmark input contract, so a token-ID-only parity run would omit learned inputs. |
| `blt` | **Closed.** `models/blt/generic-blt.axon` implements entropy patching densely: per-token patch IDs, max-length splitting, scatter-reduced patch states, local encoder and decoder cross-attention, the global transformer, hash embeddings, and all learned patch projections remain ordinary Axon. The feature-complete test matches HF generation and top-1 at max abs `1.19e-6`; validated-by `log/causal-blt-r7-20260724`. |
| `xlnet` | **Closed.** `models/xlnet/generic-xlnet.axon` implements relative shift, content/position attention, two-stream dummy-token prediction, and XLNet's two-token memory recomputation window as ordinary Axon. The real `xlnet-base-cased` checkpoint is top-1 equal at max abs `1.05e-4` in forward and produces identical cached-generation completions at max abs `7.82e-5`. Validated-by `log/causal-xlnet-base-cpu-20260724-forward-c` and `log/causal-xlnet-base-cpu-20260724-generate-a`. |

### Newly measured text-family coverage

| Family | Axon source | Checkpoint | Evidence |
|---|---|---|---|
| `longcat_flash` | `models/longcat_flash/generic-longcat-flash.axon` | `test/LongCat-Flash-Test` | CPU fully optimized forward top-1 equal, max abs `7.37e-5`; both MLA sublayers, shortcut MoE, learned and identity experts, interleaved RoPE, and dense MLPs are exercised; validated-by `log/causal-close-longcat-flash-cpu-r3` |
| `gemma3n_text` | `models/gemma3n_text/generic-gemma3n-text.axon` | `test/Gemma3n-Text-Test` | CPU fully optimized forward top-1 equal, max abs `1.21e-2`; AltUp, Laurel, per-layer embeddings, mixed local/full attention, shared KV, and Gaussian activation sparsity are exercised; validated-by `log/causal-close-gemma3n-text-cpu-r7` |
| `reformer` | `models/reformer/generic-reformer-causal.axon` | `test/Reformer-Causal-Test` | CPU forward and generation top-1 equal, max abs `1.94e-7`; causal local and LSH attention, axial positions, random hashing, stable sorting, reversible carries, and generation are exercised; validated-by `log/causal-reformer-cpu-20260724-c` and `log/causal-reformer-cpu-20260724-d` |
| `xlnet` | `models/xlnet/generic-xlnet.axon` | `xlnet/xlnet-base-cased` | CPU real-checkpoint forward top-1 equal, max abs `1.05e-4`; cached two-stream generation gives identical completions, max abs `7.82e-5`; validated-by `log/causal-xlnet-base-cpu-20260724-forward-c` and `log/causal-xlnet-base-cpu-20260724-generate-a` |
| `bitnet` | `models/bitnet/generic-bitnet.axon` | `microsoft/bitnet-b1.58-2B-4T-bf16` | CPU real-checkpoint forward and generation preserve top-1 and generated tokens at max abs `0.557`; validated-by `log/causal-bitnet-flat-mean-20260724` and `log/causal-bitnet-flat-mean-generate-20260724` |
| `blt` | `models/blt/generic-blt.axon` | `test/BLT-Test` | CPU fully optimized generation top-1 equal, max abs `1.19e-6`; dense entropy patching, forced initial boundaries, max-length splits, hash embedding, local/global/local stack, and multi-layer cross-attention are exercised; validated-by `log/causal-blt-r7-20260724` |
| `bert-generation` | `models/bert_generation/generic-bert-generation.axon` | `test/BertGeneration-Test` | CPU fully optimized cached generation top-1 equal, max abs `1.79e-7`, with identical completions; validated-by `log/causal-bert-generation-20260724` |

### Remaining text-family queue

Reformer, XLNet, BitNet, BLT, and BERT-generation are closed. All other entries
are measured, covered by an exact architectural alias, task-specific, non-text,
or explicitly out of scope. No unresolved standalone text causal-LM family
remains in this mapping snapshot.

## Seq2seq Gap Ledger

Last mapping snapshot: Transformers `MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES`
on 2026-07-23.

### Existing coverage

| Transformers family | Axon coverage | Status |
|---|---|---|
| `bart` | `models/bart/generic-bart.axon` | existing |
| `marian` | `models/bart/generic-marian.axon` | existing |
| `mbart` | `models/bart/generic-mbart.axon` | existing |
| `mt5` | `models/mt5/generic-mt5.axon` | existing |
| `t5` | `models/t5/generic-t5.axon` | existing |
| `t5gemma`, `t5gemma2` | `models/t5gemma/` | existing |

### New text implementations

| Family | New Axon source | Evidence checkpoint | Current evidence |
|---|---|---|---|
| `blenderbot` | `models/blenderbot/generic-blenderbot.axon` | `facebook/blenderbot-400M-distill` | healthy: top-1 equal, max abs `1.19e-5`; `log/seq2seq-gaps-20260723-bart-lineage-rerun` |
| `blenderbot-small` | `models/blenderbot_small/generic-blenderbot-small.axon` | `facebook/blenderbot_small-90M` | healthy: top-1 equal, max abs `1.91e-5`; `log/seq2seq-gaps-20260723-bart-lineage` |
| `mvp` | `models/mvp/generic-mvp-seq2seq.axon` | `RUCAIBox/mvp` | healthy: top-1 equal, max abs `0`; prompt-disabled checkpoint path validated |
| `pegasus` | `models/pegasus/generic-pegasus.axon` | `google/pegasus-xsum` | healthy: top-1 equal, max abs `3.62e-5`; `log/seq2seq-gaps-20260723-batch2` |
| `m2m_100` | `models/m2m_100/generic-m2m-100.axon` | `facebook/m2m100_418M` | healthy with native tokenizer: top-1 equal, max abs `9.06e-6`; `log/seq2seq-gaps-20260723-native-tokenizers` |
| `fsmt` | `models/fsmt/generic-fsmt.axon` | `facebook/wmt19-en-de` | healthy: top-1 equal, max abs `1.24e-5`; `log/seq2seq-gaps-20260723-fsmt-led-rerun` |
| `led` | `models/led/generic-led.axon` | `allenai/led-base-16384` | healthy: top-1 equal, max abs `3.15e-5`; local/global attention is represented directly; `log/seq2seq-gaps-20260723-fsmt-led-rerun` |
| `longt5` | `models/longt5/generic-longt5-local.axon` | `saekomdalkom/long-t5-local-base-finetuned-xsum` | healthy real untied-embedding checkpoint: top-1 equal, max abs `6.41e-4`; `log/real-checkpoint-evidence-cpu-20260724-rerun8` |
| `switch_transformers` | `models/switch_transformers/generic-switch-transformers.axon` | `google/switch-base-8` | healthy real checkpoint: top-1 equal, max abs `5.72e-5`; learned relative bias is applied only in block 0, matching current Transformers; `log/seq2seq-switch-cpu-20260724-c` |
| `umt5` | `models/umt5/generic-umt5.axon` | `google/umt5-small` | healthy real checkpoint: top-1 equal, max abs `1.45e-3`; each block uses its own serialized relative-attention-bias parameter; `log/real-checkpoint-evidence-cpu-20260724-rerun7` |
| `plbart` | `models/plbart/generic-plbart.axon` | `uclanlp/plbart-base` | healthy after completing native tokenizer artifacts: top-1 equal, max abs `2.67e-5`; `log/seq2seq-gaps-20260723-plbart-rerun` |
| `bigbird_pegasus` | `models/bigbird_pegasus/generic-bigbird-pegasus.axon` | `google/bigbird-pegasus-large-arxiv` | healthy real checkpoint: full-attention max abs `1.93e-5`; a 768-token block-sparse probe has max abs `9.78e-6`; eval-time duplicate block-0 slots are represented by an equivalent score bias |
| `encoder-decoder` | `models/encoder_decoder/generic-bert2bert.axon` | Patrick CNN/DailyMail and both Google WMT directions | all three real BERT and BERT-generation compositions are top-1 equal, max abs `2.15e-5` to `2.57e-5`; config selects token-type embeddings and the appropriate decoder head; the legacy Google tokenizer requires an unpadded prompt because its generated pad ID exceeds the serialized vocabulary; `log/real-checkpoint-evidence-cpu-20260724-bert2bert-all-fix4` |
| `nllb-moe` | `models/nllb_moe/generic-nllb-moe.axon` | `test/NLLB-MoE-Test` | intended top-2 routing, Q/K/V, attention, and expert sums have direct op parity; end-to-end comparison is blocked by a current Transformers double-one-hot expert-dispatch regression |
| `pegasus_x` | `models/pegasus_x/generic-pegasus-x.axon` | `google/pegasus-x-base`, `google/pegasus-x-large` | both real sizes exercise global tokens, local blocks, and alternating stagger; top-1 equal with max abs `7.82e-5` and `1.91e-5`; `log/real-checkpoint-evidence-cpu-20260724-rerun2`, `log/real-checkpoint-evidence-cpu-20260724-seq2seq-variants` |
| `prophetnet` | `models/prophetnet/generic-prophetnet.axon` | `microsoft/prophetnet-large-uncased` | healthy real checkpoint: top-1 equal, masked max abs `1.57e-5`; the source reproduces both predicting streams and HF's joint predicting-relative-bias tensor layout; `log/seq2seq-prophetnet-cpu-20260724-m` |

### Remaining and unaccepted text families

| Family | Status | Concrete unresolved point |
|---|---|---|
| `nllb-moe` | implemented, reference-blocked | Current Transformers returns a one-hot top-1 mask from `NllbMoeTop2Router.route_tokens`, then applies `one_hot` to that mask again in `NllbMoeExperts.forward`. It consequently executes expert 1 for every top-1 route and expert 0 for every second route, irrespective of selected expert ID. CPU op-parity isolates the final mismatch to this upstream behavior: Axon positions, Q/K/V, attention, selected IDs/weights, and the intended top-2 weighted expert sum are exact. Encoding the erroneous dispatch in Axon would break the architecture and future corrected references. |

The text seq2seq mapping is therefore exhaustively classified as of
2026-07-24: existing/healthy families are listed above and NLLB-MoE has one
explicit upstream-reference blocker. Deferred audio/multimodal mappings remain
a separate tranche.

### Deferred non-text mappings

`audioflamingo3`, `glmasr`, `granite_speech`, `granite_speech_plus`,
`musicflamingo`, `qwen2_audio`, `seamless_m4t`, `seamless_m4t_v2`,
`vibevoice_asr`, `voxtral`, and `voxtral_realtime` are recorded in the
multimodal/audio tranche. They are not silently counted as text seq2seq
coverage.

## Multimodal / Audio / OCR Gap Ledger

Last mapping snapshot: installed Transformers on 2026-07-24.

### Validation-contract blocker

The standard fidelity harness currently accepts only `causal_lm`, `masked_lm`,
and `seq2seq_lm`. It constructs rank-2 text token IDs, text attention masks,
and textual generation outputs. This is enforced in
`brainsurgery/synapse/axon_test.py`; it does not construct or compare pixel
tensors, image-token grids, video frames, audio features/waveforms, protein
features, robot-action states, or waveform outputs.

Under this gap effort's change boundary, the harness and existing compiler
layers cannot be extended. Therefore a new specialized model source cannot
reach the required HF/Axon benchmark evidence, even when its operations are
expressible in Axon. Every entry below was checked against the current auto
mapping and is classified `blocked-validation-contract`, rather than silently
counted as covered or omitted. A future tranche must first add typed benchmark
contracts for these modalities as a separately reviewed generic change.

Existing text backbones such as `aria_text`, `cohere2`, `ernie4_5_moe`,
`gemma3`, `gemma3n_text`, `gemma4`, `lfm2`, `llama4`, `mistral3`,
`mistral4`, `qwen3_5`, and `t5gemma2` cover only their text components. They
do not constitute fidelity evidence for the corresponding multimodal wrapper.

### Image text-to-text

Transformers has 71 current entries; all require non-text inputs or
non-text output validation and are `blocked-validation-contract`:

`aria`, `aya_vision`, `blip`, `blip-2`, `chameleon`, `cohere2_vision`,
`deepseek_vl`, `deepseek_vl_hybrid`, `emu3`, `ernie4_5_vl_moe`, `evolla`,
`exaone4_5`, `fast_vlm`, `florence2`, `fuyu`, `gemma3`, `gemma3n`,
`gemma4`, `git`, `glm46v`, `glm4v`, `glm4v_moe`, `glm_ocr`, `glmga`,
`got_ocr2`, `granite4_vision`, `idefics`, `idefics2`, `idefics3`,
`instructblip`, `instructblipvideo`, `internvl`, `janus`, `kosmos-2`,
`kosmos-2.5`, `lfm2_vl`, `lighton_ocr`, `llama4`, `llava`, `llava_next`,
`llava_next_video`, `llava_onevision`, `minicpmv4_6`, `mistral3`,
`mistral4`, `mllama`, `ovis2`, `paddleocr_vl`, `paligemma`,
`perception_lm`, `pi0`, `pix2struct`, `pp_chart2table`, `pp_formulanet`,
`qianfan_ocr`, `qwen2_5_omni_thinker`, `qwen2_5_vl`, `qwen2_vl`,
`qwen3_5`, `qwen3_5_moe`, `qwen3_omni_moe_thinker`, `qwen3_vl`,
`qwen3_vl_moe`, `shieldgemma2`, `smolvlm`, `t5gemma2`, `udop`,
`video_llama_3`, `video_llava`, `vipllava`, and
`vision-encoder-decoder`.

OCR-specialized entries in this mapping (`florence2`, `glm_ocr`,
`got_ocr2`, `lighton_ocr`, `paddleocr_vl`, `pp_chart2table`,
`pp_formulanet`, and `qianfan_ocr`) share the same pixel-input blocker.

### Speech seq2seq

All 17 current entries require acoustic inputs and are
`blocked-validation-contract`:

`cohere_asr`, `dia`, `granite_speech`, `granite_speech_plus`,
`kyutai_speech_to_text`, `moonshine`, `moonshine_streaming`, `pop2piano`,
`seamless_m4t`, `seamless_m4t_v2`, `speech-encoder-decoder`,
`speech_to_text`, `speecht5`, `vibevoice_asr`, `voxtral`,
`voxtral_realtime`, and `whisper`.

### Text to waveform

All 11 current entries produce audio rather than token logits and are
`blocked-validation-contract`:

`bark`, `csm`, `fastspeech2_conformer_with_hifigan`, `higgs_audio_v2`,
`musicgen`, `musicgen_melody`, `qwen2_5_omni`, `qwen3_omni_moe`,
`seamless_m4t`, `seamless_m4t_v2`, and `vits`.

### Causal image modeling

`imagegpt` consumes quantized image-code sequences and is
`blocked-validation-contract`: the current text tokenizer/input builder cannot
construct its image vocabulary or compare reconstructed image outputs.

This completes classification of the current specialized-generation auto
mappings. Vision/audio classification, depth, segmentation, CTC, and x-vector
mappings are discriminative tasks outside the generation-focused goal of this
page; they are not claimed as covered.

## Completion Criteria

- Every masked-LM mapping entry is either linked to proven existing coverage,
  implemented and benchmarked, or blocked with a reproducible reason.
- Every newly implemented family has a real or test checkpoint, parameter
  evidence, benchmark log path, and fidelity result.
- No compiler/runtime/builtin or existing-model source file is changed.
- The causal, seq2seq, and multimodal queues are refreshed from upstream when
  their tranche starts.
