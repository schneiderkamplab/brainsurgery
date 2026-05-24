# TODO

- Add Axon syntax for scoped/absolute variables or buffers, likely `#xyz`, before caching reusable generated tensors such as RoPE tables as model buffers.
- After buffer syntax exists, evaluate moving reusable config-derived tables into backend `setup(...)` instead of recomputing them during `forward(...)`.
- Explore native distributed Axon inference as a future language/runtime direction. One possible design is a Jolie/process-calculus-style layer with typed services/processes, typed channels or ports, explicit `spawn`/`send`/`recv`/`await`/`select`, placement annotations, and graph IR nodes for communication, barriers, pipeline stages, and collectives. This should be semantic and visible in Axon/Graph IR if pursued, not hidden as backend-only magic.
- Separately evaluate backend-only tensor parallelism / DTensor-style support as a lower-risk near-term direction. This could shard parameters and tensors in codegen2/runtime2 using placement/layout metadata without adding process-calculus constructs to Axon. The key design question is what must be language-visible: tensor layouts, placement, and collectives may need typed Graph IR metadata even if the high-level Axon source stays sequential.
- Explore training/autograd support so Axon models can be used beyond inference. The near-term path should stay as close to backend-native training as possible: for PyTorch, codegen2 should emit a normal `nn.Module` with trainable tensors registered as `nn.Parameter`s, non-trainable tensors registered as buffers, PyTorch-compatible `state_dict()` / `load_state_dict()`, `_param(path)` returning the registered object, tied weights represented by shared parameter objects, and `forward(...)` using ordinary differentiable torch ops with `use_cache=False` by default for training. Graph IR should carry backend-neutral parameter metadata such as path, shape, dtype, trainable flag, alias group, and role (`weight`, `bias`, `buffer`, `cache`) so tinygrad can later expose explicit trainable tensors and JAX can expose a params/buffers PyTree without baking PyTorch semantics into Axon. Optional task/loss wrappers such as causal-LM cross entropy can be backend-specific initially. A later language-level design could add explicit parameter declarations, loss definitions, optimizer/state hooks, gradient checkpointing/rematerialization annotations, and validation that Graph IR ops have differentiable rules or approved stop-gradient boundaries.
- Build a first-class dtype and quantization roadmap for codegen2/runtime2. Current support is partial: float32/bfloat16/float16 are handled in benchmark and test-checkpoint paths, and MXFP4 has load-time materialization support. Missing work includes backend-neutral Graph IR dtype/quantization metadata, an explicit policy for load-time dequantization vs native low-bit execution, backend capability checks and clear fallback/error behavior, fidelity/performance tests for bf16/fp16, fine-grained FP8 support, MXFP4 coverage beyond current aliases, NVFP4 support, and synthetic packed-weight fixtures plus real-checkpoint coverage for each format.

## Further Optimization Ideas

These should stay optional until each pass has precise preconditions, typed validation, roundtrip coverage, and model-fidelity coverage. Prefer Graph IR for rewrites that need effect information, dataflow, shape metadata, or backend lowering knowledge; keep AST-level optimization limited to semantics-preserving canonical cleanup.

- Benchmark fidelity fix list from the 2026-05-24 full-opt HF sweeps:
  - `deepseekv2`: real DeepSeek-V2-Lite/Coder-V2-Lite rows have top-1 parity
    but large max-abs diffs (`~1.1..2.3`) in both generic and materialized
    Axon files. Treat this as a family fidelity issue, not a materialization
    mismatch.
  - `deepseekv4`: `test/DeepSeek-V4-Test` still has top-1 mismatch and large
    max-abs diff (`~0.35`). Fix the family/test fidelity before trusting the
    real Flash/Pro path.
  - `mt5`: mT5 rows have very large max-abs diffs, with `mt5-xl` also showing
    top-1 mismatch in the 4B..16B sweep. Generic and materialized quality
    match, so treat this as a model-family or shared seq2seq fidelity issue.
  - `olmo3`: both test and real OLMo3 rows show top-1 mismatches/large
    max-abs diffs. Generic and materialized quality match on real 7B rows, so
    treat this as a family fidelity issue.
  - `phi3small`: real Phi-3-small generic rows have large max-abs diffs while
    materialized rows are good. Resolve this as a generic-vs-materialized
    mismatch first, likely in generic Axon/materialization/config semantics,
    before changing backend behavior.
- Optimizer performance target ladder:
  - fast dense smoke: `openai-community/gpt2` and `google/gemma-3-270m`
  - inner MoE target: `allenai/OLMoE-1B-7B-0924`
  - main MoE target: `Qwen/Qwen3-30B-A3B`
  - dense sanity target: `google/gemma-3-4b-pt` or `Qwen/Qwen2.5-14B`
  - large stress targets: `deepseek-ai/DeepSeek-V4-Flash` and `openai/gpt-oss-20b`
  Use the inner MoE target for tight optimizer/codegen iteration, promote only
  validated speedups to the main MoE target, and reserve large stress targets for
  final regression/performance checks.
- High priority: lower tail-recursive loop-helper SCCs to explicit iterative
  graph/codegen loops. Detect the canonical flattened pattern with a loop index,
  loop bound/step, carry tuple, base return, and tail calls only in tail
  position. Emit a backend loop (`for range(...)` when the bound/step are
  statically normal, otherwise `while` with the validated loop-done predicate)
  while preserving path-template dependencies such as `@@'h.{i}'`.
- High priority: avoid materializing grouped-query / multi-query attention KV
  repeats. Models such as Qwen2.5 repeat K/V heads before attention; Graph IR and
  codegen2 should represent this as a view/expand-style operation or teach
  attention lowering to consume grouped K/V directly. This must stay generic over
  GQA/MQA patterns and should not introduce model-family branches. Current
  status: T5Gemma2, T5Gemma UL2, T5Gemma prefix-LM, and SmolLM3 have been
  migrated off explicit KV repeats and use `attention_gqa_with_additive_mask`.
  Future work: recognize the generic matmul/scale/additive-mask/softmax GQA
  subgraph in Graph IR/codegen and lower it to backend SDPA when all semantic
  guards hold, instead of exposing SDPA as an Axon primitive.
  The typecheck2 callee-dim capture and codegen2 inlined-temp/type-metadata
  collision exposed by those migrations have been fixed generically; remaining
  work is broader family coverage and performance tuning, not blocker work for
  these migrated rows.
- High priority: optimize the factorized RoPE apply path. After hoisting
  sin/cos creation out of layer loops, the remaining cost is repeated
  `rotate_half`, expand, multiply, and add work. Add a validated graph rewrite or
  backend-neutral fused op for common RoPE apply patterns, starting with the
  non-interleaved path used by Qwen2/OLMoE-style models.
- High priority: validate proportional-RoPE factor hoisting on real Gemma4
  checkpoints when GPU memory is available.
  `Positions.rope_proportional_factors` /
  `Positions.rope_pair_proportional_factors` now exist, and Gemma4 dense/E/MoE
  have been migrated to precompute local/full proportional RoPE factors once per
  forward. Test rows and graph/codegen dumps are clean. Dense 31B and MoE 26B
  still need clean real-checkpoint performance reruns on idle GPUs; an attempted
  MoE 26B run fell back to CPU after CUDA OOM on an occupied GPU.
- High priority: continue DeepSeek-family real-checkpoint reruns.
  DeepSeek v1 now precomputes base RoPE factors and causal masks and uses
  grouped selected-expert FFN on the test/materialized rows. DeepSeek v2 now
  precomputes its causal mask and exact YaRN RoPE factors on the tiny generic
  and materialized test rows. DeepSeek v3 now precomputes causal masks and both
  base/YaRN split-interleaved RoPE factors on the tiny generic and materialized
  test rows. Remaining work is clean real 16B/v2-lite/v3 reruns on idle GPUs.
  Keep changes model-file/generic-builtin only; avoid checkpoint-family routing
  or codegen special cases.
- High priority: add a generic fused selected-expert FFN path for MoE models.
  GPT-OSS 20B still spends most Axon time in the two `_expert_linear` calls even
  after RoPE/mask/GQA cleanup. The safe next step is a typed Graph IR/codegen
  pattern for selected-expert FFN:
  `expert_linear(gate_up) -> alpha/limit-aware gated activation -> expert_linear(down) -> scale/sum`.
  This must preserve configurable GPT-OSS `SWIGLU_ALPHA` semantics, avoid
  model-family branches, and validate on GPT-OSS plus existing direct
  `NN.expert_linear` users such as Mixtral and GraniteMoE. Also audit the current
  codegen2-torch expert-bank materialization helpers: GPT-OSS checkpoints
  already expose `gate_up_proj_*` aliases after MXFP4 materialization, so any
  extra `gate_proj/up_proj -> gate_up_proj` synthesis should be proven useful
  for other generic MoE layouts or moved to explicit load/config adaptation
  rather than staying as unexplained core codegen scaffolding.
- High priority: add an incremental/cache-aware Mamba selective-scan execution
  path. `SSM.causal_conv1d_full` now uses the generic `_conv1d` primitive, and
  `mamba_scan_full` is expressed as an Axon loop over `mamba_scan_step`. Decoder
  generation still recomputes a full sequence scan for each token. A correct
  fix should make recurrent SSM state explicit in Axon/Graph IR or add a
  validated backend-neutral scan/cache region, then codegen should execute one
  step per generated token. If the pure Axon scan loop does not scale well
  enough, treat selective scan as a future optimized-lowering target rather than
  reintroducing a permanent high-level `_mamba_scan` primitive. Avoid
  model-family branches; validate on `BlackMamba-2.8B`, `mamba-2.8b-hf`, and
  Jamba/Mamba2-style users separately.
- High priority: implement main-module-anchored intra- and inter-procedural
  domain analysis over the pruned reachable Graph IR. It should infer facts that
  hold on all non-dead paths from `MAIN`, including null/non-null, boolean,
  numeric literal/range, path/global-value, and callsite-restricted argument
  domains. Initial analysis-only support exists for unknown/null/non-null, exact
  literals, paths, and model-global values; next steps are wiring these facts
  into validated folding/dead-branch cleanup and extending to ranges.
- Graph-level shape/list literal cleanup: initial renderer-visible slice is
  implemented for atomic `core.list`/`core.tuple` expressions. The optimized
  graph already carries these structurally; Graph IR Axon rendering now keeps
  flat-safe atomic shape/order lists inline in primitive call arguments instead
  of introducing temporary list binds. Future work is broader single-use cleanup
  with explicit use-count checks if graph rewrites still materialize scaffolds.
- Graph-level scalar/dim expression simplification: continue expanding the
  initial implementation. It now simplifies typed dimension expressions such as
  `NUM_HEADS * (MODEL_DIM / NUM_HEADS)` to `MODEL_DIM` in term refs and
  type/dim metadata. Remaining work is to add only proof-gated identities that
  are valid under current integer dimension semantics and keep validating all
  updated term refs, type refs, dim metadata, and constraints.
- Graph-level return scaffolding cleanup: initial renderer-visible slice is
  implemented for atomic tuple/list graph expressions in module outputs. Existing
  graph cleanup removes return-only `core.tuple` producer nodes; Graph IR Axon
  rendering now keeps the resulting atomic tuple expression inline in `return`
  rather than reintroducing a temp.
- Graph-level optional/null specialization: continue specializing helpers with
  literal `null`/boolean arguments so existing branch cleanup can remove dead
  paths before inlining. Do not introduce branch-region or do-expression
  operands.
- AST-level safe cleanup: continue limiting `--optimize-safe` to rooted pruning, atomic alias cleanup, and literal-only folding. Candidate extensions are only small local rewrites that do not duplicate or remove potentially partial calls and can be re-typechecked to a fixpoint after every iteration.
- AST-level constraint folding: explore folding conditionals from fresh, provenance-aware constraints only after re-typecheck refreshes constraints. Avoid using stale constraints after any rewrite that changes operands, guards, signatures, or helper bodies.
- Graph-level dead branch cleanup: use Graph IR effects so total-pure dead
  computations can be removed while partial/impure operations are preserved.
  Branch cleanup should require literal conditions or validated constraints and
  must preserve eager Axon semantics outside ternary branches.
- Graph-level masking simplification: use domain facts and shape/cache facts to
  select simpler attention-mask paths when `attn_mask` is known null/non-null or
  when the causal-only path is provably sufficient. This should remove dead
  masking branches before codegen without changing Axon's eager semantics.
- Graph-level CSE: extend current total-pure CSE only if operand identity,
  attributes, path templates, dtype, shape, and constraints match. Unknown module
  calls should not be CSE candidates until effect metadata proves they are total
  and deterministic.
- Graph-level specialization and inlining: keep expanding only flat-safe cases.
  Specialize helpers when call-site constants or path templates materially
  improve types/codegen, but avoid unbounded cloning. Inlining should skip
  recursive SCCs, constrained helpers unless constraints are substituted
  correctly, and any helper containing partial effects unless proven safe.
- Graph-level shape/layout optimization: use preserved tensor shapes and path metadata to plan transposes, reshape/expand chains, layout choices, and backend-friendly argument forms without weakening Graph IR typing.
- Graph-level fusion and custom kernels: identify typed patterns such as RMSNorm, attention score/mask/softmax/value, SwiGLU/MLP, RoPE, and selected-expert MoE routing. These should rewrite to backend-neutral graph ops or annotated regions, not backend-specific model-name branches.
- Codegen2 overhead cleanup: reduce residual wrapper and parameter-lookup
  overhead for small hot operations such as RMSNorm and path-derived weight
  access. Prefer graph-validated inlining or generated-code local caching over
  ad-hoc backend special cases.
- Graph-level parameter/path normalization: continue canonicalizing path templates structurally, but do not inline constants or scopes by string rewriting. Template symbols are first-class dependencies and must remain visible to closure, validation, and codegen.
- Backend-aware but semantic-preserving lowering: add optional graph passes for pipeline partitioning, parameter placement, buffer hoisting, and dtype/dequant planning once Graph IR carries enough metadata to validate the transformed graph before code generation.
