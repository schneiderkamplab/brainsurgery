# TODO

- Add Axon syntax for scoped/absolute variables or buffers, likely `#xyz`, before caching reusable generated tensors such as RoPE tables as model buffers.
- After buffer syntax exists, evaluate moving reusable config-derived tables into backend `setup(...)` instead of recomputing them during `forward(...)`.
- Explore native distributed Axon inference as a future language/runtime direction. One possible design is a Jolie/process-calculus-style layer with typed services/processes, typed channels or ports, explicit `spawn`/`send`/`recv`/`await`/`select`, placement annotations, and graph IR nodes for communication, barriers, pipeline stages, and collectives. This should be semantic and visible in Axon/Graph IR if pursued, not hidden as backend-only magic.
- Separately evaluate backend-only tensor parallelism / DTensor-style support as a lower-risk near-term direction. This could shard parameters and tensors in codegen2/runtime2 using placement/layout metadata without adding process-calculus constructs to Axon. The key design question is what must be language-visible: tensor layouts, placement, and collectives may need typed Graph IR metadata even if the high-level Axon source stays sequential.
- Explore training/autograd support so Axon models can be used beyond inference. The near-term path should stay as close to backend-native training as possible: for PyTorch, codegen2 should emit a normal `nn.Module` with trainable tensors registered as `nn.Parameter`s, non-trainable tensors registered as buffers, PyTorch-compatible `state_dict()` / `load_state_dict()`, `_param(path)` returning the registered object, tied weights represented by shared parameter objects, and `forward(...)` using ordinary differentiable torch ops with `use_cache=False` by default for training. Graph IR should carry backend-neutral parameter metadata such as path, shape, dtype, trainable flag, alias group, and role (`weight`, `bias`, `buffer`, `cache`) so tinygrad can later expose explicit trainable tensors and JAX can expose a params/buffers PyTree without baking PyTorch semantics into Axon. Optional task/loss wrappers such as causal-LM cross entropy can be backend-specific initially. A later language-level design could add explicit parameter declarations, loss definitions, optimizer/state hooks, gradient checkpointing/rematerialization annotations, and validation that Graph IR ops have differentiable rules or approved stop-gradient boundaries.
- Build a first-class dtype and quantization roadmap for codegen2/runtime2. Current support is partial: float32/bfloat16/float16 are handled in benchmark and test-checkpoint paths, and MXFP4 has load-time materialization support. Missing work includes backend-neutral Graph IR dtype/quantization metadata, an explicit policy for load-time dequantization vs native low-bit execution, backend capability checks and clear fallback/error behavior, fidelity/performance tests for bf16/fp16, fine-grained FP8 support, MXFP4 coverage beyond current aliases, NVFP4 support, and synthetic packed-weight fixtures plus real-checkpoint coverage for each format.

## Further Optimization Ideas

These should stay optional until each pass has precise preconditions, typed validation, roundtrip coverage, and model-fidelity coverage. Prefer Graph IR for rewrites that need effect information, dataflow, shape metadata, or backend lowering knowledge; keep AST-level optimization limited to semantics-preserving canonical cleanup.

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
  GQA/MQA patterns and should not introduce model-family branches.
- High priority: optimize the factorized RoPE apply path. After hoisting
  sin/cos creation out of layer loops, the remaining cost is repeated
  `rotate_half`, expand, multiply, and add work. Add a validated graph rewrite or
  backend-neutral fused op for common RoPE apply patterns, starting with the
  non-interleaved path used by Qwen2/OLMoE-style models.
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
