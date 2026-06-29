# Axon Optimize Refactor Plan

This document classifies the current Axon optimizer pieces by semantic safety and
defines the hardening path for optional AST and Graph IR optimization.

The current rule should be:

- `optimize-ast` and `optimize-graph` are optional and off by default
- safe local rewrites may become a small always-available conservative pass
- speculative rewrites stay disabled until they have explicit semantic preconditions,
  validators, and fidelity coverage
- anything required by Graph IR/codegen should be represented as a stage contract,
  not hidden inside optional optimization

## Current Status

Top optimizer strengthening priorities:

- Eliminate stale shape symbols in Graph IR by construction. This is the top
  blocker before re-enabling broader shape folding and helper inlining. After
  every graph rewrite, tensor `type_expr`, `dims`, node result types, module
  return types, and constraints must be re-instantiated from the producing op and
  callsite actuals, then validated against runtime-shape-sensitive uses. In
  particular, inlining helpers such as `Attention.merge_heads` must preserve the
  fact that `Tensor.size`/`_shape` queries are authoritative for dynamic
  generation shapes; prompt-time symbols such as `S`/`K` must not be reused for a
  one-token decode-step tensor unless the op/type rule proves they are equal.
  The current implementation keeps a conservative safety barrier: multi-node
  helpers containing runtime shape queries are not inlined, and graph-level
  shape-query folding only applies to stable module inputs. Direct primitive
  `_tensor_size`, `_shape`/`_list_index`, and structural one-node forwarders to
  those primitives are folded only under that stable-shape guard. This should be
  relaxed only after graph validation can prove shape metadata freshness
  end-to-end.
- Lower flattened tail-recursive loop-helper SCCs to explicit iterative
  graph/codegen loops. The recognizer should require a single-entry SCC, a loop
  index/bound/step state, tail calls only in tail position, and a carry tuple
  whose arity is stable across the base and recursive paths. Codegen can then
  emit `for range(...)` for normal static positive-step loops and a validated
  `while` form otherwise, preserving template-symbol dependencies such as
  `@@'h.{i}'`.
- Add a main-module-anchored intra- and inter-procedural domain analysis. The
  analysis should run over the pruned reachable graph and derive facts that hold
  on all non-dead paths from `MAIN`, including null/non-null facts, boolean
  values, numeric literal/range equalities, path/global-value equalities, and
  callsite-restricted argument domains. These facts should feed constant folding,
  dead-branch cleanup, specialization, and inlining without relying on syntactic
  guesses. Example: if every reachable call to `Positions.position_ids` passes
  `attn_mask=null`, the callee should know `attn_mask is_null`, fold
  `attn_mask == null` to `true`, and remove the masked branch before any clone or
  inline rewrite is accepted.

  Implementation plan:

  - First land an analysis-only fact domain: unknown, null, non-null, exact
    literal, exact path, and exact model-global value.
  - Infer facts only from the `MAIN`-reachable graph. Unreachable definitions do
    not contribute.
  - Aggregate callsite facts per callee formal by intersection: if all reachable
    callsites pass the same null/literal/path/global value, record that fact;
    disagreements become unknown.
  - Propagate facts intraprocedurally through aliases/ascriptions, literal
    selects, and simple equality/inequality comparisons.
  - Only after Graph IR validation is strong enough, feed the facts into
    fold/dead-branch cleanup and then into specialization/inlining.
  - Iterate fact derivation and cleanup to a validated graph fixpoint.
  - Use branch facts to unlock safe inlining of helpers called from lazy select
    branches. The inliner now threads branch-local domain facts through nested
    `core.select` operands, so a call in the false branch of `values == null`
    sees `values` as non-null without relaxing optional-to-nonoptional call
    compatibility globally. Flatten now also preserves branch-refined optional
    parameter types for extracted condition helpers: only names proven non-null
    by the enclosing branch are stripped from optional form. Remaining metadata
    cleanup should carry outer branch refinements through nested select
    expressions during graph type refresh.
- Inline single-use total-pure list/tuple literals when they are only argument
  scaffolding for flat-safe primitive calls such as `_reshape`, `_expand`, and
  `_permute`.
- Continue simplifying scalar/dimension expressions in Graph IR. The first slice
  reduces direct algebraic identities such as `NUM_HEADS * (MODEL_DIM /
  NUM_HEADS)` to `MODEL_DIM` and updates term/type/dim metadata coherently.
  Future identities must remain proof-gated and validator-backed.
- Remove remaining return-only tuple scaffolding when the tuple expression is
  flat-safe, typed, used exactly once, and does not duplicate partial/effectful
  work.
- Continue flat-safe optional/null specialization and inlining. Do not add
  branch-region or do-expression operands; helpers containing live lazy branches
  should remain as calls unless a flat-safe branch is eliminated first.
- Keep path/template symbols and model-global bindings first-class through all
  rewrites; do not turn them into string substitutions or definition aliases.
- Backend-specific intrinsics are opt-in only. Graph optimization may introduce
  names of the form `__<backend>_<name>` only when the caller explicitly selects
  a compatible backend target, and backend-neutral optimization must never emit
  them. The SDPA intrinsics `__torch_sdpa` and `__tinygrad_sdpa` are derived from
  provenance facts for attention subgraphs plus additive-mask provenance. These
  rewrites currently assume the attention keep mask has no fully masked rows; the
  future hardening target is a provenance/domain proof for nonempty keep rows
  before the rewrite can be made generally safe.
- Backend-specific intrinsic rewrites must not infer semantics from ordinary
  Axon definition names. Definition names may identify a callee for graph
  traversal, but the semantic proof must come from the callee body provenance or
  from already-inlined primitive provenance. The current Torch RoPE intrinsic
  follows this rule: it derives a `rope_apply_factors` provenance fact from the
  primitive DAG `x*cos + rotate_half(x)*sin`, then rewrites eligible one-output
  or pair-output callsites to `__torch_rope_apply_factors` /
  `__torch_rope_pair_apply_factors` only when `--graph-backend-intrinsics
  codegen2-torch` is selected.
- Fill/scatter unit-slice is a backend-neutral graph rewrite: it rewrites to
  `_assign_slice(base, src, dim, index, index + 1)` when output provenance proves a primitive
  `_scatter(base, index, src, dim)` whose `index` provenance is primitive
  `_fill(src, value, ...)`. This works through ordinary Axon wrappers without
  treating wrapper names as semantic evidence.
- Dense gate/up linear pairing is a backend-neutral graph rewrite: adjacent
  same-input primitive `_linear` computations become one normal `_linear`
  against an explicit packed-parameter graph metadata entry, followed by
  `_chunk`. The rewrite is rejected unless both outputs are type-compatible,
  both linears are non-expert and unbiased, and the exact gate/up source weight
  paths have no other semantic parameter reads in the reachable graph, including
  non-linear contexts such as `Params.param`. Codegen materializes the packed
  weight from the explicit graph metadata, removes the original source tensors
  from runtime state, and lowers only the resulting normal `_linear`/`_chunk`.
  The runtime-side materialization for dense packed pairs, expert banks, and
  fused expert gate/up banks now goes through one generic parameter-join helper
  (`cat`/`stack` plus optional source removal), even where the source of the
  join is still path-derived backend storage metadata rather than a graph
  rewrite.

## Codegen Lowering Tricks To Move Toward Graph Rewrites

The active migration backlog is limited to patterns that are already useful,
not deselected, and not already implemented as graph intrinsics.  Backend
intrinsics should still require provenance or primitive-level proof; ordinary
Axon definition names are not semantic evidence.

- Static shape-query lowering: implemented as a graph-level rewrite for direct
  primitive `_tensor_size`, primitive `_shape` followed by primitive
  `_list_index`, and structural one-node forwarders to those primitives. It is
  deliberately stable-shape guarded: module-input tensor dimensions may be
  folded, but value-dependent intermediate tensors such as `_where_indices`
  outputs must keep runtime shape queries unless a later freshness proof says
  otherwise. If the graph rewrite does not prove a static replacement, codegen
  emits a real runtime shape query; it no longer has a second static
  tensor-size inference path.
- Single-primitive forwarder detection: graph optimization owns this. Structural
  one-node forwarders are inlined by Graph IR before codegen, including argument
  reordering and multi-output destructuring cases. Codegen no longer treats
  ordinary module names as primitive-like; it only lowers real primitives and
  explicit backend intrinsics.
- Fill/scatter unit-slice lowering: implemented as a provenance-backed
  backend-neutral `_assign_slice` graph op. The older codegen syntactic
  peepholes for this pattern have been removed, so the optimization is owned by
  Graph IR.
- List append/list construction: `_list_init`, `_list_append`, `_list_index`,
  and `_list_length` are primitive lowerings. Codegen must not infer cache
  semantics from normal Axon definitions such as `Cache.append` or
  `Cache.past_length`; those definitions should be optimized only by generic
  graph inlining/domain facts. `_list_append` currently preserves value
  semantics by returning a new Python list expression. A future graph-level
  cache representation pass may introduce an explicit affine/in-place cache
  update intrinsic when usage analysis proves it safe.
- Static parameter-key fast paths for embedding/layernorm/linear: this remains
  a backend storage decision, not a graph rewrite. Graph IR already carries
  `GraphPath` operands; codegen may bypass generic path composition only when
  that path metadata proves an untemplated absolute key. Templated paths must
  keep dynamic path rendering. Tests cover both cases.
- RMSNorm/LayerNorm algebraic expansions: not currently actionable as a graph
  rewrite for the migrated model set. Normalization is already represented by
  `_rmsnorm`, `_layernorm`, or `_l2norm` primitives in the builtins/models that
  matter. Revisit only if a model intentionally expresses normalization as
  primitive arithmetic and profiling shows backend-native norm lowering is
  missed.
- Dense SwiGLU/gated-MLP gate/up projection: implemented as a backend-neutral
  packed-parameter `_linear` plus `_chunk` rewrite when provenance proves
  adjacent same-input non-expert `_linear` computations. The full dense
  `silu(gate) * up -> down`
  region is also implemented for Torch as `__torch_swiglu_ffn` when provenance
  proves the gate/up pair, `_activations_silu`, multiply, and unbiased
  non-expert down `_linear` chain, and when the fused gate/up source weights
  have no other semantic parameter reads in the reachable graph.
- MoE SwiGLU/gated-MLP blocks: the straight-line expert pattern is implemented
  for Torch as `__torch_expert_swiglu_ffn` when provenance proves
  same-input/same-expert-index unbiased, non-transposed `_expert_linear` gate
  and up projections, `_activations_silu(gate) * up`, and an unbiased,
  non-transposed `_expert_linear` down projection. This works through ordinary
  wrappers because the proof is primitive provenance, not definition names.
  The packed gate/up variant used by `grouped_swiglu_ffn_basic` is implemented
  as `__torch_expert_packed_swiglu_ffn` when provenance proves one unbiased
  `_expert_linear`, `_chunk(..., parts=2)`, `_activations_silu`, multiply, and
  an unbiased down `_expert_linear` with the same expert-index and transpose
  provenance. Remaining variants with clamps or separate grouped routing should
  become additional provenance-backed rewrites only after a profiled slow family
  motivates them.
- Expanded packed GeGELU activation should not be canonicalized back to
  `_activations_gegelu` by default. `ActivationsBasic.axon` intentionally shows
  the internals of activation formulas, and the optimizer should not erase that
  structure without a concrete backend/runtime performance reason.
- Top-k MoE routing plus grouped expert matmul: a provenance-backed
  `__torch_grouped_moe` intrinsic is high-value for MoE families, provided the
  match is derived from primitive dataflow and routing semantics rather than
  model or module names.
  A small backend-specific reduction slice is implemented as
  `__torch_weighted_topk_sum`: provenance must prove
  `sum(values * unsqueeze(topk_scores, -1), dim=2, keepdim=false)` with matching
  `[B,T,TOPK,D]` / `[B,T,TOPK]` / `[B,T,D]` tensor metadata. This intentionally
  does not infer router semantics; it only collapses the final weighted expert
  reduction common to grouped MoE blocks.
  The selected-weight normalization slice is implemented as
  `__torch_topk_normalize`: provenance must prove `_topk` weights followed by
  `cumsum(weights, -1)`, slicing the last running sum, division by that
  denominator, and `cast_like`. The rewrite leaves the `_topk` indices path
  unchanged.
  The first selected-expert loop slice is implemented for packed SwiGLU experts
  as `__torch_selected_expert_packed_swiglu_ffn`: provenance must prove a
  `core.repeat` expert loop whose body selects tokens via primitive
  `_where_indices` over top-k indices, computes an unbiased packed gate/up
  `_linear -> _split -> _activations_silu -> multiply -> _linear` expert body,
  and scatters back with primitive `_index_add` weighted by the corresponding
  top-k scores. The rewrite removes the expert-id segment from the proven
  templated expert weight paths to request banked expert weights from codegen.
  A direct grouped selected-expert packed SwiGLU form is also implemented:
  provenance must prove `unsqueeze/expand(hidden) -> _expert_linear(gate_up) ->
  chunk -> silu(gate) * up -> _expert_linear(down) -> weighted sum`, either
  directly in the caller or through an ordinary callee body. The proof still
  comes from primitive provenance, not from the callee definition name.
  A direct grouped selected-expert separate-gate/up SwiGLU form is implemented
  as `__torch_selected_expert_swiglu_ffn`: provenance must prove the same
  selected-token expansion and weighted sum, but with separate unbiased
  `_expert_linear(gate)` and `_expert_linear(up)` paths feeding
  `_activations_silu(gate) * up` before the down expert linear.
  A second selected-expert loop slice is implemented for ReLU2 two-linear
  experts as `__torch_selected_expert_relu2_ffn`: provenance must prove the same
  select/scatter loop shape plus an unbiased `_linear -> _activations_relu2 ->
  _linear` expert body.
  A third selected-expert slice is implemented for GPT-OSS-style packed GeGLU
  experts as `__torch_selected_expert_packed_gegelu_ffn`: provenance must prove
  either the select/scatter loop shape or the direct grouped top-k-axis shape, a
  packed gate/up expert linear, an alpha/limit-aware GeGLU activation on that
  packed result, a down expert linear with matching expert index, matching
  transpose, and matching bias setting. Bias paths are used only when the proven
  expert-linear provenance says bias is enabled; the direct form carries the
  explicit `alpha` operand rather than assuming GPT-OSS defaults.
  A fourth selected-expert loop slice is implemented for DeepSeek-style clamped
  packed SwiGLU experts as `__torch_selected_expert_clamped_packed_swiglu_ffn`:
  provenance must prove the same select/scatter loop shape, one unbiased packed
  gate/up expert `_linear`, chunk into gate/up halves, matching finite clamp
  limits, `_activations_silu(gate) * up`, and an unbiased down `_linear`.
- Causal Conv1D state/update blocks: a provenance-backed
  `__torch_causal_conv1d_step` or related block intrinsic is a candidate for
  Mamba/Jamba-style models if pure Axon lowering remains a bottleneck.
- Backend intrinsics already implemented in graph optimize:
  `__torch_sdpa`, `__tinygrad_sdpa`, `__torch_rope_apply_factors`, and
  `__torch_rope_pair_apply_factors`, `__torch_swiglu_ffn`, and
  `__torch_expert_swiglu_ffn`, `__torch_expert_packed_swiglu_ffn`,
  `__torch_weighted_topk_sum`, `__torch_topk_normalize`, and
  `__torch_selected_expert_packed_swiglu_ffn`,
  `__torch_selected_expert_swiglu_ffn`,
  `__torch_selected_expert_relu2_ffn`.
  These are the target style for future backend-specific fusions.
- Backend-neutral graph rewrites already implemented include `_assign_slice`
  and dense gate/up packed-parameter `_linear` plus `_chunk`.
- Path-derived expert-bank handling for `_expert_linear` is intentionally not
  in this graph-rewrite backlog for now. It is primarily a backend storage and
  state-dict layout decision. Graph IR should keep precise path/expert metadata;
  codegen may choose banked parameter access. Revisit only if cross-backend
  scheduling, memory planning, or custom kernel selection needs the banked form
  before lowering.

`optimize_flat_typed_axon_file` currently mixes several categories:

- required backend-shape normalization
- rooted dead-definition pruning
- atomic alias cleanup
- literal/local constant folding
- temp elimination
- parameter pruning
- path/template rewrites
- specialization
- inlining
- repeated re-typecheck/re-validate fixpoint orchestration

That mixture is the main problem. The passes do not all have the same semantic risk.
Some are safe as typed AST rewrites. Some are only safe under narrow preconditions.
Some are better done on typed Graph IR instead of Axon AST.

Implementation status:

- `optimize_safe_flat_typed_axon_file` exists as an optional conservative pre-Graph
  optimizer. It runs rooted pruning, atomic alias cleanup, and literal/atomic-only
  folding to fixpoint with `typecheck2` and typed validation after each iteration.
- `optimize_graph_program` exists as an optional typed Graph IR optimizer. It
  currently performs graph pruning, atomic alias cleanup, dead temp elimination for
  total-pure nodes, literal-only constant folding, proof-gated literal dim
  substitution, total-pure common-subexpression elimination, conservative
  single-callsite specialization, and constrained safe inlining of total-pure
  helper modules. It iterates to a fixpoint with a high iteration limit and now
  fails loudly instead of returning a non-converged graph.
- Graph IR pruning is metadata-aware: type annotations, dim metadata, constraints,
  and path-template symbols are treated as dependencies. This prevents pruning
  constants such as `CFG`/shape names that only appear in templates or typed
  metadata.
- Literal zero-arg atomic constants are now inlined by default. If a dimension,
  int, or float value is statically proven literal, Graph IR should carry and
  render the literal rather than preserving a symbolic alias for readability.
- Graph constant-dim substitution is enabled by default through
  `GraphOptimizeConfig.constant_dim_substitution=True`. It is still local,
  proof-gated, and validates each candidate rewrite. For metadata substitution,
  a module must both reference the constant in dim/type metadata and carry a
  local equality constraint such as `VOCAB_SIZE = 151936` before substitution can
  apply.
- Graph common-subexpression elimination is enabled by default through
  `GraphOptimizeConfig.common_subexpression_elimination=True`. The initial
  implementation eliminates duplicate single-output nodes and duplicate nested
  expression operands proven `total_pure`; partial-pure nodes/expressions are not
  CSE'd because Axon evaluation is eager and duplicated partial failures are
  observable enough that the optimizer should not change them without a stronger
  semantic rule.
- Graph literal folding now includes a narrow scalar primitive table. `_sqrt`
  folds when its argument is a non-negative int/float literal, so derived
  arithmetic such as `1.0 / _sqrt(64)` can collapse through the existing binary
  folding path. Tensor-valued primitives, config/param calls, and partial-pure
  operations are not folded by this table.
- Graph symbolic dim simplification now folds simple algebraic identities in
  both metadata and local scalar-dim dataflow. For example, a producer sequence
  equivalent to `hd = MODEL_DIM / NUM_HEADS; d = NUM_HEADS * hd` renders the
  downstream shape as `MODEL_DIM`. The pass is structural and validator-backed;
  it does not use string rewriting.
- Graph-to-Axon rendering now preserves optimized atomic `core.list`/`core.tuple`
  expressions inline when doing so is flat-safe. This exposes graph cleanup for
  shape/order arguments such as `_reshape x [B,S,D]` and return tuple expressions
  without allowing nested eager calls in argument positions.
- Graph specialization now also handles call-site constants that flow through
  nested `GraphExpr` calls and through local zero-arg global-value producer
  temps. This lets helper calls such as `Masking.causal_mask_keep(...,
  CONTEXT_SIZE)` specialize even when they sit inside lazy ternary operands or
  are reached through another specialized helper. Null comparisons are folded
  for substituted global values with statically non-null types, but not for
  arbitrary local refs whose annotations may be over-refined.
- Graph tuple-return cleanup now removes multi-output `core.tuple` destructuring
  nodes when they only repackage atomic values or nested tuple repackaging
  expressions. This eliminates scaffolding such as `__cond_result_2,
  __cond_result_3, __cond_result_4 <- (k_all, v_all, (k_all, v_all))` while
  preserving effect order and avoiding duplication of partial/effectful work.
- Graph substitution now uses one shared utility for operand refs, path-template
  refs, type/dim metadata, constraints, and value renaming. Specialization uses
  this shared path to keep helper signatures, return types, outputs, constraints,
  and body annotations coherent when literal dim formals are removed. Safe
  inlining also substitutes call-site type/dim bindings through copied helper
  nodes and returned operands, so rendered graph Axon does not keep stale callee
  symbols such as `d` when the call site has already fixed that dimension.
- Graph domain analysis has an initial analysis-only implementation. It derives
  main-reachable callsite facts for module formals and local facts through
  aliases/ascriptions, literal selects, and simple equality comparisons. It is
  not yet wired into graph mutation; this is intentional until Graph IR
  validation and optimize-graph idempotence are stronger.
- Graph final naming now canonicalizes specialized module names and generated
  local value names after the semantic optimizer reaches a fixpoint. The value
  pass rewrites node outputs, value refs, path-template placeholders, type/dim
  metadata, constraints, and return operands consistently, while preserving
  source-readable names such as `logits`, `mask`, `scores`, or `new_kv`.
- Graph optimization now promotes zero-arg, single-output modules proven
  `total_pure` to model-global bindings before inlining. This keeps config-like
  values such as `MODEL_DIM` and `NUM_HEADS` as explicit global values that
  codegen2 can evaluate once and cache, rather than expanding their bodies at
  every use site.
- Graph-to-Axon rendering now treats zero-arg graph modules as callable values
  when they occur in term positions. Repeated local zero-arg refs are canonicalized
  through the first local binding, which keeps graph-rendered Axon typecheckable
  without relying on downstream normalize to reinterpret bare names.
- Constraint-aware specialization is implemented for the narrow safe case:
  literal/path actuals are substituted into helper constraints, trivially true
  constraints are pruned, false or unrepresentable substitutions reject the clone,
  and unguarded stale constraint metadata is dropped conservatively. Primitive
  wrapper modules are not specialized, because their apparent signatures can be
  broader than the op-specific primitive type rule.
- Graph IR validation now includes a conservative type/arity verifier for typed
  operands, module calls, core ops, stale value references, and node/module output
  contracts. It intentionally accepts current Axon polymorphism such as `Any`,
  type variables, symbolic dimension binders, row variables, optional-null
  defaults, Dim/Int coercions, and Int-to-Float widening.
- The graph optimizer now validates at pass boundaries. Each mutating subpass
  checks the resulting Graph IR for structural validity and stale metadata before
  the next subpass runs. The metadata check is intentionally strict for concrete
  numeric shape mismatches, but tolerant of current row-variable summaries such
  as `..S`/`..R` until upstream type metadata canonicalization is stronger.
- Optimizer validation also checks module-call result contracts after applying
  call-site dimension substitutions. A helper returning `Tensor[B,d]` and called
  with `d=8` must produce a call result compatible with `Tensor[B,8]`; stale
  concrete annotations such as `Tensor[B,9]` are rejected before any rewrite.
- Graph IR has an explicit effect model. Unknown calls are `partial_pure` by
  default; only known-total `core.*` ops are currently `total_pure`. Graph DCE and
  safe inlining consult this model instead of a local ad-hoc allowlist.
- Smoke validation:
  `generic-llama4.axon` with checkpoint `test/Llama4-Test`, backend
  `codegen2-torch`, dtype `float32`, device `cuda:0`, max length 16 passed Graph
  IR validation and codegen execution. Result: `masked_top1_eq=True`,
  `masked_max_abs_diff=0.010181993246078491`. Log:
  `log/smoke-llama4-graph-validate-20260511c`.
- Both are off by default. CLI switches are `--optimize-ast` and
  `--optimize-graph`.
- Broad graph-optimized weak and strong roundtrip coverage currently passes:
  `tests/test_synapse_axon_graph_ir_roundtrip.py::{test_graph_optimized_graph_ir_weak_roundtrip_is_canonical,test_graph_optimized_graph_ir_strong_roundtrip_is_canonical}`
  produced `512 passed` on 2026-05-12 after fixing the graph closure,
  path-template substitution, atomic-constant inlining, and loop-helper return
  type propagation clusters.
- A max-4B `--optimize-graph` fidelity run on 2026-05-13 completed 153/153 rows
  with zero `ERROR` rows and zero top-1 failures. Six rows exceeded the strict
  `1e-3` max-abs threshold while preserving top-1: `Apertus-Test`,
  `GPT-OSS-Test`, `Llama4-Test`, `Mistral4-Test`, `mt5-small`, and `mt5-base`.
- A longer-prompt DeepSeek-V4 smoke on 2026-05-13 covered the non-empty CSA path
  with `masked_top1_eq=True` and `masked_max_abs_diff=6.56e-7`.
- A 10-row diverse max-4B `--optimize-graph` fidelity smoke on 2026-05-14 passed
  before and after nested-expression CSE. All rows had `masked_top1_eq=True`;
  known top-1-preserving strict-threshold outliers remained `GPT-OSS-Test`,
  `Llama4-Test`, and `mt5-small`.
- A 6-file optimized Graph IR weak+strong roundtrip smoke on 2026-05-14 passed
  after inlining metadata substitution: GPT-2 KV, BERT, T5, Gemma3, Llama4, and
  DeepSeek-V4.
- A 20-file optimized Graph IR weak+strong roundtrip smoke on 2026-05-14 passed
  after pass-level graph optimizer validation was added: `40/40` canonical
  roundtrips. Artifact directory:
  `tmp/graph-roundtrip-phase-validate-selected20-20260514b`.
- A 10-row diverse max-4B `--optimize-graph` fidelity smoke on 2026-05-14 passed
  after pass-level validation: `10/10` completed, zero errors, zero top-1
  failures. The same known top-1-preserving strict-threshold outliers remained:
  `GPT-OSS-Test`, `Llama4-Test`, and `mt5-small`. Log:
  `log/opt-graph-phase-validate-fidelity10-20260514`.
- A 20-file optimized Graph IR weak+strong roundtrip smoke on 2026-05-14 passed
  after call-result validation was added: weak `20/20`, strong `20/20`.
  Artifact directory:
  `tmp/graph-roundtrip-call-result-validate-selected20-20260514`.
- A 10-row diverse max-4B `--optimize-graph` fidelity smoke on 2026-05-14 passed
  after call-result validation: `10/10` completed, zero errors, zero top-1
  failures. The same known top-1-preserving strict-threshold outliers remained:
  `GPT-OSS-Test`, `Llama4-Test`, and `mt5-small`. Log:
  `log/opt-graph-call-result-validate-fidelity10-20260514`.
- A 20-file combined optimized AST+Graph IR weak+strong roundtrip smoke on
  2026-05-14 passed after constraint-aware specialization: weak `20/20`, strong
  `20/20`. Artifact directory:
  `tmp/graph-roundtrip-constraint-specialize-selected20-20260514`.
- A 10-row diverse max-4B `--optimize-graph` fidelity smoke on 2026-05-14 passed
  after constraint-aware specialization: `10/10` completed, zero errors, zero
  top-1 failures. The same known top-1-preserving strict-threshold outliers
  remained: `GPT-OSS-Test`, `Llama4-Test`, and `mt5-small`. Log:
  `log/opt-graph-constraint-specialize-fidelity10-20260514`.
- The old broad AST optimizer implementation remains as internal experimental
  code, but the user-facing `--optimize` flag has been removed.
- There is still no full Graph IR inference engine. Graph optimization relies on
  typed Graph IR input, type-preserving rewrites, the graph verifier, and
  conservative effect defaults.

## Removed Backend-Required Normalize

The old backend-required normalize stage has been removed from the active
codegen2/Graph IR pipeline. Graph IR supports multi-output nodes natively, so
list destructuring is represented structurally instead of being rewritten into
backend-specific `_list_index` scaffolding.

The removed pass used to rewrite:

```axon
x, y, z <- Tensor.chunk value parts=3
```

becomes roughly:

```axon
tmp <- Tensor.chunk value parts=3
x <- _list_index tmp 0
y <- _list_index tmp 1
z <- _list_index tmp 2
```

Current decision:

- keep no legacy backend-required stage just for old backend constraints
- keep no `validate.backend_required` stage
- if a backend cannot consume multi-output nodes, that backend should lower or split
  the graph IR explicitly, not force all Axon lowering through `_list_index`

## Safe Passes

These are safe candidates for a conservative pre-Graph-IR optimizer, provided they
keep retyping and validation after each iteration.

### Rooted Unreachable Definition Pruning

This is safe when it uses the shared resolve/typecheck reachability logic and an
explicit selected main module.

Rules:

- use `resolve_main_module`
- use shared `prune_unreachable_definitions`
- never rely on an implicit "last module" default after main selection is known
- preserve pragmas and type aliases
- validate closed/flat/typed shape after pruning

This should not live as optimizer-specific call-graph code.

### Atomic Alias Cleanup

Safe:

```axon
y <- x
return y
```

to:

```axon
return x
```

Safe atomic expressions are names, scalar literals, `null`, strings, paths, and
containers whose elements are all atomic.

Required constraints:

- do not duplicate non-atomic expressions
- do not cross rebinding of the same name
- preserve return arity and typed metadata
- re-typecheck after rewriting

### Literal-Only Constant Folding

Safe when both sides are genuinely literal or the rewrite is an algebraic identity
whose behavior is independent of evaluation effects.

Safe examples:

```axon
x <- 1 + 2
y <- true and flag
z <- null == null
```

Unsafe unless proven:

```axon
x <- f() == f()
y <- null == maybe_partial()
```

Current implementation should be treated as safe only for:

- int/int arithmetic and comparisons with literal operands
- float/float arithmetic with literal operands
- bool/bool operations with literal operands
- `true or x`, `false and x`, `x and true`, `x or false` only if they do not
  introduce or remove evaluation of partial expressions incorrectly
- `x == x` only when `x` is atomic and already evaluated, not an arbitrary call
- null-vs-non-null folding only when non-null is derived from a typed value that
  is already evaluated, not by deleting a partial expression

Hardening requirement:

- the folding helper should expose a predicate like `is_safe_fold_operand`
- every fold case should document whether it is literal-only, atomic-only, or
  type-derived
- tests should include rejected folds for calls/config/param lookups

### Retype And Validate

Safe and required.

Every optimization iteration should:

- validate input invariants
- apply one or more monotonic rewrites
- re-run typecheck
- validate typed output
- stop on structural fixpoint

If constraints are consumed, they must be refreshed by re-typecheck before the next
constraint-consuming pass.

## Unsafe Or Experimental Passes

These should stay out of default optimization until hardened.

### `_atomicize_call_args_statements`

Purpose:

- make complex call arguments into explicit temporary binds
- produce a shape that is easier for downstream codegen/runtime
- example intended rewrite:

```axon
y <- f (g x)
```

to:

```axon
tmp <- g x
y <- f tmp
```

Problem:

Axon is eager except for ternary branches. Hoisting out of a ternary can change
semantics:

```axon
y <- cond ? good : bad()
```

must not become:

```axon
tmp <- bad()
y <- cond ? good : tmp
```

because `bad()` was previously evaluated only when `cond` was false.

Hardening:

- only atomicize inside an expression when the expression position is eager
- never hoist from ternary branches to a prefix outside the ternary
- either leave branch expressions nested or introduce explicit branch-local graph
  structure at Graph IR level
- add validator coverage that atomicization preserves ternary laziness

Recommendation:

- do not run this pass on Axon AST by default
- prefer doing backend-specific argument normalization on Graph IR, where branch
  semantics can be represented explicitly

### Purity-Based Dead Code Elimination

Current risk:

The optimizer currently treats calls as pure if their arguments are pure. That is
not sound for Axon.

Example:

```axon
main :: Tensor[B,S,D]
main = do
  unused <- Params.param @@missing
  return x
```

Deleting `unused` removes a missing-parameter error. That changes semantics.

Other partial operations:

- `Config.int @@missing`
- `List.index xs i`
- tensor shape/index ops with invalid dims
- any primitive that can reject at runtime
- parameter or config lookup

Hardening:

- do not classify calls as pure by default
- introduce an explicit primitive/definition effect model:
  - `total_pure`: can be removed/duplicated/reordered
  - `partial_pure`: deterministic but may fail; can be common-subexpression
    optimized only if evaluation is preserved
  - `effectful`: cannot be removed/reordered
- until this exists, DCE may only delete unused binds whose RHS is atomic or a
  literal/container of atomics

### Single-Use Pure-Bind Inlining

This has the same purity issue as DCE, plus an ordering issue.

Constrain it to:

- atomic RHS
- literal/container RHS
- already-evaluated names
- no call RHS unless the effect model proves total purity
- no movement across ternary branch boundaries
- no change that produces non-flat Axon unless the next stage explicitly accepts it

### Constant-Param Specialization

Useful but not safe enough as a broad default.

Hardening requirements:

- only specialize parameters whose actuals are atomic literals or paths
- do not specialize on values that may be rebound in recursive SCCs
- use SCC analysis to avoid non-termination
- require a monotonic metric:
  - number of removed params decreases, or
  - number of callsites decreases, or
  - rendered/structural hash reaches fixpoint without cycles
- bind callsites against the old signature and rewrite against the new signature
  through one shared signature-rewrite utility
- preserve constraints by substituting specialized params into constraints
- re-typecheck immediately after specialization
- reject the pass result if typecheck changes observable signature semantics

### Single-Callsite Specialization

Safe only when the callsite is unique and actuals are safe.

Additional requirements:

- do not specialize main
- do not specialize recursive self-calls
- do not specialize across recursive SCCs unless a formal recursive specialization
  rule exists
- do not specialize on path templates unless all template variables are explicitly
  in lexical scope and represented structurally
- include tests with recursive loop helpers, optional caches, and path templates

### Multi-Statement / Module Inlining

Inlining can be correct, but it is a high-risk transformation.

Hardening:

- inline only straight-line modules:
  - no `if`
  - no `for` / `repeat`
  - no `scope`
  - no `yield`
  - no branch-sensitive ternary hoisting
- require one callsite unless a separate duplication-cost policy exists
- freshen names structurally, not by regex
- substitute type variables, dim variables, path variables, and constraints through
  one shared substitution engine
- validate closedness after inlining
- validate flatness after inlining
- re-typecheck after inlining
- compare call result type before/after inlining

Inlining in expression position is especially risky. It should not introduce prefix
statements that are evaluated outside the expression's original lazy branch context.

### Constraint-Driven Branch Folding

Constraint folding is safe only if constraints are fresh and provenance-aware.

Requirements:

- constraints must be produced by the latest typecheck run
- optimizer must mark constraints stale after any rewrite that changes:
  - definitions
  - call arguments
  - guards
  - params
  - return expressions
  - types/dims
- any pass using constraints must run only after fresh typecheck
- constraints need provenance:
  - local fact
  - branch guard
  - callsite guard
  - propagated interprocedural guard
- branch folding may only use facts whose guard dominates the folded expression
- after folding, re-typecheck and revalidate constraints

Do not use global per-module facts as if they held unconditionally unless the
constraint store explicitly marks them unconditional.

### Loop / Path Helper Prefixing

The phrase refers to optimizer logic that detects generated loop helper definitions
from their names and rewrites absolute path templates in their bodies.

Example shape:

```axon
gpt2__loop_h_recur_continue_2 ...
```

The helper name encodes that it belongs to loop `h` with loop variable `i`; the
optimizer then tries to prefix paths such as:

```axon
@@attn.c_proj
```

into something like:

```axon
@@'h.{i}.attn.c_proj'
```

Problem:

This is semantic information inferred from generated names. That is brittle.

Hardening:

- flatten should attach structural metadata to generated helpers:
  - original loop name
  - loop variable
  - lexical scope path
  - template variables in scope
- optimize should consume that metadata, not parse helper names
- path composition should operate on structured `AxonExprPath`, not strings
- validate that all template placeholders are lexically bound after rewrite
- if metadata is absent, skip the pass rather than guessing

## Graph IR Versus Pre-Graph Optimization

Most optimization should eventually happen at Graph IR level.

Better pre-Graph-IR passes:

- rooted pruning
- atomic alias cleanup
- literal-only constant folding
- safe path template normalization that still needs Axon lexical metadata
- readability/canonical naming if the output is an Axon inspection artifact

Better Graph IR passes:

- dead temporary elimination
- CSE
- pure op folding
- backend argument normalization
- shape-based simplification
- graph-level branch/select simplification
- MoE selected-expert rewrite from compact select/scatter form to top-k-axis
  batched expert execution
- op fusion
- custom kernel selection
- layout planning
- pipeline partitioning

Reasons:

- Graph IR has typed operands and explicit node outputs.
- Graph IR can preserve multi-output destructuring without lowering through lists.
- Backend constraints belong closer to backend IR.
- Eager/lazy evaluation hazards are easier to make explicit if branch/select nodes
  have clear semantics.
- Future custom kernels need graph-level shape/type facts.

Recommendation:

- keep pre-Graph optimizer small and conservative
- move aggressive optimization to typed Graph IR
- define an explicit Graph IR effect/partiality model before DCE, inlining, or CSE
  over calls

Graph-level pruning is still needed even though resolve/typecheck prune Axon
definitions. Graph optimization can introduce specialized clones, inline helpers,
and remove callsites; pruning keeps the executable graph rooted at `main` after
those graph-only rewrites.

A full Graph IR typechecker is not currently required for the implemented safe
passes, but the validation boundary should grow. The near-term target is a
Graph IR verifier that checks:

- all operands are typed
- op outputs preserve the declared node/module output types
- specialization substitutes inputs consistently in nodes and outputs
- inlining preserves the original call result arity and result types
- no non-total call is deleted, duplicated, or reordered

## Missing Work

The current implementation is a first conservative slice. Before any optimizer is
enabled by default, these gaps need to be closed.

### Graph IR Verifier / Type Checker

Status: first conservative verifier implemented. Remaining items below are the
next hardening steps.

`validate_graph_program` is structural. It checks references, module existence,
input/output names, and duplicate definitions. It does not yet prove that graph
rewrites preserve type and arity semantics.

Missing:

- stricter signature-substitution checks for specialization
- explicit before/after call-result equivalence checks for inlining
- better distinction between alpha-equivalent dim binders and accidental dim
  mismatch
- validation of lowered constraints against graph values
- optional stricter mode for tests that should reject broad `Any`/type-variable
  compatibility

Until the remaining checks exist, graph optimization must stay conservative and
off by default.

### Effect / Partiality Model

Status: first explicit model implemented. Unknown calls default to
`partial_pure`; known-total `core.alias`, `core.ascribe`, `core.list`,
`core.tuple`, and `core.binary.*` are `total_pure`. `core.select` remains
`partial_pure` because the selected branch may be partial and must not be deleted.

Graph DCE currently deletes only known-total `core.*` nodes. This is safe but
limited. Most Axon calls must be treated as partial unless explicitly proven
otherwise, because deleting an unused call can delete a required runtime error.

Missing:

- primitive effect declarations
- richer derived definition effect inference with SCC-aware fixed-point reporting
- effect propagation through nested graph expressions
- richer distinction between:
  - `total_pure`: removable, duplicable, reorderable
  - `partial_pure`: deterministic but may fail; evaluation must be preserved
  - `effectful`: must not be removed, duplicated, or reordered
- tests that `Params.*`, `Config.*`, `List.index`, and invalid tensor ops are not
  deleted unless an effect rule permits it

Broader DCE, CSE, and inlining should wait for this model.

### Specialization Hardening

Current graph specialization is deliberately narrow: it specializes only safe
literal/path actuals, does not specialize `main`, and is mainly intended for
single-callsite helpers.

Status:

- SCC analysis is implemented for specialization.
- Recursive SCCs and self-recursive modules are not specialized.
- Constrained modules can be specialized only when constraint substitution is
  representable and leaves no false constraints. Stale unguarded constraints are
  dropped as unusable optimization metadata; callsite-guarded interprocedural
  facts may mention caller-side refs and are preserved.
- Path actuals are specialized only when they do not contain template
  placeholders.
- Primitive wrappers are not specialized; op-specific primitive type rules should
  remain attached to direct primitive calls.
- Specialized clone names are collision-safe.
- Graph validation runs after clone creation and callsite rewrite.
- Tests cover successful single-callsite path/dim specialization, constraint
  substitution, false-constraint rejection, primitive-wrapper rejection, stale
  constraint cleanup, callsite-guarded constraint preservation, and recursive
  self-specialization rejection.

Missing:

- structural substitution for path templates and template variables
- rewriting support for nested module calls, or explicit validation that no nested
  callsites are being specialized accidentally
- monotonic termination metric beyond iteration cap
- benchmark coverage on generated loop helpers and path-templated helpers

### Inlining Hardening

Current graph inlining only inlines helpers proven `total_pure` by the graph
effect model. It is intentionally limited to single-callsite helpers.

Status:

- Inlining skips `main` and recursive SCCs.
- Constrained helpers may be inlined only when their constraints can be
  substituted into the caller, are not false, and remain representable.
- Inlining requires exactly one callsite.
- Nested expression callsites are supported under the same total-pure and
  representable-constraint requirements.
- The call node must pass all args positionally, with matching input/output arity.
- Actual/formal and returned/output types are checked with the same compatibility
  rules as `validate_graph_program`.
- Inlining rewrites module returns through the same substitution used for nodes.
- Graph validation runs after inlining and pruning.
- Optimizer validation checks call-result types after applying call-site dim
  substitutions, catching stale concrete result annotations before inlining or
  specialization can preserve them.
- Tests cover single-output helpers, multi-output helpers, path operands,
  constrained helpers, dim value refs instantiated from call-site types, and
  nested expression callsites.

Missing:

- explicit branch/laziness checks for nested expression inlining
- cost policy for duplicating work
- tests for tuple/list-valued helpers
- broader benchmark coverage on generated model graphs

Until then, broad module inlining belongs behind `--optimize-graph` only.

### Graph Roundtrip / Stability Tests

Focused unit tests exist. Graph IR roundtrip scripts now accept an
`--optimize-graph` switch, and dedicated optimized weak/strong script entrypoints
exist for manual or CI runs.

Status:

- `scripts/axon_graph_optimize_weak_roundtrip.py` runs the weak optimized Graph IR
  roundtrip.
- `scripts/axon_graph_optimize_strong_roundtrip.py` runs the strong optimized
  Graph IR roundtrip.
- A focused weak pytest covers `generic-gpt2.axon` with graph optimization.
- A focused weak pytest covers `generic-llama4.axon` with graph optimization.
- Graph IR now treats path-template placeholders such as `{CFG}` as first-class
  dependencies for validation, DCE liveness, and pruning.
- `_topk` no longer infers a fake concrete rank from variadic `Tensor[..S]`;
  when rank is unknown, the primitive rule declines and leaves the broader
  wrapper signature.
- Graph IR validation reuses the same variadic dim compatibility rule for value
  references that it uses for type compatibility.
- Graph IR validation no longer compares instantiated call-result types against
  unsubstituted generic callee return signatures; it still validates actual
  arguments, local node output contracts, module returns, and graph closedness.
- Typecheck2 now re-annotates return expressions after unification against an
  explicit return signature, and branch joins preserve rank for matching
  variadic tensor branches.
- Full non-optimized Graph IR weak and strong roundtrips pass for all 256 model
  `.axon` files under `pytest-xdist -n 8`.
- Broader optimized roundtrips are intentionally not enabled in pytest yet.
  Strong optimized reparsing still needs a cleaner contract for already-closed
  graph-rendered artifacts.

Missing:

- tests across all model `.axon` files, run with `pytest-xdist`
- stable textual diffs for failures, similar to the Axon stage roundtrips
- resolving whether strong optimized graph roundtrip should reresolve closed
  graph-rendered artifacts from arbitrary temp paths or parse them directly

### Benchmark / Fidelity Validation

The flags are wired but not validated on full model families.

Status:

- `--optimize-ast` smoke passed on `generic-gpt2.axon` with
  `openai-community/gpt2`: `masked_top1_eq=True`, max abs diff
  `0.0001068115234375`.
- Historical broad AST optimizer smokes passed on:
  - `generic-gpt2.axon` with `openai-community/gpt2`:
    `masked_top1_eq=True`, max abs diff `0.0001068115234375`
  - `bert-base-uncased.axon` with `google-bert/bert-base-uncased`:
    `masked_top1_eq=True`, max abs diff `2.956390380859375e-05`
  - `generic-llama4.axon` with `test/Llama4-Test`:
    `masked_top1_eq=True`, max abs diff `0.010181993246078491`
  - `generic-olmoe.axon` with `test/OLMoE-Test`:
    `masked_top1_eq=True`, max abs diff `5.364418029785156e-07`
- The broad AST smoke exposed and fixed two generic primitive/type metadata bugs:
  `_topk` should not preserve the eliminated `IdxTensor` alias in core typed AST,
  and `_tensor_like` should use the reference tensor shape.

Status:

- `--optimize-graph` smoke passed on `generic-gpt2.axon` with
  `openai-community/gpt2`: `masked_top1_eq=True`, max abs diff
  `0.0001068115234375`. Latest log:
  `log/gpt2-kv-global-backend-smoke/run.log`.
- combined `--optimize-ast --optimize-graph` smoke passed on
  `generic-gpt2.axon` with `openai-community/gpt2`: `masked_top1_eq=True`,
  max abs diff `3.0517578125e-05`. Log:
  `log/smoke-optimize-flags-20260511`.
- all-file combined optimized weak Graph IR roundtrip is now covered as a
  non-strict xfail tracker. Latest run:
  `pytest -q -n 8 tests/test_synapse_axon_graph_ir_roundtrip.py -k safe_optimized_graph_ir_weak_roundtrip_is_canonical`
  produced 66 stable files and 190 files with changed rendered text across
  generations. The hard all-file roundtrip requirements remain the non-optimized
  weak and strong Graph IR roundtrips.
- max-4B combined optimized fidelity is covered by an opt-in long pytest:
  `BS_LONG=1 pytest tests/test_synapse_optimizer_fidelity_max4b.py`. It runs all
  `generic-*.axon` files through `axon-benchmark` with
  `--max-billion-parameters 4`, `--optimize-ast`, and `--optimize-graph`, and
  requires masked top-1 equality plus max absolute diff below `1e-2`.
- max-1B and max-4B fidelity comparison against the current non-optimized default
- targeted large-model reruns for families affected by path templates, loops, MoE,
  cache helpers, and multi-output calls

Any optimizer default change requires clean fidelity evidence first.

### Legacy Optimizer Decomposition

The old broad AST optimizer implementation still exists internally and still
mixes safe, unsafe, and backend-shape rewrites. It is no longer exposed through a
CLI flag.

Missing:

- move every safe pre-Graph rewrite into `optimize_safe`
- move graph-suitable rewrites into `optimize_graph`
- delete or quarantine any remaining legacy-only rewrites that exist for old backend contracts
- ensure no required backend behavior is hidden behind optional optimization

### Graph-Level Constraints

Graph IR does not yet own a fresh constraint store.

Missing:

- lower typed constraints into Graph IR, or recompute graph constraints directly
- mark constraints stale after any graph rewrite that changes operands, guards,
  calls, outputs, or types
- validate that branch facts dominate the folded expression before using them
- support guarded facts for `core.select`, null checks, bool checks, and dim/int
  equalities

Until this exists, constraint-driven branch folding should stay out of graph
optimization.

### Higher-Level Graph Optimizations

The current graph optimizer performs cleanup, not substantial backend optimization.

Missing future passes:

- scalar/dim expression simplification
- single-use shape/list literal cleanup for flat-safe primitive arguments
- return-only tuple scaffolding cleanup
- CSE for total operations
- dead branch elimination with fresh constraints
- backend argument normalization where needed
- op fusion
- layout planning
- custom kernel selection
- pipeline partitioning support
- MoE selected-expert rewrite from compact `where/scatter` form to top-k-axis
  batched expert execution

### MoE Selected-Expert Rewrite

Current derived MoE code often has an expert-centric shape:

```axon
routed <- Tensor.zeros_like hidden
routed <- for e <- [0..E) carry (routed) do
  x_sel, token_idx, topk_pos, sel_scores <- MoE.select hidden topk_scores topk_indices e
  x_upd <- expert_body e x_sel
  routed <- MoE.scatter_add routed token_idx x_upd sel_scores
  yield routed
return routed
```

`MoE.select` currently reaches the primitive `_where_indices`, which produces
data-dependent compact index tensors. That is hard for tensor backends without a
native nonzero/compaction primitive and should not be handled by backend
special-casing normal Axon modules.

Graph IR optimization recognizes proven slices of this semantic pattern, with
structural provenance checks rather than module-name hacks, and rewrites them
into selected-expert batched execution over the router top-k axis:

```text
hidden_sel = unsqueeze(hidden, 2) expanded to [B,T,K,D]
gate_up    = indexed_expert_linear(gate_up_weight, hidden_sel, topk_indices)
act        = expert activation/body over [B,T,K,*]
down       = indexed_expert_linear(down_weight, act, topk_indices)
routed     = sum(down * unsqueeze(topk_scores, -1), dim=2)
```

Required preconditions for the general form:

- the loop range covers expert ids and has no side effects beyond the routed carry
- `MoE.select` consumes the same `hidden`, `topk_scores`, and `topk_indices` for
  each expert
- the selected expert id is used only to select expert parameters, not as an
  arbitrary term with other semantics
- `MoE.scatter_add` is the inverse accumulation of `MoE.select`
- the expert body is straight-line and preserves the leading selected-token axis
- all parameter paths and expert weight layouts are represented structurally

The rewrite target should be generic Graph IR primitives or canonical library ops
such as indexed expert linear/gathered expert matmul. Backends can then implement
those efficiently:

- torch can use advanced indexing plus `einsum`/batched matmul
- tinygrad can use the same selected-weight indexing pattern used by its local
  MoE example (`weight[sel]`) and reduce over the top-k axis

The first implemented slices cover packed SwiGLU experts, clamped packed SwiGLU
experts, packed GeGELU experts, and ReLU2 two-linear experts, lowering to
`__torch_selected_expert_packed_swiglu_ffn`,
`__torch_selected_expert_clamped_packed_swiglu_ffn`,
`__torch_selected_expert_packed_gegelu_ffn`, or
`__torch_selected_expert_relu2_ffn` for `codegen2-torch` only. This pass is an
optimization, not a semantic crutch. If the pattern is not proven, the graph
remains unchanged; a backend that cannot lower `_where_indices` should then fail
with an unsupported primitive error.

## Proposed Refactor Slices

1. Keep backend-required normalization removed.
- Graph IR/codegen2 should handle multi-output nodes directly.
- Required backend behavior must be represented in Graph IR or in explicit backend lowering.
- Do not reintroduce non-optional AST rewrites for old backend contracts.

2. Introduce `optimize-ast` / `optimize_safe`.
- Contains rooted pruning, atomic alias cleanup, literal-only folding.
- Runs to fixpoint with typecheck/validate.
- No call DCE, no specialization, no inlining.

3. Add optimizer validators.
- `validate.optimized_safe`
- checks no stale typed metadata
- checks no branch-hoisted statements
- checks typed/flat/closed invariants

4. Add an effect model.
- Start with conservative defaults: calls are partial unless declared otherwise.
- Primitive declarations can opt into `total_pure`.
- User definitions inherit the least-safe effect of their body.

5. Harden specialization and inlining.
- Shared signature rewrite utility is partially implemented through the Graph IR
  substitution helpers and should be extended instead of adding local rewrite
  functions.
- Shared substitution utility for terms/types/dims/constraints/paths is
  implemented and is used by dim substitution, specialization, and inlining.
- SCC-aware termination checks are implemented for specialization/inlining.
- Inlining metadata substitution is implemented for copied nodes and returned
  operands.
- Explicit metadata for generated helpers remains open.

6. Move broader rewrites to Graph IR.
- DCE
- CSE for single-output `total_pure` nodes and repeated nested `total_pure`
  expression operands is implemented; broader CSE for partial-pure expressions
  remains disabled until there is an explicit eager-semantics rule.
- backend argument normalization
- op fusion
- shape/layout driven optimizations

## Default Policy

For now:

- `--optimize-ast` and `--optimize-graph` should remain off by default
- full `optimize_flat_typed_axon_file` should be treated as internal experimental
  code, not as a CLI/backend contract
- safe pieces should be split out before enabling any optimizer by default
- benchmark/fidelity runs should include both non-optimized and safe-optimized paths
  before changing defaults
