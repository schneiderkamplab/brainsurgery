# Refactor Axon Plan

This document defines the target compiler architecture for Axon.

The key design constraint is:

- there is one canonical Axon AST
- every pre-lowering stage consumes and produces that same AST
- stages differ by invariants and populated metadata, not by switching AST types

## Canonical Pipeline

Stage status as of now:

- done for current slice: `parse`, `load`, `materialize`, `resolve`, `validate`
- usable, but not final-form complete: `normalize`, `flatten`, `typecheck`, `optimize`
- active refactor now started, but still not final-form complete: `lower`
- active-path aligned but still not final-form complete: `runtime`, `codegen`
- separate legacy area still ahead: `pipeline`

1. `parse`
- input: Axon source text
- output: one-file Axon AST
- status: done

2. `load`
- input: one-file Axon AST plus origin path / search roots
- output: list of one-file Axon ASTs (root + imported files)
- status: done

3. `materialize`
- input: one-file Axon AST
- output: one-file Axon AST
- scope: only resolve materialization-specific `Config.*`
- exclusions:
  - no constant folding
  - no dead-code elimination
  - no alias expansion
  - no inlining
- status: done

4. `resolve`
- input: list of one-file Axon ASTs
- output: one closed Axon AST
- responsibilities:
  - merge imported files into one closed program
  - resolve qualified and unqualified references
  - remove import surface state from the output AST
- exclusions:
  - no file loading
  - no warning policy
  - no strict/fail-on-warning policy
  - no type inference
  - no lowering
- status: done

5. `validate`
- input: closed Axon AST
- output: the same closed Axon AST, certified to satisfy pre-flatten invariants
- responsibilities:
  - closedness checks
  - unresolved-name rejection
  - type-alias arity and visibility checks
  - strict/warning diagnostics via the validation layer
- exclusions:
  - no type inference
  - no dimension inference
  - no arity inference
  - no lowering-time fallback inference
- status: done

6. `normalize`
- input: validated closed Axon AST
- output: normalized closed Axon AST
- responsibilities:
  - normalize raw pragma occurrences into semantic pragma values
  - desugar call syntax only:
    - callee path suffixes such as `NN.linear@w` become ordinary `Path` arguments
    - pipe syntax becomes ordinary calls
    - bare callable-name expressions that require call semantics become zero-arg calls
  - normalize loop protocol:
    - make implied loop `carry` explicit when assignment-target loop rules imply it
    - make implied final `yield` explicit
    - insert explicit null-yield for effect-only loops so flatten has one loop protocol
  - keep all rewrites structural in the AST; no render/parse roundtrips
- invariants after this stage:
  - no callee path sugar remains
  - no pipe expressions remain
  - no bare callable-name expression remains when it semantically means a call
  - every loop body has an explicit final `yield`
  - every value-producing loop has explicit `carry`
  - certified by `validate.normalized`
- exclusions:
  - does not expand omitted optional/default call arguments globally
  - signature-changing optimizer passes must bind affected callsites against the old callee signature and re-emit them against the new signature
- status: implemented as mandatory pre-flatten normalization

7. `flatten`
- input: validated closed Axon AST
- output: flat closed Axon AST
- responsibilities:
  - desugar nested expression forms
  - flatten complex control/statement sugar
  - normalize all paths to absolute templated paths
- current flat-core invariants:
  - no `scope`
  - no `for`
  - no statement-level `if`
  - no relative paths
  - no hidden path sugar in callees
  - `return` / `yield` values are atomic names or tuples of names
- status: usable, but not final-form complete
- still open:
  - some helper shapes remain flattening artifacts rather than a final canonical presentation
  - some path/template normalization is intentionally left for `optimize`
  - synthesized scope arguments are still introduced here because flatten owns scope elimination

8. `typecheck`
- input: flat closed Axon AST
- output: fully typed flat closed Axon AST
- responsibilities:
  - type checking against declared Axon types
  - primitive op signature checking
  - type inference
  - dimension inference
  - arity inference
- typed metadata placement:
  - every expression node carries:
    - `inferred_type`
    - `inferred_arity`
    - `inferred_dims`
  - every definition/module carries:
    - `constraints`
- current constraint shape:
  - symbolic and not solver-restricted
  - per-module
  - records `Dim`/`Int` arithmetic relations, boolean facts, and null-vs-not-null facts
  - direct guarded call sites are threaded interprocedurally into callee modules under synthetic `callsite` guards
- status: usable, but not final-form complete
- still open:
  - constraint recording is broader than constraint reasoning
  - richer future type/inference cases are still ahead
  - Axon lacks a first-class way to write dependent return types for value-indexed
    sequence outputs such as `Tensor.split x sizes=[A, B]`
    - current surface wrapper therefore uses an intentionally broad declared
      return and relies on the primitive `_split` type rule to infer the precise
      tuple/list result from the `sizes` value
    - example desired inferred return:
      `(Tensor[..prefix,A,..suffix], Tensor[..prefix,B,..suffix])`
    - replace the broad wrapper signature once the type language can express
      size-indexed tuple/list outputs directly
  - lowering-oriented typing concerns are intentionally deferred to later stages

9. `optimize`
- input: flat fully typed closed Axon AST
- output: optimized flat fully typed closed Axon AST
- responsibilities:
  - inline alias definitions and wrappers
  - constant folding
  - dead-code elimination
  - inlining
  - other local rewrites
  - canonicalize path-sugar module parameters into ordinary `Path`-typed params
- examples:
  - simplify loop termination logic after flatten, such as folding `step=1`
  - remove dead temporaries introduced by flatten
  - fold arithmetic or boolean constants after typecheck
  - simplify ternaries whose condition becomes constant
  - compose absolute templated paths when a template environment is available
    - example:
      - caller provides `@@'h.{i}'`
      - callee body uses `@@'{__scope}.attn.c_attn'`
      - optimize may rewrite this to `@@'h.{i}.attn.c_attn'`
    - this is template-aware substitution / specialization, not plain constant folding
    - it requires threading bound template variables such as `i`
- status: usable, iterated to fixpoint, but not final-form complete
- implemented so far:
  - inline definition-alias wrappers at call sites
  - constant folding for local arithmetic / comparison / ternary expressions
  - fold boolean identities such as `true or x`, `false and x`
  - eliminate trivial and some broader single-use pure temporary binds without violating flat invariants
  - convert scoped/path helper params to ordinary `Path`-typed params
  - prune unused callee params and rewrite direct callsites accordingly
  - specialize single-callsite modules by constant actuals
  - inline single-callsite straight-line modules
  - canonicalize generated helper and local names for readability
  - rerun structural and local rewrites to fixpoint with re-typecheck / re-validate between iterations
- remaining work:
  - richer constraint-driven simplification and dead-code elimination
  - more template-aware path composition / specialization
  - further cleanup of flattening artifact helpers where semantics permit
  - any more aggressive inlining/specialization only with careful semantic justification

10. `backend-required-normalize`
- input: flat fully typed closed Axon AST
- output: flat fully typed closed Axon AST satisfying backend shape requirements
- responsibilities:
  - run structural rewrites required by lowering/runtime even when `--no-optimize` is set
  - currently includes list destructuring normalization into explicit `_list_index` binds
- invariants after this stage:
  - no list-valued multi-target destructuring bind remains
  - certified by `validate.backend_required`
- status: implemented as a non-optional lowering preparation pass
- note:
  - `--optimize/--no-optimize` controls semantic optimization, not required canonical backend shape

11. `lower`
- input: flat fully typed closed Axon AST
- output: Synapse graph
- responsibilities:
  - consume the canonical flat typed optimized Axon AST directly
  - lower explicit leading `Path` arguments on primitive calls directly, without callee-string path sugar
  - reject pre-flat constructs instead of trying to lower them
- current active-path invariants:
  - no `scope`
  - no `for` / `repeat`
  - no lowering-time scope/root threading
  - no leading-`Path` rewrite into legacy `callee@path` form
- status: started and usable on the active path, but not final-form complete
- still open:
  - more of the old helper layer can still be deleted
  - final cleanup should keep only the minimal graph forms actually produced by the new lowering path
  - typed artifact loading is not stage-correct yet:
    - rendered `*.typed.axon` files currently print inferred metadata as normal `::` ascriptions
    - parsing those files back treats annotations as source-level ascriptions, which can change re-typechecking behavior
    - lowering should eventually have a typed-artifact loader mode that restores rendered annotations into `inferred_type` / `inferred_arity` / `inferred_dims` metadata and then consumes the flat typed AST directly, without re-running typecheck/optimize unless explicitly requested
    - until that exists, benchmark and lowering entrypoints should use original source Axon files rather than generated `tmp/*.typed.axon` inspection artifacts

12. `runtime`
- input: Synapse graph
- output: executed model outputs
- status: active emitted-graph path partly aligned, but not final-form complete
- implemented so far:
  - no active-path `for` execution
  - no active-path `_param_root` handling
  - parameter-path inference centered on `_abs_path`, `_params`, and `param_base`
- still open:
  - broader backend cleanup against the smaller emitted graph contract
  - separation from still-legacy pipeline/runtime consumers that build older graph forms manually

13. `codegen`
- input: Synapse graph
- output: generated PyTorch code
- status: active emitted-graph path partly aligned, but not final-form complete
- implemented so far:
  - no active-path generated `for` support
  - no active-path generated `_param_root` / root-stack machinery
  - generated parameter-path inference simplified to `_abs_path`, `_params`, and `param_base`
- still open:
  - further pruning of legacy helper/code paths not exercised by the new lowering output
  - separation from still-legacy pipeline/codegen consumers

14. `pipeline`
- input: either compiler metadata plus lowered graph, or analysis results over the closed/flat/typed AST
- output: pipeline partitioning plan and stage-specific lowered graphs/specs
- status: still legacy-backed
- open design choice:
  - either collect the same loop-bound / layer-range information from the original higher-level Axon AST and preserve it as compiler metadata for later pipeline use
  - or analyze the closed, flat, typed AST directly and recover the same partitioning facts there
- note:
  - this should be treated as a separate design problem from the active lowering/runtime/codegen cleanup
  - the current pipeline backend still assumes older `for`-based graph structure

## Next IR Direction

The current Synapse YAML-shaped graph should be replaced by an in-memory graph IR before runtime/codegen.

Initial implementation status:

- `brainsurgery/synapse/axon/graph_ir/` defines a typed in-memory graph IR.
- `lower_axon_program_to_graph_ir(...)` is available as an alternative lowering target.
- The graph IR lowering reuses the existing canonical flat typed Axon preparation pipeline.
- Graph nodes carry typed positional operands, typed kwarg operands, SSA-style outputs, structured paths, constraints, constants, and inferred type metadata.
- `brainsurgery/synapse/axon/codegen2/` exists as the separate future backend entry point and consumes graph IR, not Synapse spec dictionaries.
- Existing Synapse-spec lowering/codegen remains unchanged while codegen2 is filled in incrementally.

Reasons:

- YAML-shaped dictionaries keep reintroducing string serialization bugs.
- Many lowering/codegen bugs come from reconstructing typed facts from strings.
- Path templates, symbolic dims, tuple/list values, and op kwargs need typed fields, not ad-hoc payload conventions.

Recommended direction:

- Keep Axon AST through `canonicalize`.
- Lower canonical flat typed Axon into a typed graph IR, not directly into YAML dictionaries.
- Make graph nodes dataclasses with typed fields:
  - `op`
  - `inputs`
  - `outputs`
  - `kwargs`
  - `path`
  - `type/shape metadata`
  - optional guards / constraints
- Make runtime/codegen consume this graph IR directly.
- Add a debug/export renderer from graph IR to YAML or JSON only for inspection, not as an internal compiler boundary.

Direct AST-to-codegen/runtime is possible, but less attractive:

- codegen/runtime would need to understand too much Axon semantics
- graph-level backend passes would become harder to isolate
- pipeline partitioning still wants a backend-neutral graph-like representation

So the target should be:

```text
canonical flat typed Axon AST -> typed Graph IR -> runtime/codegen/pipeline
```

Active transition shape:

```text
canonical flat typed Axon AST -> Synapse graph -> runtime/codegen
canonical flat typed Axon AST -> typed Graph IR -> codegen2
```

## Canonical AST

There is one AST representation for all pre-lowering stages.

It must be able to represent:

- one-file surface Axon
- materialized Axon
- loaded import sets
- closed resolved Axon
- fully typed Axon
- flat Axon

The AST changes by:

- removing or normalizing fields
- populating inferred metadata
- strengthening invariants

It must not change by switching to separate stage-specific AST node families.

## AST Requirements

The AST package owns:

- node definitions
- type-expression definitions
- file/program container definitions
- path-expression representation
- Axon rendering for any AST
- human-inspection export helpers
  - textual tree or similar
  - Graphviz `.dot`
- stage validators for AST invariants

Typed-stage support in the canonical AST includes:

- per-expression inferred type/arity/dims
- per-module symbolic constraints

The AST should also support optional metadata such as:

- source spans
- canonical symbol identities
- inferred expression types
- inferred arities
- path-normalization state

These are metadata on the one AST, not separate AST types.

## Source Spans

Source spans are diagnostic metadata attached to AST nodes.

Typical components:

- origin file path
- start line / column
- end line / column
- optional byte offsets

These are for diagnostics and inspection, not semantics.

## Stage Contracts

### Parse

- pure source-text to AST
- must not load imports
- must not resolve modules by path
- public parser entrypoint should take source text, not a path

### Load

- discovers imports from the root AST
- loads imported Axon sources
- parses them into one-file ASTs
- returns a list of ASTs
- does not resolve names or rewrite programs

### Materialize

- only handles materialization-specific config specialization
- returns AST

### Resolve

- takes a list of ASTs
- returns one closed AST
- removes import surface state from the resolved result

### Validate

- takes a closed AST
- checks pre-flatten invariants only
- does not perform full typing

### Flatten

- takes a validated closed AST
- returns a flat closed AST

### Typecheck

- takes a flat closed AST
- returns a fully typed flat closed AST

### Optimize

- takes a flat fully typed closed AST
- returns a rewritten flat fully typed closed AST
- this stage does not exist as a real package yet and is the next planned stage
- it owns post-flatten simplification work such as:
  - constant folding
  - dead temporary elimination
  - dead branch elimination after conditions become constant
  - trivial helper inlining where justified
  - local algebraic cleanup that should not complicate `flatten`
  - template-aware path composition / specialization when lexical template bindings are available
    - this is where forms such as:
      - `gpt2_block @@'h.{i}' ...`
      - `@@'{__scope}.attn.c_attn'`
      can be combined into:
      - `@@'h.{i}.attn.c_attn'`
    - this requires passing and substituting template variables like `i`
    - it should not be done in `flatten`

### Lower

- takes a flat fully typed closed AST
- returns a Synapse graph

## Validation

Validation should be grouped in a dedicated `validate` package and reused across stages.

Representative validator groups:

- `validate.surface`
- `validate.closed`
- `validate.flat`
- `validate.normalized`
- `validate.typed`
- `validate.backend_required`

Validation owns:

- structural closedness checks
- explicit normalized-call and normalized-loop protocol checks
- flat-core checks
- typed metadata checks
- backend-required flat typed checks
- unused-import warnings
- unused-definition warnings
- strict-mode warning escalation used by CLI/workflow layers

Stage packages should validate their required input shape and validate the properties they produce.
They should return ASTs plus diagnostics where needed.
They should not own user-facing warning policy.

Validation modules report structural problems.
User-facing error collection and stage orchestration should live in a higher-level orchestration layer, not inside individual validators.

## Rendering And Inspection

Any AST produced by any stage should be:

- renderable back to Axon source
- intended to remain parsable as Axon source

For typed ASTs, rendering may use explicit type ascriptions where needed, for example:

```axon
((Math.exp D) :: Float)
```

The AST package should also expose Graphviz export so each AST stage can be inspected visually.

## Packaging Targets

Target package layout:

- `axon/ast/`
- `axon/parse/`
- `axon/load/`
- `axon/materialize/`
- `axon/resolve/`
- `axon/typecheck/`
- `axon/flatten/`
- `axon/optimize/`
- `axon/lower/`

Backend package targets:

- `synapse/runtime/`
- `synapse/codegen/`

## Near-Term Refactor Steps

1. Make `parse` pure source-text to AST.
2. Separate `load` from `parse`.
3. Move `materialize` into its own package and narrow its semantics.
4. Make `resolve` consume a list of ASTs and produce one closed AST.
5. Move `typecheck` into its own package and make it return a fully typed AST.
6. Add `flatten`.
7. Add `optimize`.
8. Make `lower` consume only fully typed flat AST.
