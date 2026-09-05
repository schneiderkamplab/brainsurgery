# BrainSurgery Interfaces Reference

This document explains the external interfaces in `brainsurgery`, what each is for, and when to use them.

## 1. Interface overview

`brainsurgery` exposes five practical interfaces:

1. Batch CLI (`brainsurgery ...` / `brainsurgery cli ...`)
2. Interactive CLI mode (`brainsurgery -i ...`)
3. OLY one-line syntax (interactive input format)
4. Web CLI server (`brainsurgery webcli`)
5. Web UI server (`brainsurgery webui`)

For internal integration, it also exposes Python-level extension interfaces (transform/expr/provider registration) described in `docs/codebase-reference.md`.

## 2. Batch CLI interface

Command:

```bash
brainsurgery [OPTIONS] [CONFIG_ITEMS]...
```

`CONFIG_ITEMS` behavior:

- YAML file paths (`.yaml`/`.yml`) are loaded and deep-merged in order.
- `key=value` overrides are applied last.
- Nested paths support dotted keys and list indices (`transforms[0].dump.format=compact`).

Use this interface when:

- You want reproducible, versioned plans.
- You are running CI/batch jobs.
- You need deterministic execution with fail-fast behavior.

Core options:

- `--provider inmemory|arena`
- `--num-workers`
- `--shard-size` (binary units: `100MB` = 104,857,600 bytes of tensor data per shard)
- `--summary-mode raw|resolve`
- `--summarize-path`
- `-i/--interactive` (switches to post-plan REPL)

Execution behavior:

- Non-interactive transform failures abort the run.
- If `output` is omitted, no final checkpoint is written.
- If summarize is on, executed transforms are emitted as YAML summary.

## 3. Plan YAML interface

Minimal shape:

```yaml
inputs:
  - model::/path/to/in.safetensors
transforms:
  - copy: { from: ln_f.weight, to: ln_f_copy.weight }
output:
  path: /path/to/out
  format: safetensors
  shard: 5GB
```

Notes:

- `inputs` is optional; aliases are required for multi-input plans.
- `transforms` is a list of single-key mappings.
- `output` may be omitted or string/mapping.
- `output` writes one alias. With one input that is its alias; with several inputs it is
  inferred as the single alias the transforms write to (destinations of `copy`/`move`/...,
  in-place targets, `delete`). Writing to several aliases, or to none, fails with
  `cannot infer output model uniquely`. `assert`, `diff` and `dump` do not count as writes.
- `output.shard` (and `--shard-size`) use binary units (`100MB` = 104,857,600 bytes) and
  count tensor data only; a tensor larger than the budget is written alone in its own shard,
  and an index file `model.safetensors.index.json` maps every tensor to its shard.

## 4. Tensor reference interface

Reference forms:

1. `alias::expr`
2. `expr` (uses default alias when unambiguous)
3. `alias::expr::[slice]`

`expr` modes:

- regex string (full-match against tensor names)
- structured list tokens (segment-aware matcher/rewriter)

Structured list syntax supports capture/rewrite semantics such as:

- `$x` (single segment binding)
- `*xs` (variadic segment binding)
- `~REGEX` and `~x::REGEX` forms (source-side regex matching)
- `${x}` interpolation in output patterns

## 5. Interactive CLI interface

Start interactive mode:

```bash
brainsurgery -i plan.yaml
```

Session behavior:

1. Executes configured plan first.
2. Opens prompt for additional transforms.
3. Accepts:
   - a single transform mapping (YAML or OLY)
   - a YAML list of transform mappings
4. On submitted-block error, block stops and prompt remains active.

Interactive features:

- Readline tab completion for transform names, payload keys, aliases, tensor names.
- Command history via `~/.brainsurgery_history`.
- Special control transforms:
  - `help`
  - `prefixes`
  - `exit`

## 6. OLY interface (One Line YAML)

OLY is a compact one-line command format handled by `brainsurgery/cli/oly.py`.

Examples:

- `copy: from: ln_f.weight, to: ln_f_copy.weight`
- `help: assert: equal`
- `assert: exists: model::ln_f.weight`

Use OLY when:

- You are in interactive mode and want fast iterative commands.
- You want concise single-transform edits without writing full YAML blocks.

Spec references:

- `docs/oly-spec.md`
- `docs/oly-yaml-grammar.md`
- `docs/oly-conformance.md`

## 7. Web CLI interface (`brainsurgery webcli`)

Purpose:

- Browser form that submits a full plan YAML and returns logs/output/summary in one request.

Server:

- default host: `127.0.0.1`
- default port: `8765`

Primary API:

- `POST /api/run`
  - Input fields include: `plan_yaml`, `shard_size`, `num_workers`, `provider`, `arena_root`, `arena_segment_size`, `summarize`, `summary_mode`, `log_level`
  - Response includes: `ok`, `logs`, `output_lines`, `summary_yaml`, `written_path`, `error`

Use this interface when:

- You want a thin web wrapper over batch execution.
- You do not need a long-lived editing session.

## 8. Web UI interface (`brainsurgery webui`)

Purpose:

- Session-oriented browser interface for loading models, applying transforms incrementally, and inspecting progress/model state.

Server:

- default host: `127.0.0.1`
- default port: `8766`

Key GET endpoints:

- `/api/transforms` -> transform metadata for UI controls
- `/api/state` -> current models + runtime flags
- `/api/progress` -> active transform progress snapshot

Key POST endpoints:

- `/api/load` -> upload/load model checkpoint into alias
- `/api/_apply_transform` -> apply one transform payload
- `/api/save_download` -> run save and return downloadable file bytes
- `/api/model_dump` -> render compact/tree dumps for alias/filter

Preview confirmation behavior (Web UI):

- When runtime `preview` is enabled and `dry-run` is disabled, tensor-impacting transforms require confirmation.
- `/api/_apply_transform` returns `preview_confirmation_required: true` and preview output first.
- UI shows a modal with `Go` / `No-go`:
  - `Go` resubmits with `preview_decision: "go"` and applies.
  - `No-go` resubmits with `preview_decision: "no-go"` and skips apply.

Use this interface when:

- You want iterative visual exploration.
- You need progress feedback for iterating transforms.
- You want model dump browsing without leaving browser.

Keyboard navigation (Web UI):

- `Tab` in the Transforms panel cycles between panel-level controls (search box and transform list).
- `Shift+Tab` follows the same forward panel cycle behavior used in this UI.
- `ArrowUp` / `ArrowDown` in the transform list moves selection through ready transforms.
- `Enter` in the transform list activates the focused/selected transform and moves focus to Transform Options.
- `Shift+Enter` in Transform Options runs the selected transform.

## 9. Transform command interface (runtime command surface)

Transforms are discovered through runtime registry and auto-imported modules.

Common categories:

1. Control/inspection: `help`, `prefixes`, `set`, `exit`, `dump`, `diff`, `assert`
2. IO: `load`, `save`
3. Batch runner: `execute`
4. Copy/move/delete/assign: `copy`, `move`, `delete`, `assign`
5. Binary/ternary math: `add`, `subtract`, `multiply`, `matmul`, plus in-place variants
6. Type/shape ops: `cast`, `reshape`, `permute`, `clamp`, `fill`, `scale`, split/concat
7. Init and decomposition: `zeroes`, `ones`, `rand`, `phlora`, `phlora_`

Mapping note:

- Binary mapping transforms (including `copy`, `move`, `assign`, `add_`, and `subtract_`) support destination synthesis from captures when `from` and `to` use the same expression kind:
  - regex mode: both strings (`from` regex captures reused in `to`)
  - structured mode: both lists (`$name` / `${name}` capture + rewrite tokens)
- Ternary mapping transforms (including `add`, `multiply`, `subtract`, and `matmul`) support the same capture-based rewrite model across `from_a`, `from_b`, and `to`.
- `assign` still requires synthesized destination tensors to already exist (`MUST_EXIST` policy).

`execute` transform note:

- `execute` runs a nested transform batch from one or more sources:
  - `transforms`: inline transform object/list
  - `plan`: YAML-like plan mapping/list
  - `plan-yaml`: plan YAML string
  - `path`: filesystem path to a YAML plan
- If nested plan has `inputs`, they are translated into `load` transforms at the front.
- If nested plan has `output`, it is translated into a trailing `save` transform.
- Nested execution uses the same runtime flags as the current session:
  - `dry-run` respected
  - `preview` respected
  - `verbose` respected

For payload details, run:

- `help`
- `help: <command>`
- `help: assert`
- `help: { assert: <expr> }`

## 10. Assert expression interface

Supported expression operators:

- scalar checks: `exists`, `count`, `shape`, `dimensions`, `dtype`, `iszero`
- pairwise comparison: `equal`
- access counters: `reads`, `writes`
- composition: `all`, `any`, `not`

Expression shape:

```yaml
assert:
  equal:
    left: a.weight
    right: b.weight
    eps: 1e-6
```

Expression help metadata is exposed to both CLI help and web UI transform metadata.

## 11. Choosing an interface

Use this quick matrix:

1. Reproducible automated run -> batch CLI
2. Fast terminal iteration on top of a plan -> interactive CLI (+ OLY)
3. Browser-based one-shot run -> webcli
4. Browser-based iterative session with progress/model inspection -> webui
5. Framework extension/new operations -> Python runtime interfaces (`core/`, `transforms/`, `expressions/`, `engine/providers.py`)
