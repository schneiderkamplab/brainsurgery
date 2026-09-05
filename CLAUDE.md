# CLAUDE.md

Guidance for Claude Code working in this repository. The authoritative
contributor policy lives in the `AGENTS.md` files; this file imports the root
one and adds operational knowledge that is not written down elsewhere.

@AGENTS.md

## What this repo is

`brainsurgery` is two things sharing one package:

1. **Checkpoint surgery engine.** YAML plans (`inputs` / `transforms` / `output`)
   that load safetensors or torch checkpoints, apply ordered tensor transforms
   (copy, move, split, concat, phlora, assert, ...), and write a new checkpoint.
   Flow: `compile_plan` -> `create_state_dict_provider` -> `SurgeryPlan.execute_pending`
   -> `save_output`. Code: `engine/`, `transforms/`, `expressions/`, `core/`, `cli/`, `web/`.
2. **Synapse / Axon.** A Haskell-flavored DSL (`.axon`) for transformer
   architectures plus a compiler that lowers it to a Graph IR and emits code for
   torch, triton, jax, mlx, tinygrad and vllm backends. Axon models are
   validated against HuggingFace Transformers for logits parity and speed.
   Code: `synapse/axon/` (compiler), `synapse/builtins/*.axon` (stdlib),
   `synapse/models/<family>/` (model definitions), `synapse/ops/` (primitives).

Axon compiler stages, in order: parse -> resolve -> normalize -> elaborate ->
flatten -> typecheck2 -> graph_ir -> optimize -> codegen2_<backend>.
Backend names used on the CLI and in plans: `runtime2-torch`, `codegen2-torch`,
`codegen2-triton`, `codegen2-jax`, `codegen2-mlx`, `codegen2-tinygrad`,
`codegen2-vllm`, and `hf` for the Transformers reference.

## Environment

- Requires Python >= 3.13. Upstream develops in a conda env named
  `brainsurgery` (see `validation/README.md` and `skills/*/SKILL.md`).
- Before running anything, check the interpreter actually has the deps:
  `python -c "import torch, brainsurgery"`. If that fails, report it and do not
  `pip install` into the system interpreter without asking.
- Model weights live under `models/` (gitignored). Test fixtures download from
  HuggingFace on first use via `tests/model_downloads.py`; `HF_TOKEN` is read
  if set. Expect network and disk use on a fresh checkout.
- GPU runs are optional. Most unit tests and all roundtrip tests are CPU-only.

## Commands

```bash
pip install -e ".[dev]"            # dev install (ruff, mypy, pre-commit, pytest-xdist, mlx)
pre-commit run --all-files         # ruff --fix, ruff-format, yaml/toml checks, mypy, pytest -q
ruff check brainsurgery tests      # lint (E, F, I; line length 100)
ruff format brainsurgery tests
mypy brainsurgery                  # config in pyproject; tests/examples excluded

pytest -q tests/test_cli.py                    # one file
pytest -q tests/test_agents_policy_guards.py   # architectural policy guards, run after synapse edits
pytest -q -n 8 --dist load tests/test_synapse_axon_graph_ir_roundtrip.py  # parallel roundtrips
pytest -q                                      # full suite: slow, downloads models

brainsurgery examples/gpt2.yaml                # run a surgery plan
brainsurgery -i plan.yaml                      # plan, then interactive REPL (YAML or OLY input)
brainsurgery synapse axon-test examples/gpt2.axon models/gpt2 --device cpu
brainsurgery synapse axon-stage-dump ...       # inspect a compiler stage
brainsurgery synapse axon-benchmark brainsurgery/synapse/models --axon-backend codegen2-torch \
  --log-dir log/<run-id> --stream-csv log/<run-id>/stream.csv
python scripts/benchmark_report_3tables.py log/<run-id>
```

Prefer `pytest -n 8 --dist load` for broad parametrized runs. Never use
`--dist loadfile` for per-model parametrized tests, it serializes them.

## Layout

| Path | Purpose |
|---|---|
| `brainsurgery/__init__.py` | Typer entry. Subcommands: `cli` (default), `synapse`, `webcli`, `webui` |
| `brainsurgery/core/` | Transform base classes, registry, `TensorRef`, structured path matching |
| `brainsurgery/engine/` | `SurgeryPlan`, execution loop, providers (`inmemory`, `arena`), checkpoint IO |
| `brainsurgery/transforms/` | One module per transform; each calls `register_transform` at import |
| `brainsurgery/expressions/` | `assert` operators (`equal`, `shape`, `all`, `not`, ...) |
| `brainsurgery/synapse/axon/` | Axon compiler; one subpackage per stage, `codegen2_*` per backend |
| `brainsurgery/synapse/builtins/` | Axon standard library (`Attention`, `MoE`, `SSM`, `Cache`, `NN`, ...) |
| `brainsurgery/synapse/models/` | `generic-<family>.axon` is the source of truth; other files are materialized |
| `examples/` | Standalone `.axon` models and the `gpt2.yaml` demo that exercises most transforms |
| `flexmore_examples/` | Paper examples: dense-to-MoE upcycling and PHLoRA compression with references |
| `validation/` | Original-vs-restored checkpoint inference and regression checks |
| `docs/` | Specs: `axon-spec.md`, `axon-grammar.ebnf`, `oly-spec.md`, `codebase-reference.md`, `interfaces-reference.md` |
| `wiki/` | Agent memory: benchmark conventions, vLLM debug log, sanity-check tables |
| `scripts/` | Agent-owned benchmark and roundtrip automation, documented in `wiki/scripts.md` |
| `skills/` | Repo-native skill docs (`benchmark`, `report`, `roundtrip`). Read the `SKILL.md` before those tasks |
| `tests/` | pytest suite; `conftest.py` and `model_downloads.py` manage model fixtures |

Path-scoped rules for `synapse/`, `tests/`, `docs/`+`wiki/`, and `scripts/`
are in `.claude/rules/` and load automatically when those files are touched.

## How to work here

- Read the nearest `AGENTS.md` before editing under `brainsurgery/synapse/`,
  `tests/`, `docs/`, `scripts/` or `wiki/`. They are policy, not suggestions.
- The hard rule: no model-family or HF-quirk branching in the parser,
  typechecker, lowering, codegen, runtime, `ops/`, or `builtins/*.axon`. Such
  logic belongs only in `synapse/axon_test.py`, `synapse/axon/tokenization.py`,
  and `transforms/infer_runtime.py`. `tests/test_agents_policy_guards.py`
  enforces this by grepping; run it after touching `synapse/`.
- Cross-subpackage imports go through the target subpackage's `__init__.py`
  and its `__all__`. Same-subpackage modules may import each other directly.
- Behavior changes in `brainsurgery/*` need user approval before landing.
  Fix failing tests by fixing tests, or ask before changing package code.
- Prefer the smallest generic fix. No compatibility shims, no silent fallbacks
  that hide path or type errors, no duplicated logic across builtins or models.
- Edit `generic-*.axon` files, then rematerialize; do not hand-edit
  materialized model files. Use `scripts/rematerialize_all_generic.sh`.
- Write run artifacts under `log/<run-id>/` (gitignored). Write scratch files
  under `tmp/` (gitignored) or the session scratchpad, never at repo root.
- Durable operational findings (a recurring failure class, a benchmark
  protocol change) go in `wiki/` with the metadata header from `wiki/AGENTS.md`.
  Dated events go in `wiki/log.md`.
- Keep `docs/` edits scoped to the requested change and copy-paste-safe.
- Commit only when asked. Branch off `main` first if committing.
