# Targeting `torch.fx` from Synapse YAML

Targeting `torch.fx` is feasible, but it is not a small switch. In this codebase, Synapse currently has two execution targets:

- An interpreter that walks the normalized Synapse graph and executes each op module's `interpret(...)` against an `env` dict and a raw `_state` tensor map in [`runtime.py`](/work/training/brainsurgery/brainsurgery/synapse/runtime.py#L21) and [`runtime.py`](/work/training/brainsurgery/brainsurgery/synapse/runtime.py#L284).
- A Python source emitter that lowers the same graph into handwritten PyTorch code by calling each op module's `compile(...)` method, which returns source lines, in [`codegen.py`](/work/training/brainsurgery/brainsurgery/synapse/codegen.py#L26) and [`codegen.py`](/work/training/brainsurgery/brainsurgery/synapse/codegen.py#L620).

The main thing it would take is adding a third lowering target, not replacing "plain PyTorch" with tracing. You do not want to `symbolic_trace` the current runtime model. The runtime uses dynamic Python dictionaries, list/tuple/dict-valued env entries, and control flow in ways that are hostile to FX tracing. The right approach is: build an FX graph directly from Synapse YAML.

## What Has To Change

- The op interface would need an FX lowering hook. Today every op module is required to export `interpret`, `compile`, and `uses_node_path` in [`ops/__init__.py`](/work/training/brainsurgery/brainsurgery/synapse/ops/__init__.py#L9). You would need something like `compile_fx(...)` or `lower_fx(...)` alongside those.
- Parameter handling must change. Today parameters live in `self._state`, a plain `dict[str, Tensor]`, and ops resolve paths dynamically, for example linear in [`linear.py`](/work/training/brainsurgery/brainsurgery/synapse/ops/linear.py#L110). FX wants values to come from placeholders, constants, or `get_attr` on module attributes. So an FX backend needs a real backing module that registers tensors as buffers/parameters under stable attribute names.
- Graph construction must replace string emission. The current codegen walks graph items, handles `for`, block `call`, nested scopes, and output recording in [`codegen.py`](/work/training/brainsurgery/brainsurgery/synapse/codegen.py#L620). An FX builder would need the same traversal, but instead of emitting Python strings it would create `fx.Graph` nodes and track `env: dict[str, fx.Node]`.
- Each op needs explicit FX lowering. Simple ops like `linear`, `add`, `mul`, `reshape`, `concat` are straightforward. More complex ops like `attention` in [`attention.py`](/work/training/brainsurgery/brainsurgery/synapse/ops/attention.py#L117) may need helper functions or tiny submodules as `call_function` or `call_module` targets.
- Non-tensor values need a policy. Synapse env entries are `Any`, not just tensors, and runtime output recording already handles tensors, lists, tuples, and dicts in [`runtime.py`](/work/training/brainsurgery/brainsurgery/synapse/runtime.py#L458). FX can represent tuples and some Python literals, but arbitrary dict-heavy flow is more awkward. You would probably want to define "FX-safe Synapse" and reject or specially lower ops that produce unsupported container patterns.
- CLI/API plumbing is missing. The only emit command today writes `.py` source via [`cli/synapse.py`](/work/training/brainsurgery/brainsurgery/cli/synapse.py#L55). You would need something like `emit-fx`, or `emit --target fx`, and likely an in-memory API that returns `fx.GraphModule`.

## Biggest Real Blockers

- `self._state` as an unregistered tensor dictionary.
- Per-op lowering being source-string based instead of IR-node based.
- Some Synapse constructs are compile-time/static-friendly (`for`, block calls), but some runtime behaviors are not ideal FX surface area.
- `generate()` should stay outside the FX core graph; it is a Python decoding loop in both runtime and codegen today, not something you should try to express as FX.

## Pragmatic Implementation Plan

1. Add `SynapseFXModuleBuilder` that consumes normalized Synapse spec and produces `torch.fx.GraphModule`.
2. Register checkpoint tensors as attributes on a backing `nn.Module`, with a reversible mapping from Synapse param paths to attribute names.
3. Add `lower_fx(...)` to ops, starting with the easy tensor-only ops.
4. Reuse the existing graph traversal semantics from interpreter/codegen for `for`, block calls, scopes, and outputs.
5. Keep unsupported ops on the interpreter path until coverage is good enough.
6. Add parity tests against `SynapseProgramModel.from_spec(...)` for a few existing example specs.

## Effort

Rough effort: a minimal prototype for tensor-only models is probably a few days. Full parity with the current Synapse surface, especially attention/cache/custom ops, is more like a moderate project.
