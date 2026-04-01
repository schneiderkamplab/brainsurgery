# Synapse & Axon Guide

Synapse is a declarative model definition language for brainsurgery. It lets you describe neural network architectures as data (YAML graphs) and generate runnable model code for multiple backends.

## The Three Layers

```
Axon (high-level DSL)  -->  Synapse (YAML graph spec)  -->  Backend (PyTorch / TinyGrad)
```

- **Axon** (`.axon`) -- a terse, readable DSL for writing model architectures by hand.
- **Synapse** (`.yaml`) -- a lowered YAML representation of the computational graph. Machines read and write this.
- **Backend** -- compiles a synapse spec into executable Python. Two backends ship today:

| Backend | Output | Class style |
|---------|--------|-------------|
| `pytorch` (default) | `nn.Module` subclass | `class Model(nn.Module):` |
| `tinygrad` | Plain class with `Tensor` ops | `class Model:` |

## CLI

```bash
# Generate PyTorch code from a synapse spec
brainsurgery synapse emit examples/gpt2_synapse.yaml gpt2.py --class-name GPT2

# Generate TinyGrad code from the same spec
brainsurgery synapse emit examples/gpt2_synapse.yaml gpt2_tiny.py --class-name GPT2 --backend tinygrad

# Lower an Axon file to a Synapse YAML spec
brainsurgery synapse axon-to-synapse examples/gpt2.axon gpt2_synapse.yaml

# Render a Synapse spec back to Axon source
brainsurgery synapse synapse-to-axon examples/gpt2_synapse.yaml gpt2.axon --module-name gpt2
```

## Programmatic API

```python
from brainsurgery.synapse import emit_model_code_from_synapse_spec
from omegaconf import OmegaConf

# Load a spec
spec = OmegaConf.to_container(OmegaConf.load("examples/gpt2_synapse.yaml"), resolve=True)

# Emit PyTorch source (default)
pytorch_src = emit_model_code_from_synapse_spec(spec, class_name="GPT2", backend="pytorch")

# Emit TinyGrad source from the same spec
tinygrad_src = emit_model_code_from_synapse_spec(spec, class_name="GPT2", backend="tinygrad")
```

## Round-Trip: Axon <-> Synapse

Axon and Synapse are interchangeable representations. You can lower Axon to Synapse and render Synapse back to Axon:

```
axon file  --(axon-to-synapse)-->  synapse YAML
synapse YAML  --(synapse-to-axon)-->  axon file
```

Symbols (dimensions like `D`, `H`, `L`) are resolved during Axon-to-Synapse lowering, so the Synapse YAML contains concrete values. The reverse (Synapse-to-Axon) re-emits readable source with the resolved constants inlined.

## Synapse Spec Anatomy

A minimal spec:

```yaml
synapse: 1
model:
  symbols:
    D: 768
    H: 12
    L: 12
    V: 50257
  inputs:
    input_ids: {optional: false}
    attn_mask: {optional: true}
  graph:
    - n_op_1:
        _op: embedding
        _args: input_ids
        _bind: tok
        dim: D
        _params:
          weight: wte.weight
    # ... more nodes ...
  outputs:
    logits: logits
  blocks:
    gpt2_block:
      inputs: {x: {optional: false}}
      graph: [...]
      outputs: {out_0: out_0}
```

Key concepts:
- **symbols** -- named integer/float constants (dimensions, layer counts).
- **inputs** -- model entry points with optional flags.
- **graph** -- ordered list of op nodes forming the forward pass.
- **outputs** -- mapping from output names to graph variable references.
- **blocks** -- reusable sub-graphs invoked via `_op: call`.

Each node in the graph is a single-key mapping `{name: {spec}}` where the spec contains:
- `_op` -- the operation name (`embedding`, `linear`, `add`, `attention`, etc.)
- `_args` -- input variable references (string or list of strings)
- `_bind` -- output variable binding (string or list of strings)
- `_params` -- weight parameter path mappings (resolved against a state dict at runtime)

## Example Models

The `examples/` directory contains specs for real architectures:

| File | Architecture |
|------|-------------|
| `gpt2_synapse.yaml` | GPT-2 (117M) |
| `gpt2_kv_synapse.yaml` | GPT-2 with KV-cache |
| `gemma3_270m_synapse.yaml` | Gemma 3 270M |
| `llama3_2_1b_synapse.yaml` | Llama 3.2 1B |
| `mistral_7b_v0_1_synapse.yaml` | Mistral 7B v0.1 |
| `qwen2_5_0_5b_synapse.yaml` | Qwen 2.5 0.5B |
| `falcon_rw_1b_synapse.yaml` | Falcon RW 1B |
| `olmoe_1b_7b_0924_synapse.yaml` | OLMoE 1B-7B |

Each has a matching `.axon` file (human-authored) and `.yaml` (lowered from Axon).

## Backend Architecture

```
synapse/backends/
  __init__.py          -- registry, emit_model_code_from_synapse_spec()
  base.py              -- BaseEmitter (shared graph-walking logic)
  pytorch/
    emitter.py         -- PyTorchEmitter  -->  nn.Module source
    op_map.yaml        -- PyTorch op dispatch table
  tinygrad/
    emitter.py         -- TinyGradEmitter -->  plain class source
    op_map.yaml        -- TinyGrad op dispatch table
    ops/               -- per-op compile() functions (auto-discovered)
```

Both backends share the same graph-walking engine (`BaseEmitter`). Each backend overrides `render()` for class scaffolding and `_compile_op()` for op-specific code generation.

The TinyGrad backend uses a dedicated ops directory (`synapse/backends/tinygrad/ops/`) with per-op modules that each expose a `compile()` function emitting TinyGrad-specific Python source lines.
