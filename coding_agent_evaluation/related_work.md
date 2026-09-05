# Related systems and baselines

The systems below cover both direct checkpoint-rewriting competitors and adjacent
checkpoint, adaptation, and interpretability tooling. “Declarative” distinguishes
an external reusable specification from transformations expressed entirely in
host-language control flow.

| System | Primary focus | Declarative | Targeting | Validation | Export |
|---|---|:---:|---|---|:---:|
| PyTorch/`safetensors` scripts | General scripts | — | Manual | Manual | Manual |
| [MergeKit](https://github.com/arcee-ai/mergekit) | Merging and MoE construction | Yes | Layer slices and tensor filters | Limited | **Yes** |
| [LoRA](https://arxiv.org/abs/2106.09685)/adapter tooling | Adapters | Partial | Adapter-specific | Limited | Partial |
| [`safetensors`](https://github.com/huggingface/safetensors)/Hugging Face utilities | Storage and loading | — | — | — | **Yes** |
| [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html) | Distributed checkpoint I/O | — | State dictionaries and planners | Metadata/load checks | **Yes** |
| [Orbax](https://github.com/google/orbax) | JAX checkpointing and model surgery | Partial | PyTrees and partial restore | Restore/metadata checks | **Yes** |
| [`torch-state-bridge`](https://pypi.org/project/torch-state-bridge/) | State-dictionary key rewriting | Partial | Rules and regex captures | Preview, diff, collisions | Manual |
| [RYS](https://dnhkng.github.io/posts/rys/) / [LLM Circuit Finder](https://github.com/alainnothere/llm-circuit-finder) | Layer-path surgery | Partial | Explicit layer indices | Task probes | **Yes** |
| [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) | Interpretability | — | Model internals | Analysis | — |
| [nnsight](https://github.com/ndif-team/nnsight) | Tracing and interventions | — | Activations | Analysis | — |
| **BrainSurgery** | **General checkpoint rewriting** | **Yes** | **Regex and structured references** | **Assertions and checkpoint diff** | **Yes** |

## Scope and comparison notes

- **MergeKit is a substantive overlapping baseline.** Its YAML configurations,
  layer slicing, tensor-name filters, out-of-core execution, LoRA extraction,
  MoE construction, and multi-stage workflows overlap with several BrainSurgery
  demonstrations. It remains specialized around merging and related model-building
  workflows rather than exposing a general tensor-rewrite language with executable
  postconditions.
- **Orbax is an adjacent JAX baseline.** Its current V1 interface supports partial
  PyTree restoration and restore-time dtype, shape, and sharding changes. The older
  regex/value-function transformations API is deprecated, while the broader V1
  arbitrary model-surgery interface is documented as planned. Comparisons should
  identify the Orbax API version rather than treating all of these capabilities as
  simultaneously current.
- **`torch-state-bridge` is a direct but narrow rewriting baseline.** It provides
  rule-based and regex-capture key mappings, arithmetic substitutions, composable
  pipelines, previews, diffs, and collision detection. It transforms in-memory
  state dictionaries and leaves checkpoint persistence to the caller. At the time
  of this audit, its published artifact is version 0.1.0 and has limited project
  metadata, so experiments should pin the exact package version.
- **PyTorch Distributed Checkpoint is primarily an I/O and scaling baseline.** It
  supports parallel multi-rank save/load and load-time resharding. BrainSurgery can
  currently read and write DCP directories, but its adapter uses single-process
  `no_dist=True` operations over plain tensor state dictionaries. DCP therefore
  belongs in format/scaling comparisons, without implying parity for multi-rank
  execution, optimizer-state handling, or resharding.
- **RYS / LLM Circuit Finder is a motivating surgery use case, not a general-purpose
  framework baseline.** It specifies alternative layer execution paths and can
  physically duplicate selected GGUF layers, then evaluates the resulting model
  using downstream probes. The work is currently documented through the
  [RYS blog post](https://dnhkng.github.io/posts/rys/) and a software repository
  rather than a peer-reviewed publication; capability
  comparisons should avoid relying on unverified performance claims.

The most useful direct comparisons are: `torch-state-bridge` for key-renaming and
collision behavior; MergeKit for layer slicing, dtype conversion, LoRA extraction,
and MoE construction; and PyTorch Distributed Checkpoint for checkpoint I/O,
memory, and scaling. Orbax is informative for cross-framework positioning, while
RYS / LLM Circuit Finder demonstrates a concrete structural-surgery application.
