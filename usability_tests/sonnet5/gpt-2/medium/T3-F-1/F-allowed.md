# Condition F: allowed repositories and packages

Derived from the paper's related-systems table (checkpoint-rewriting
competitors plus adjacent checkpoint, adaptation and interpretability
tooling), restricted to what is pip-installable and can act on the inputs of
these tasks. Everything below is pre-installed in the condition-F environment
at the exact versions in `requirements-F.txt`; nothing else may be installed.
Use any of them, in any combination, including plain scripts on top of them.

| Package | Version | What it is for here |
|---|---|---|
| `torch` | 2.14.0 | tensors; also PyTorch Distributed Checkpoint (`torch.distributed.checkpoint`) for checkpoint I/O |
| `safetensors` | 0.5.3 | safetensors load/save |
| `numpy` | 2.5.2 | arrays |
| `transformers` | 5.12.1 | HuggingFace model loading from the input directories (config and tokenizer files are present), `prune_heads`, `save_pretrained` with a dtype, sharded export |
| `peft` | 0.20.0 | LoRA adapters: loading `adapter_config.json`/`adapter_model.safetensors`, `merge_and_unload` |
| `mergekit` | 0.1.4 | YAML merge configurations: layer slicing (passthrough), task arithmetic, dtype conversion, sharded output |
| `torch-state-bridge` | 0.1.0 | rule-based and regex-capture key rewriting on state dictionaries, previews, diffs, collision detection; persistence is up to you |
| `accelerate`, `huggingface_hub` | pinned | loading and download helpers used by the packages above |

Not included, and why: Orbax (JAX-only; its regex/value-function surgery
API is deprecated), TransformerLens and nnsight (activation-level analysis,
no checkpoint export), RYS / LLM Circuit Finder (a GGUF layer-path
application, not a package), BrainSurgery (the tool under test, condition B).

Every task has at least one plausible route through this list: T1 via
mergekit layer slicing or torch-state-bridge key rewriting, T2 via
transformers `prune_heads`, T3 via transformers dtype export, T4 via mergekit
task arithmetic, T5 via peft `merge_and_unload`. Whether those routes are
actually faster or safer than a script is exactly what condition F measures.
