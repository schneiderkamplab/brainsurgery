# Scripts Inventory

This file tracks maintained scripts in `../scripts/`.

## Entries

- `path`: `scripts/create_deepseek_v4_random.py`
- `purpose`: Create a deterministic synthetic DeepSeek V4 HF checkpoint for Axon/HF parity tests without downloading real DeepSeek V4 weights.
- `owner`: agents
- `inputs`: optional output directory, seed, storage dtype, and max shard size.
- `outputs`: HF-compatible model directory with `config.json`, `model.safetensors`, tokenizer files, `README.md`, and `random_checkpoint_summary.json`.
- `env`: run through the `brainsurgery` conda env so the installed `transformers.models.deepseek_v4` classes are available.
- `example`: `conda run --no-capture-output -n brainsurgery python scripts/create_deepseek_v4_random.py --output models/test/DeepSeek-V4-Test`
- `variants`: `test` is a small feature-complete decoder-stack target with sliding/HCA/CSA attention and hash/routed MoE.
- `notes`: the helper is also imported by `scripts/create_min4_family_test_models.py` so the full min4B test-checkpoint inventory can be regenerated from one script.
- `failure-modes`: requires the DeepSeek V4 Transformers implementation in the `brainsurgery` conda env.

- `path`: `scripts/create_min4_family_test_models.py`
- `purpose`: Create small random HF checkpoints under `models/test/` for generic Axon families whose real checkpoints are all above 4B parameters.
- `owner`: agents
- `inputs`: local source checkpoints, optional `--only` family-test names, seed, dtype, and output root.
- `outputs`: HF-compatible tiny test checkpoint directories with config, tokenizer files, `model.safetensors`, and `random_checkpoint_summary.json`.
- `env`: run through the `brainsurgery` conda env; some remote-code families may require optional HF modeling dependencies.
- `example`: `conda run --no-capture-output -n brainsurgery python scripts/create_min4_family_test_models.py --only Qwen3-MoE-Test`
- `failure-modes`: requires local source checkpoints under `models/`; families backed by remote-code may need source-specific config/key rewrites in this script to make the tiny checkpoint match the real checkpoint layout. `DeepSeek-V2-Test` removes stale `auto_map` remote-code hooks so HF and Axon both consume the same native DeepSeek-V2 safetensors.

- `path`: `scripts/axon_graph_ir_weak_roundtrip.py`
- `purpose`: Verify Graph IR can render back to canonical typed flat Axon after the initial full frontend pipeline.
- `owner`: agents
- `inputs`: Axon files or directories; optional `--main-module`.
- `outputs`: render generations under `tmp/axon-stage-roundtrip-graph-ir-weak` by default.
- `env`: run through the `brainsurgery` conda env.
- `example`: `conda run --no-capture-output -n brainsurgery python scripts/axon_graph_ir_weak_roundtrip.py brainsurgery/synapse/models/gpt2/gpt2-kv.axon`
- `notes`: defaults to signature/type-header rendering only; pass `--show-inferred-expression-types` to also render inferred body expression ascriptions.
- `failure-modes`: weak mode reparses/renormalizes/retypechecks rendered flat Axon without reresolving or reflattening, so failures usually point at graph-rendered Axon not satisfying the flat typed frontend contract.

- `path`: `scripts/axon_graph_ir_strong_roundtrip.py`
- `purpose`: Verify Graph IR rendered Axon is stable when every generation reruns resolve, normalize, elaborate, flatten, typecheck2, lower-to-graph, and graph-render.
- `owner`: agents
- `inputs`: Axon files or directories; optional `--main-module`.
- `outputs`: render generations under `tmp/axon-stage-roundtrip-graph-ir-strong` by default.
- `env`: run through the `brainsurgery` conda env.
- `example`: `conda run --no-capture-output -n brainsurgery python scripts/axon_graph_ir_strong_roundtrip.py brainsurgery/synapse/models/gpt2/gpt2-kv.axon`
- `notes`: defaults to signature/type-header rendering only; pass `--show-inferred-expression-types` to also render inferred body expression ascriptions.
- `failure-modes`: strong mode exposes resolver/frontend instability as well as graph-render instability; inspect `render1`, `render2`, and `render3` artifacts in the output directory.

## Entry Template

- `path`:
- `purpose`:
- `owner`:
- `inputs`:
- `outputs`:
- `env`:
- `example`:
- `failure-modes`:
