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

## Entry Template

- `path`:
- `purpose`:
- `owner`:
- `inputs`:
- `outputs`:
- `env`:
- `example`:
- `failure-modes`:
