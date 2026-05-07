# Scripts Inventory

This file tracks maintained scripts in `../scripts/`.

## Entries

- `path`: `scripts/create_deepseek_v4_random.py`
- `purpose`: Create a deterministic synthetic DeepSeek V4 HF checkpoint for Axon/HF parity tests without downloading real DeepSeek V4 weights.
- `owner`: agents
- `inputs`: optional output directory, seed, storage dtype, and max shard size.
- `outputs`: HF-compatible model directory with `config.json`, safetensors shards/index, tokenizer files, `README.md`, and `random_checkpoint_summary.json`.
- `env`: run through the `brainsurgery` conda env so the installed `transformers.models.deepseek_v4` classes are available.
- `example`: `conda run --no-capture-output -n brainsurgery python scripts/create_deepseek_v4_random.py --output models/random/DeepSeek-V4-Random`
- `variants`: `random` is a zero-layer ~1B head/embedding parity target; `random2` is a small feature-complete decoder-stack target with sliding/HCA/CSA attention and hash/routed MoE.
- `failure-modes`: requires enough disk/RAM for about 1B parameters; use `--dtype bfloat16` storage by default to keep the checkpoint smaller.

## Entry Template

- `path`:
- `purpose`:
- `owner`:
- `inputs`:
- `outputs`:
- `env`:
- `example`:
- `failure-modes`:
