# Wiki Log

## [2026-04-20] init | bootstrap llmwiki structure

- Added `wiki/AGENTS.md`, `wiki/index.md`, `wiki/log.md`, and `wiki/scripts.md`.
- Established cross-references with root `AGENTS.md`.

## [2026-05-09] test checkpoints | regenerated min4B family fixtures

- Regenerated all 31 `models/test/*-Test` checkpoints in float32 with `scripts/create_min4_family_test_models.py`.
- Regenerated `models/test/DeepSeek-V4-Test` with `scripts/create_deepseek_v4_random.py`.
- `DeepSeek-V2-Test` now preserves the real DeepSeek-V2-Lite `q_lora_rank=null` / `q_proj` attention weight layout.
- `Phi-MoE-Test` now materializes locally via native Transformers `phimoe` plus checkpoint-key/config normalization for Axon parity.
