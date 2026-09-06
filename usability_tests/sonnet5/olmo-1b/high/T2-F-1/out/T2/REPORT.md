# Participant self-report — T2

- **Final artifact path:** `out/T2/solution.py` (produces `out/T2/model.safetensors`)
- **Number of times you executed the script or plan:** 2
- **Which executions failed, and why (one line each):**
  1. `FileNotFoundError` — wrong relative-path depth to `inputs/base` (three
     `.parents[]` levels instead of two) computed from the script's own
     location under `out/T2/`.
- **Pitfalls or surprises you hit (one line each):**
  - The task gives the exact row/column ranges to keep (`0..639`, `768..2047`),
    so there was no need to derive the removed span from `head_index *
    head_dim`; used the given ranges directly and cross-checked they equal
    `head 5 * 128 .. head 5 * 128 + 128` removed.
  - Considered `transformers`' `PreTrainedModel.prune_heads`, but that API
    reloads/re-derives shapes from the model's `config.num_attention_heads`
    and is built around models that implement `_prune_heads`; going straight
    at the safetensors shards with `torch.cat` slicing is more direct, avoids
    depending on OLMo having pruning support wired up, and keeps every other
    tensor byte-identical by construction (only the four per-layer
    head-bearing tensors are ever touched).
- **Anything in the task text or documentation that was unclear:** No.
- **Tools used (condition F): name, version, and why:**
  - `safetensors` 0.5.3 — read each input shard directly via `safe_open`
    (avoids materializing a full HF model) and write the single merged output
    shard via `save_file`.
  - `torch` 2.14.0 — `torch.cat` to drop the head-5 row/column block from
    each of `q_proj`/`k_proj`/`v_proj`/`o_proj` per layer; `.contiguous()`
    before saving since safetensors requires contiguous tensors.
  - Not used: `transformers.prune_heads` (see pitfall above), `mergekit`
    (built for merging/slicing whole layers across checkpoints, not
    intra-tensor head slicing), `peft` (no LoRA adapter involved in this
    task), `torch-state-bridge` (renames/rewrites keys; this task never
    changes tensor names, only tensor contents on 4 of them per layer).
- **Approximate time spent, if you can tell:** ~10 minutes.
