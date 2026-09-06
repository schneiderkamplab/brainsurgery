# T5 — participant self-report

- **Final artifact path:** `out/T5/solution.py` (run as
  `.venv/bin/python out/T5/solution.py` from the sandbox root); output in
  `out/T5/model-0000{1..10}-of-00010.safetensors` + `model.safetensors.index.json`.
- **Number of times you executed the script or plan:** 2
- **Which executions failed, and why:** neither execution failed. Execution 1
  produced a correct, fully verified output, but the script cleaned the output
  directory with `shutil.rmtree(out/T5)` and so deleted its own source file
  (`out/T5/solution.py`) on the way — the run itself still finished because
  Python had already loaded the module. Execution 2 is the same script with the
  cleanup narrowed to `model-*.safetensors` + `model.safetensors.index.json`, so
  authored files under `out/T5/` survive a re-run.
- **Pitfalls or surprises you hit:**
  - PEFT name prefix: adapter keys carry `base_model.model.` on top of the base
    name, so the mapping is `base_model.model.<base-minus-.weight>.lora_{A,B}.weight`;
    I derived the base name by regex instead of hardcoding the layer/module list.
  - Scaling is `lora_alpha / r` read from `adapter_config.json` (= 2 here), not a
    constant, and `fan_in_fan_out` decides whether `B @ A` needs a transpose; the
    script reads both and refuses rather than guessing if `fan_in_fan_out` is true.
  - The sharding note in TASK.md says `model.embed_tokens.weight` and
    `lm_head.weight` (412 MB each) are "larger than" the 512 MiB budget and must be
    stored alone; 412 MB is actually below 536,870,912 bytes, so a plain greedy
    packer would *not* isolate them. I followed the explicit instruction and gave
    each of those two tensors its own shard, then greedy-packed the rest.
  - OLMo-1B has no parametric norm weights, so the 112 non-embedding tensors are
    exactly 16 × 256 MiB layers and pack into 8 shards of exactly 512 MiB.
  - Self-inflicted: the task requires authored files and the output to share
    `out/T5/`, so a wipe-and-recreate of the output directory destroys the
    solution script. Cleanup has to be scoped to the checkpoint files.
  - The base input is sharded at ~5 GB per file, so I streamed tensors per output
    shard rather than materialising the whole 5 GB state dict.
- **Anything in the task text or documentation that was unclear:** only the
  sharding sentence above — "a single tensor larger than that (here
  `model.embed_tokens.weight` and `lm_head.weight`, 412 MB each)" is
  self-contradictory, since 412 MB < 512 MiB. Resolved in favour of the explicit
  naming of those two tensors.
- **Tools used (condition F):**
  - `safetensors` 0.5.3 — lazy `safe_open` reads from the base shards and
    `save_file` for the output. Chosen because the task is pure file surgery: it
    lets me touch one tensor at a time and control shard membership exactly, which
    the 512 MiB budget + "big tensor alone" rule requires.
  - `torch` 2.14.0 — float32 `B @ A` matmul and the Frobenius-norm verification.
  - `peft` 0.20.0 — **considered and not used.** `merge_and_unload` is the
    advertised route, but it needs `AutoModelForCausalLM.from_pretrained` on a
    5 GB float32 checkpoint, and `save_pretrained(max_shard_size=...)` uses a greedy
    packer whose shard layout does not match the "embed/lm_head alone" rule. Its
    `adapter_config.json` (`r`, `lora_alpha`, `fan_in_fan_out`) is the only piece I
    needed, and I read that as JSON.
  - `mergekit` — not applicable: it does task arithmetic / layer slicing over full
    models, not LoRA factor folding at the checkpoint level.
- **Checks enforced by the run** (`fail()` raises `SystemExit`, so the run aborts):
  before writing — exactly 32 complete adapter pairs, every adapter target present
  in the base, planned key set == base key set and 114 tensors, no `lora_` name,
  `model.layers.0.self_attn.q_proj.weight` is `[2048, 2048]`, every multi-tensor
  shard ≤ 536,870,912 bytes. After writing, `verify_output()` re-reads the shards
  from disk and re-runs all of those against the actual files, plus per-tensor
  relative Frobenius error ≤ 1e-6 for all 32 merged weights (and that each differs
  from the base), and bit-equality for sampled untouched tensors.
- **Approximate time spent:** ~8 minutes, of which each run itself is ~10 s.
