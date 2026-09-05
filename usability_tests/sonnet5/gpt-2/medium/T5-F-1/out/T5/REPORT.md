# T5 participant self-report

- **Final artifact path:** `out/T5/solution.py` (invoked via `out/T5/run.sh`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none; the single execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - The delta from `B @ A` is in `nn.Linear` (`[out, in]`) layout and must be transposed before adding to the Conv1D (`[in, out]`) base weight — got this right on the first pass by reading the task's explicit note on `fan_in_fan_out`.
  - `wte.weight` (154 MB) exceeds the 100 MiB shard budget on its own and needed a special case (own shard) in the bin-packing loop rather than a plain greedy pack.
- **Anything in the task text or documentation that was unclear:** No — the spec fully pinned down the scale factor, transpose direction, shard budget, and required checks, so no ambiguity had to be resolved by guessing.
- **Tools used (condition F):** `torch` 2.14.0 and `safetensors` 0.5.3 only, via a plain script. Considered `peft.merge_and_unload`, but it requires instantiating a full HF model object and then re-deriving a sharded safetensors export from its state dict, plus reconciling PEFT's own key-remapping with the raw checkpoint keys given here; a direct script operating on the two safetensors files is simpler, avoids loading the model architecture at all, and makes the required pre-write checks (adapter-pair count, no leftover `lora_` names, shape, tensor count) explicit and easy to assert.
- **Approximate time spent, if you can tell:** ~10 minutes (script authoring + verification).
