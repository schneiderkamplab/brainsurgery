# T5 — LoRA adapter merge with sharded export (Pythia-1B), condition F

## Participant self-report

- **Final artifact path:** `out/T5/solution.py` (run as
  `.venv/bin/python out/T5/solution.py` from the sandbox root).

- **Number of times you executed the script or plan:** 2

- **Which executions failed, and why (one line each):**
  - Neither execution failed. Execution 1 produced a correct, fully verified
    output but deleted its own source file (see pitfalls); execution 2 is the
    same run with that cleanup bug fixed, and reproduced the identical output.

- **Pitfalls or surprises you hit (one line each):**
  - `shutil.rmtree(OUT_DIR)` to clear stale shards deleted `out/T5/solution.py`
    along with them, because the task requires the artifact to live in the same
    directory as the output; replaced with a targeted unlink of `*.safetensors`
    plus the index file.
  - The obvious condition-F route, `peft.merge_and_unload()` +
    `save_pretrained()`, is a trap here: the base has 16 `attention.bias`,
    16 `attention.masked_bias` and 16 `rotary_emb.inv_freq` buffers, and
    transformers 5.12 no longer keeps those in GPTNeoX's persistent state, so a
    model round-trip would emit 196 tensors instead of 244. Checked the key
    inventory before choosing a route rather than after.
  - `attention.bias` is `uint8`, not a float mask, so any byte accounting that
    assumes 2 or 4 bytes per element overstates the checkpoint by 192 MiB and
    would produce the wrong number of shards.
  - `safetensors.torch.load_file` returns keys in sorted order, not in the
    file's header order (the header groups by dtype). Since greedy sharding is
    order-sensitive, this decides the shard boundaries; I used the `load_file`
    order as the canonical one.
  - Shard budget interpretation: the limit is on tensor payload, not file size.
    Shard 1 holds 529,608,772 bytes of tensor data (under the 536,870,912 limit)
    but the file on disk is 529,611,484 bytes. Checking file sizes instead of
    payload would be wrong in the other direction near the boundary.

- **Anything in the task text or documentation that was unclear:**
  - TASK.md states that `gpt_neox.embed_in.weight` and `embed_out.weight` are
    "larger than" 512 MiB and must each be stored alone in a shard, but both are
    50304×2048 float16 = 206,045,184 bytes, well under the limit. No tensor in
    this checkpoint exceeds 512 MiB, so the sole-tensor-shard rule never fires.
    I implemented it defensively anyway (a shard with more than one tensor is
    asserted to contain no oversized tensor) and let greedy packing place the
    embeddings, which puts both in shard 1 alongside 22 small tensors.
  - `adapter_config.json` lists `target_modules: ["query_key_value"]`, whereas
    TASK.md says `["attention.query_key_value"]`. I did not match on
    `target_modules` at all — I derived the target from the adapter tensor names
    themselves and asserted the count is exactly 16 — so the discrepancy is moot,
    but a solution that filtered base keys by the config string would be
    sensitive to which of the two spellings it trusted.
  - The task does not say whether the output directory should also carry
    `config.json` / tokenizer files. Since the required key set is exactly the
    244 base tensors and grading described only the checkpoint, I wrote only the
    shards and the index.

- **Tools used (condition F): name, version, and why:**
  - `safetensors` 0.5.3 — `load_file` / `save_file`. Direct file-level access is
    what keeps the 228 untouched tensors bit-exact and preserves the exotic
    buffers a model round-trip would drop.
  - `torch` 2.14.0+cu130 — the `B @ A` product in float32, the cast back to
    float16, and `torch.equal` / `torch.linalg.norm` in the verification pass.
  - `huggingface_hub` 1.16.1 — `split_torch_state_dict_into_shards` with
    `max_shard_size=512*1024*1024`. Used deliberately rather than hand-rolling
    the greedy packer, so the shard boundaries and the
    `model-0000i-of-0000n.safetensors` / `model.safetensors.index.json`
    conventions are the canonical HuggingFace ones rather than my guess at them.
  - Python stdlib `json` for `adapter_config.json` and the index.
  - **Considered and rejected:** `peft` 0.20.0 `merge_and_unload` and
    `transformers` 5.12.1 `save_pretrained`, for the buffer-dropping reason
    above; also, going through a model would upcast/redispatch tensors and put
    the required bit-exactness on the 228 unchanged tensors at risk.
    `mergekit` 0.1.4 has no LoRA-fold operation that maps onto this task.

- **Approximate time spent, if you can tell:** ~10 minutes. Roughly half of it
  was inspecting the inputs (key inventory, dtypes, key ordering) before writing
  anything, which is what surfaced the buffer and uint8 issues up front.

## How the required checks are enforced

All four required checks abort the run before any file is written, as
`CheckFailed` (an `AssertionError` subclass), and are then re-asserted against
the re-read output:

| Required check | Where |
|---|---|
| exactly 16 adapter pairs found and merged | pair count after grouping, plus a `merged == 16` counter after the merge loop, plus `changed == 16` tensors differing from the base in the re-read output |
| no output tensor name contains `lora_` | scan of the merged state dict before writing, repeated on the re-read shards |
| `gpt_neox.layers.0.attention.query_key_value.weight` is `[6144, 2048]` | before writing and after re-reading |
| the output has exactly 244 tensors | `len(base) == 244` before writing; `len(seen) == 244` across the re-read shards |

Beyond the required four, the script also asserts: each pair is complete;
factor ranks agree with `r`; the delta's shape matches the base tensor; shape
and dtype survive both the merge and the write; every shard's tensor payload is
within 512 MiB; no multi-tensor shard contains an oversized tensor; no tensor
appears in two shards; `weight_map` agrees with the shards in both directions;
all 244 written tensors are byte-identical to the merged state dict; and the
merge is independently re-derived from `inputs/` with a per-tensor relative
Frobenius error of at most 1e-3.

## Result of the final run

```
base tensors: 244  adapter tensors: 32
r=16 alpha=32.0 scale=2.0 fan_in_fan_out=False
merged 16 adapter pairs
wrote 4 shards + index to out/T5/
  model-00001-of-00004.safetensors: 24 tensors, 529608772 bytes
  model-00002-of-00004.safetensors: 75 tensors, 524554570 bytes
  model-00003-of-00004.safetensors: 75 tensors, 524554570 bytes
  model-00004-of-00004.safetensors: 70 tensors, 511955272 bytes
verified: 16 tensors merged, 228 bit-identical to the base
OK
```

`model.safetensors.index.json` maps all 244 tensor names to their shard and
records `total_size = 2090673184`, the sum of the four payloads above.
