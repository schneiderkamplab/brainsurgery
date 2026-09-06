# T5 report

## Participant self-report

**Tools used:** plain script on top of `safetensors` + `torch` only. No
`peft`, `mergekit`, or `torch-state-bridge`.

**Why:** `peft.merge_and_unload` needs an instantiated `PeftModel` wrapping
the base `nn.Module`, which is more machinery than the task needs — the
merge here is a pure tensor operation (`weight += scale * B @ A`) on named
keys read straight out of the two safetensors files. `mergekit`'s task-
arithmetic recipes are built around merging multiple full checkpoints, not
folding a low-rank adapter into specific submodules, and would need a
LoRA-expansion preprocessing step to fit this shape. `torch-state-bridge`'s
regex/rule rewriting is more useful for restructuring key namespaces than
for adapter math. A ~150-line script had full, auditable control over
naming (regex-matching `base_model.model.gpt_neox.layers.<i>.<module>.lora_{A,B}.weight`
against `gpt_neox.layers.<i>.<module>.weight`), scale (`lora_alpha / r`),
dtype (float32 accumulate, cast back to float16), and the sharding rule
(greedy bin-packing to a 512 MiB tensor-data budget, oversized tensors
alone in their own shard) — plus let the required checks be plain asserts
placed right before the tensors are written, so a `weight_map`/shape/count
mismatch fails loudly instead of writing a bad checkpoint.

**Required checks enforced (asserts in `solution.py`, run before any write):**

- exactly 16 adapter pairs found (`lora_A`/`lora_B` matched into
  `(layer, module)` pairs, `assert len(pairs) == 16`);
- no `lora_` tensor name in the merged dict before writing;
- `gpt_neox.layers.0.attention.query_key_value.weight` shape is
  `[6144, 2048]`;
- output has exactly 244 tensors.

**Verification beyond the required checks:** after running, independently
reloaded `out/T5` and the original base/adapter files and confirmed: exactly
16 tensors differ from the base, all 16 match `base + scale * (B @ A)` cast
to float16 bit-for-bit, dtype/shape are correct, and no shard exceeds
536,870,912 bytes.

**Attempts:** 1 execution, succeeded on first run.
