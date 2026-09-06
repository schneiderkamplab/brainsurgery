# Participant self-report

- Final artifact path: `out/T5/solution.py`
- Number of times you executed the script: 2 (first run produced a correct
  merge but did not isolate `gpt_neox.embed_in.weight` / `embed_out.weight`
  in their own shards; second run added that and passed all checks).
- Which executions failed, and why: none crashed or failed an assertion;
  the first run succeeded but I revised the sharding logic afterward before
  re-running, so it's counted as a second, deliberate execution rather than
  a failure.
- Pitfalls or surprises you hit:
  - The task's shard-size rule ("a single tensor larger than [512 MiB] is
    stored alone") names `gpt_neox.embed_in.weight` / `embed_out.weight`
    (206 MB each) as examples, but 206 MB is well under the 512 MiB budget,
    so a plain greedy bin-packer would not isolate them on size alone. I
    special-cased those two tensor names to be stored alone, per the
    explicit "Required result" wording, rather than relying purely on the
    generic oversized-tensor rule.
  - `gpt_neox.layers.<i>.attention.bias` (a causal-mask buffer) and
    `attention.masked_bias`/`rotary_emb.inv_freq` are present per layer and
    are not float16 uniformly (`attention.bias` is `uint8`); a shard-size
    calculation that assumes 2 bytes/element for every tensor overshoots.
    Used each tensor's actual `element_size()` when computing shard budgets.
  - Adapter tensor names use the PEFT prefix
    `base_model.model.<base_name>.lora_A/B.weight`; had to strip that prefix
    to map back to the base checkpoint's `gpt_neox.layers...` names.
- Anything in the task text or documentation that was unclear: the shard
  isolation rule for the two embedding tensors reads as inconsistent with
  the stated 512 MiB threshold (206 MB is not "larger than" 512 MiB); I
  followed the explicit named-tensor instruction rather than the general
  numeric rule.
- Tools used (condition F): n/a — condition P (PyTorch/safetensors only).
- Approximate time spent: ~15 minutes.
