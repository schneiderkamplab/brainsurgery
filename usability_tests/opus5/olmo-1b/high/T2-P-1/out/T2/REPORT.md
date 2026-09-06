# Participant self-report — T2 (structured attention-head pruning, OLMo-1B-0724-hf)

- **Final artifact path:** `out/T2/solution.py` (output: `out/T2/model.safetensors`, 114 tensors)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none; the single execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - The input is sharded over two safetensors files, so the state dict had to be
    reassembled from `model.safetensors.index.json` before any slicing; the output
    is a single flat file, so the index is deliberately not reproduced.
  - The two axes differ: q/k/v are `[out, in]` with heads as row blocks, o_proj is
    also `[out, in]` but the head axis is the *input* side, so it is a column cut —
    slicing dim 0 on o_proj would have silently produced a wrong-but-loadable file.
  - `torch.Tensor.narrow` returns views; I ran the two kept pieces through
    `torch.cat(...).contiguous()` so safetensors would not reject shared storage.
  - OLMo has no per-layer norm parameters, so 114 = 16 layers x 7 tensors +
    `embed_tokens` + `lm_head`; I checked that count rather than assuming norms exist.
- **Anything in the task text or documentation that was unclear:**
  - The task did not say whether the output directory should also carry
    `config.json` (with `num_attention_heads` dropped to 15) for the result to be
    "loadable as the same architecture with 15 heads". Grading is described as
    comparing tensors only, so I wrote just `model.safetensors`.
  - It was not stated whether the output should stay sharded; "a single file
    `out/T2/model.safetensors`" settled it, but the input being sharded made the
    question worth asking.
- **Tools used (condition F):** n/a — condition P (torch 2.14.0, safetensors 0.5.3).
- **Approximate time spent, if you can tell:** roughly 5 minutes: inspecting the
  index and config, writing the script, one clean run.
