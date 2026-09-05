# Participant self-report

- Final artifact path: `out/T5/solution.py`
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - None crashed or failed an assertion; run 1 produced a technically-passing
    but suspect layout (`model.embed_tokens.weight` and `lm_head.weight`
    shared shards with smaller tensors instead of sitting alone), caught by
    manual inspection rather than a script failure, so I fixed the shard
    packing and re-ran.
- Pitfalls or surprises you hit (one line each):
  - The task says the two 412 MB tensors are "stored alone in its own
    shard", but 412 MB is actually smaller than the 512 MiB budget, so a
    plain first-fit-decreasing bin packer happily packs a couple of small
    attention tensors alongside them; had to add an explicit "tensor over
    half the shard budget gets a dedicated, sealed shard" rule to match the
    stated requirement.
  - Easy to forget that a naive size-based bin packer can also let *later*
    small tensors reopen and fill a shard that was meant to stay
    single-tensor; needed a `shard_open` flag per shard, not just a size
    check, once a shard is sealed.
  - `fan_in_fan_out = false` plus PEFT's `nn.Linear` weight convention meant
    no transposition was needed for `B @ A`, but it was worth asserting this
    explicitly rather than assuming it.
- Anything in the task text or documentation that was unclear:
  - The "412 MB each ... stored alone" claim isn't literally implied by the
    stated 512 MiB per-shard budget (412 MB < 512 MiB), so satisfying it
    requires inferring an implicit "oversized tensors don't share shards"
    rule beyond a simple total-bytes cap. Spelling out the exact rule (e.g.
    a size threshold relative to the shard budget) would remove the
    ambiguity.
- Tools used (condition F): n/a (condition P)
- Approximate time spent, if you can tell: ~15 minutes
