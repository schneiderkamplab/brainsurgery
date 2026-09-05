## Participant self-report

- Final artifact path: `out/T3/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the plan passed on the first execution.
- Pitfalls or surprises you hit (one line each):
  - The causal-mask buffer `h.<i>.attn.bias` is easy to catch with a naive
    substring check but not with a proper regex anchor, since
    `attn.c_attn.bias` also contains the substring `attn.bias`; the plan
    itself uses a full-match regex (`h\.\d+\.attn\.bias`) via `delete`, which
    doesn't have this ambiguity, so it wasn't a real risk here, just something
    to watch when spot-checking the result with ad hoc string matching.
  - `output.shard` sizes are parsed as binary units (`MB` = 1024^2 bytes), so
    `shard: 64MB` gives exactly the 67,108,864-byte (64 MiB) budget the task
    asks for.
- Anything in the task text or documentation that was unclear: none; the
  README's shard-size unit (`500MB`/`5GB` examples) doesn't state whether it's
  decimal or binary, so I checked `output_paths.py` in the installed package
  to confirm `MB` means 1024^2 before trusting the 64 MiB budget.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: ~15 minutes.
