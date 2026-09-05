## Participant self-report

- Final artifact path: `out/T5/plan.yaml`
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: `crash` — the first two "count" assertions used a
    self-quoted regex (`of: model::'.*'`), which the YAML/tensor-ref parser
    treated literally, so `count.of` matched zero tensors and raised
    `TransformError`.
- Pitfalls or surprises you hit (one line each):
  - `matmul`/`add_`/other multi-tensor transforms don't independently
    regex-match `from_b`/`to` against their own tensors — `from_a`'s regex is
    matched once per tensor, then `from_b` and `to` are applied as
    `re.sub` *replacement templates* against that same match. So `from_b`
    must be written as `h.\1.lora_A.weight` (a template with a backreference),
    not as its own independent capture-group regex — writing it as a second
    `(\d+)` pattern raises `bad escape \d` because `\d` isn't a valid
    replacement-string escape.
  - Confirmed empirically (small throwaway plan) that `output` writes only
    the state-dict alias that transforms actually write into: since every
    `to:` here targets `model::...` and the adapter is only ever read from
    `lora::...`, the LoRA tensors never need an explicit delete/prefix-removal
    to be excluded from the output.
  - Intermediate `tmp.h.<i>.delta` / `tmp.h.<i>.deltaT` tensors do need
    explicit `delete` before the run ends — they live under the same `model`
    alias as the real output tensors, so anything left there would ship in
    the sharded checkpoint (and would also fail the required "exactly 160
    tensors" check).
  - `output.shard: 100MB` is interpreted as 100 MiB (binary multiplier,
    1024^2), which matches the task's 104,857,600-byte cap exactly, and
    `wte.weight` (154 MB) is correctly placed alone in its own shard by the
    library's greedy shard-packing.
- Anything in the task text or documentation that was unclear:
  - None; the `re.sub`-as-template semantics for `from_b`/`to` in ternary
    transforms like `matmul` aren't spelled out in `help.txt` beyond the
    single-capture-group examples, so it took one failing run to learn the
    exact mechanism.
- Tools used (condition F): name, version, and why: n/a (condition B)
- Approximate time spent, if you can tell: ~15 minutes, most of it reading
  `help.txt`/README and one throwaway experiment plan to confirm `matmul`
  pairing and output-alias semantics before writing the final plan.
