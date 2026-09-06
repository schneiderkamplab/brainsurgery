# Participant self-report: T1 (GPT-2 124M, condition P)

- Final artifact path: `out/T1/solution.py` (output: `out/T1/model.safetensors`, 121 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Renumbering must go in ascending original order (or into a fresh dict) so a shifted block never overwrites a survivor; I built a new dict and assert on collision.
  - `attn.bias` is a mask buffer, not a bias vector; the block regex treats every `h.<i>.*` name uniformly so it is carried along without special-casing.
  - Regex anchored with `^h\.(\d+)\.` so `wte`/`wpe`/`ln_f` are untouched and dots are escaped.
- Anything in the task text or documentation that was unclear: the "Required checks" bullet says "no tensor of blocks 9, 10, 11 remains", which reads oddly since those are surviving blocks before renumbering; I interpreted it as "no output block index >= 9" after renumbering.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 2 minutes
