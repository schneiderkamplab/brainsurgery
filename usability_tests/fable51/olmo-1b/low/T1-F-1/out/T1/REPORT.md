# T1 self-report

- Final artifact path: `out/T1/solution.py` (output `out/T1/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why: none
- Pitfalls or surprises you hit: none; renumbering is done by building a new dict with an explicit old->new map, so ordering collisions cannot occur, and a duplicate-key guard plus all required checks run before anything is written.
- Anything in the task text or documentation that was unclear: the "Required checks" list says "no tensor of blocks 12, 13, 14, 15 remains", which describes post-renumber indices rather than the dropped blocks 2, 6, 10, 14; I enforced both (no index >= 12, exactly 12 contiguous blocks, 86 tensors).
- Tools used (condition F): safetensors 0.5.3 (safe_open / save_file) and torch 2.14.0 for tensors. A plain script was the shortest and most transparent route; mergekit or torch-state-bridge would add config and a persistence step for no gain.
- Approximate time spent: about 2 minutes.
