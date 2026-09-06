# T4 participant self-report

- **Final artifact path:** `out/T4/solution.py` (runner: `out/T4/run.sh`); output `out/T4/model.safetensors`
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none; the first execution succeeded.
- **Pitfalls or surprises you hit:**
  - The base is sharded, so the key set and the per-name shard owner have to come from `model.safetensors.index.json`, while the two fine-tunes are single files — I wrote one small shard-view class so all three are read the same way.
  - The ordering hazard is real but easy to avoid by never mutating the running tensor: both task vectors are computed against the freshly-read `base[X]` inside a single expression, not by applying one merge and then the other.
  - Bit-exactness of the 66 untouched tensors is only guaranteed if they are copied straight from the base rather than round-tripped through arithmetic, so the non-MLP branch does no math at all.
  - Memory: three float32 ~5 GB checkpoints do not need to be resident; `safe_open` reads one tensor name at a time, and the run peaked well under the box's limits (13 s wall clock).
- **Anything in the task text or documentation that was unclear:** nothing material. Minor: the task says "a single file `out/T4/model.safetensors`" while grading compares the directory `out/T4`, so I left the script and report alongside it; that seems intended.
- **Tools used (condition F):**
  - `safetensors` 0.5.3 — lazy per-tensor read (`safe_open`) of the sharded base and the two fine-tunes, and `save_file` for the single-file output. Chosen because the required checks are per-tensor comparisons across three checkpoints, which is exactly what a lazy key/tensor API gives.
  - `torch` 2.14.0 — float32 arithmetic and `torch.equal` for the bit-exact shared-tensor verification.
  - Deliberately **not** mergekit, although `task_arithmetic` is the nominal condition-F route for this task: it applies its merge method to every tensor it sees rather than a verified 48-tensor subset, it offers no precondition check that base/ft1/ft2 agree outside the trained tensors (the check the task makes mandatory), and its output is a re-serialized sharded model, which puts the bit-exactness of the 66 untouched tensors and the "single file, 114 tensors" requirement at the mercy of its writer. Enforcing the required checks would have meant writing a verification script on top of it anyway, at which point the ~110-line direct script is both shorter and auditable.
  - Checks enforced by the run (each aborts with a non-zero exit): identical key sets across the three checkpoints; 114 tensors total; the MLP pattern matches exactly 48 names; shape/dtype agreement per tensor; every non-MLP tensor bit-identical in all three (`torch.equal`); exactly 48 tensors merged; the written file re-opened and confirmed to hold 114 tensors with the base key set.
- **Approximate time spent:** ~10 minutes, of which 13 s was the run itself.
