# T4 participant self-report

- **Final artifact path:** `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none; the single execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - The ordering hazard is real: both task vectors have to be taken against the untouched `base`, so I built a fresh `out` dict instead of mutating the base state dict in place.
  - `inputs/base/model.safetensors` is a symlink to a shared read-only GPT-2 checkpoint, so writing anywhere but `out/` would have failed anyway; the base tensors are `.clone()`d before saving to avoid handing safetensors views into the mmapped input.
  - GPT-2 keys carry no `transformer.` prefix and 160 (not 148) tensors means the per-layer `attn.bias` mask buffers are present; they fall outside the 48 MLP names and are copied verbatim.
  - `mlp.c_proj` vs `attn.c_proj` is an easy overmatch; I enumerated the 48 names explicitly (`h.<i>.mlp.c_{fc,proj}.{weight,bias}`) rather than pattern-matching on `c_proj`.
- **Anything in the task text or documentation that was unclear:** nothing material. The task text does not say whether the equality check in step 1 must be bit-exact or tolerant; I used bit-exact `torch.equal`, which is the stricter reading and held.
- **Tools used (condition F):** n/a (condition P: torch 2.14.0+cu130, safetensors 0.5.3).
- **Approximate time spent:** ~4 minutes.

## Checks implemented

1. Base has 160 tensors; `ft1`/`ft2` key sets identical to base; shapes and dtypes match across all three.
2. All 48 expected MLP names present and unique; MLP tensors are float32.
3. Every tensor outside the 48 MLP names is bit-identical in all three checkpoints (`torch.equal`), else abort.
4. Exactly 48 tensors merged, and the in-memory output has exactly 160 tensors.
5. Post-write read-back: the file on disk has 160 tensors, the same key set, matching shapes/dtypes, and the 112 non-MLP tensors are bit-identical to base.

Any failure raises `SystemExit("ERROR: ...")`.
