# T1 self-report (condition P)

- **Final artifact path:** `out/T1/solution.py` (output: `out/T1/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none; the single execution succeeded.
- **Pitfalls or surprises you hit:**
  - Renumbering collision hazard: renaming in place (e.g. `h.3` -> `h.2` while `h.2` still exists) would silently overwrite a surviving block, so I built a fresh dict keyed by new name and asserted no destination key was written twice.
  - The block regex has to be anchored on `^h\.(\d+)\.` with escaped dots; a loose pattern would also touch the 4 non-block tensors.
  - Each block owns 13 tensors including the `attn.bias` causal-mask buffer, which is easy to forget when counting (12*13 + 4 = 160 in, 9*13 + 4 = 121 out).
  - Checks run before any write, so a failure leaves no partial output.
  - I additionally verified the saved file round-trips bit-exactly and that each output tensor is the identical object from its source key, so no value/shape/dtype could have changed.
- **Anything in the task text or documentation that was unclear:** nothing material. The Conv1D `[in, out]` note is background only; this task is a pure rename and needed no transposes.
- **Tools used (condition F):** n/a (condition P: PyTorch 2.14.0 + safetensors 0.5.3).
- **Approximate time spent:** ~3 minutes.
