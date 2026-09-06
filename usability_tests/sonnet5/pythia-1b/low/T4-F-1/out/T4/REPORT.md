# Participant self-report — T4-F-1

Tools used: plain Python script on `torch` and `safetensors` (both from the
allowed list). I did not use `mergekit`'s task-arithmetic YAML config because
its planner is built around HuggingFace model directories and its own dtype/
sharding conventions, and expressing "verify non-MLP tensors are bit-identical
across three checkpoints, then abort loudly if not" is not something its
config schema exposes directly — it would still need a custom pre-check
script wrapped around it. A direct script gave full control over the
precondition check and the exact float32-compute/float16-cast arithmetic with
no more code than a mergekit YAML + wrapper would need.

What it does:
1. Loads all three `model.safetensors` files with `safetensors.torch.load_file`.
2. Verifies the three checkpoints have identical tensor name sets, aborting
   otherwise.
3. Verifies every tensor outside the 64 known MLP tensor names
   (`gpt_neox.layers.<i>.mlp.{dense_h_to_4h,dense_4h_to_h}.{weight,bias}` for
   `i` in 0..15) is shape-, dtype- and value-identical across base/ft1/ft2,
   aborting otherwise.
4. For each of the 64 MLP tensors, computes
   `base + 0.4*(ft1-base) + 0.4*(ft2-base)` in float32 (each task vector taken
   against the unmodified base, so ordering is not a hazard), casts back to
   float16.
5. Copies all other 180 tensors unchanged from base.
6. Asserts exactly 64 tensors were merged and the output has exactly 244
   tensors before writing `out/T4/model.safetensors`.

Attempts: 1 (first execution succeeded).

Issues encountered: none.
