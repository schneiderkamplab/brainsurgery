import torch
from safetensors.torch import load_file, save_file

LAMBDA = 0.4

base = load_file("inputs/base/model.safetensors")
ft1 = load_file("inputs/ft1/model.safetensors")
ft2 = load_file("inputs/ft2/model.safetensors")

# Step 1: verify same tensor names across all three checkpoints.
names_base = set(base.keys())
names_ft1 = set(ft1.keys())
names_ft2 = set(ft2.keys())
if not (names_base == names_ft1 == names_ft2):
    raise RuntimeError(
        f"tensor name mismatch: base={len(names_base)}, ft1={len(names_ft1)}, "
        f"ft2={len(names_ft2)}; symmetric diffs: "
        f"base^ft1={names_base ^ names_ft1}, base^ft2={names_base ^ names_ft2}"
    )

# Determine the expected set of MLP tensors that were trained.
mlp_names = set()
for i in range(12):
    mlp_names.add(f"h.{i}.mlp.c_fc.weight")
    mlp_names.add(f"h.{i}.mlp.c_fc.bias")
    mlp_names.add(f"h.{i}.mlp.c_proj.weight")
    mlp_names.add(f"h.{i}.mlp.c_proj.bias")

if len(mlp_names) != 48:
    raise RuntimeError(f"expected 48 MLP tensors, computed {len(mlp_names)}")
if not mlp_names.issubset(names_base):
    raise RuntimeError("expected MLP tensor names not found in checkpoints")

# Verify every tensor outside the 48 MLP tensors is identical across all three.
other_names = names_base - mlp_names
mismatched = []
for name in other_names:
    b = base[name]
    f1 = ft1[name]
    f2 = ft2[name]
    if b.shape != f1.shape or b.shape != f2.shape:
        mismatched.append((name, "shape"))
        continue
    if not torch.equal(b, f1) or not torch.equal(b, f2):
        mismatched.append((name, "value"))

if mismatched:
    raise RuntimeError(
        f"non-MLP tensors differ across checkpoints (frozen-backbone assumption "
        f"violated): {mismatched}"
    )

# Step 2: merge the 48 MLP tensors via task arithmetic.
out = {}
merged_count = 0
for name in names_base:
    if name in mlp_names:
        b = base[name].to(torch.float32)
        f1 = ft1[name].to(torch.float32)
        f2 = ft2[name].to(torch.float32)
        merged = b + LAMBDA * (f1 - b) + LAMBDA * (f2 - b)
        out[name] = merged.to(base[name].dtype)
        merged_count += 1
    else:
        out[name] = base[name].clone()

if merged_count != 48:
    raise RuntimeError(f"expected to merge exactly 48 tensors, merged {merged_count}")

if len(out) != 160:
    raise RuntimeError(f"expected output to have exactly 160 tensors, got {len(out)}")

save_file(out, "out/T4/model.safetensors")
print(f"Wrote out/T4/model.safetensors with {len(out)} tensors, {merged_count} merged.")
