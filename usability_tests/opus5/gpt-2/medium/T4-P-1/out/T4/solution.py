"""T4: task-vector merge of two GPT-2 fine-tunes onto the base checkpoint.

out[X] = base[X] + lambda*(ft1[X] - base[X]) + lambda*(ft2[X] - base[X])
for the 48 MLP tensors; every other tensor is copied from the base verbatim.
Each task vector is taken against the *unmodified* base.
"""

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

LAMBDA = 0.4
N_LAYERS = 12
N_MLP = 48
N_TOTAL = 160

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent          # sandbox root
IN = ROOT / "inputs"
OUT = HERE / "model.safetensors"

MLP_SUFFIXES = ("mlp.c_fc.weight", "mlp.c_fc.bias", "mlp.c_proj.weight", "mlp.c_proj.bias")
MLP_KEYS = [f"h.{i}.{s}" for i in range(N_LAYERS) for s in MLP_SUFFIXES]


def die(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


base = load_file(IN / "base" / "model.safetensors")
ft1 = load_file(IN / "ft1" / "model.safetensors")
ft2 = load_file(IN / "ft2" / "model.safetensors")

# --- step 1: the three checkpoints must agree on names, and on everything
# outside the 48 MLP tensors, before anything is touched. ---
if len(base) != N_TOTAL:
    die(f"base has {len(base)} tensors, expected {N_TOTAL}")
for name, sd in (("ft1", ft1), ("ft2", ft2)):
    if set(sd) != set(base):
        missing = sorted(set(base) - set(sd))
        extra = sorted(set(sd) - set(base))
        die(f"{name} key set differs from base; missing={missing[:5]} extra={extra[:5]}")

missing_mlp = [k for k in MLP_KEYS if k not in base]
if missing_mlp:
    die(f"expected MLP tensors absent from the checkpoints: {missing_mlp}")
mlp_set = set(MLP_KEYS)
if len(mlp_set) != N_MLP:
    die(f"MLP key list has {len(mlp_set)} unique names, expected {N_MLP}")

for k in sorted(base):
    for name, sd in (("ft1", ft1), ("ft2", ft2)):
        if sd[k].shape != base[k].shape or sd[k].dtype != base[k].dtype:
            die(
                f"{name}[{k}] has shape/dtype {tuple(sd[k].shape)}/{sd[k].dtype}, "
                f"base has {tuple(base[k].shape)}/{base[k].dtype}"
            )
    if k in mlp_set:
        continue
    for name, sd in (("ft1", ft1), ("ft2", ft2)):
        if not torch.equal(sd[k], base[k]):
            die(f"non-MLP tensor {k} differs between base and {name}; frozen-backbone assumption violated")

for k in MLP_KEYS:
    if base[k].dtype != torch.float32:
        die(f"{k} is {base[k].dtype}, expected float32")

# --- step 2/3: merge ---
out: dict[str, torch.Tensor] = {}
merged = 0
for k in sorted(base):
    if k in mlp_set:
        b = base[k].to(torch.float32)
        out[k] = (b + LAMBDA * (ft1[k].to(torch.float32) - b) + LAMBDA * (ft2[k].to(torch.float32) - b)).contiguous()
        merged += 1
    else:
        out[k] = base[k].clone()

if merged != N_MLP:
    die(f"merged {merged} tensors, expected {N_MLP}")
if len(out) != N_TOTAL:
    die(f"output has {len(out)} tensors, expected {N_TOTAL}")

OUT.parent.mkdir(parents=True, exist_ok=True)
save_file(out, str(OUT))

# --- read back and confirm what landed on disk ---
check = load_file(OUT)
if len(check) != N_TOTAL:
    die(f"written file has {len(check)} tensors, expected {N_TOTAL}")
if set(check) != set(base):
    die("written file key set differs from base")
for k in sorted(base):
    if check[k].shape != base[k].shape or check[k].dtype != base[k].dtype:
        die(f"written {k} shape/dtype mismatch")
    if k not in mlp_set and not torch.equal(check[k], base[k]):
        die(f"written non-MLP tensor {k} is not bit-identical to base")

print(f"merged {merged} MLP tensors with lambda={LAMBDA}; wrote {len(check)} tensors to {OUT}")
