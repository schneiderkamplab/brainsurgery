"""T4: task-vector merge of two GPT-2 fine-tunes (lambda = 0.4).

Plain torch + safetensors. Every required check raises on failure.
"""

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

LAMBDA = 0.4
N_LAYERS = 12
IN = Path("inputs")
OUT = Path("out/T4/model.safetensors")

MLP_KEYS = [
    f"h.{i}.mlp.{mod}.{suffix}"
    for i in range(N_LAYERS)
    for mod in ("c_fc", "c_proj")
    for suffix in ("weight", "bias")
]


def main() -> None:
    base = load_file(IN / "base" / "model.safetensors")
    ft1 = load_file(IN / "ft1" / "model.safetensors")
    ft2 = load_file(IN / "ft2" / "model.safetensors")

    # --- step 1: identical key sets ---
    for name, sd in (("ft1", ft1), ("ft2", ft2)):
        if set(sd) != set(base):
            missing = sorted(set(base) - set(sd))
            extra = sorted(set(sd) - set(base))
            raise SystemExit(
                f"{name} key set differs from base: missing={missing[:5]} extra={extra[:5]}"
            )

    mlp = set(MLP_KEYS)
    if len(mlp) != 48:
        raise SystemExit(f"expected 48 MLP key names, built {len(mlp)}")
    unknown = sorted(mlp - set(base))
    if unknown:
        raise SystemExit(f"MLP keys absent from the base checkpoint: {unknown}")

    # --- step 1: everything outside the MLP tensors is bit-identical in all three ---
    for key in sorted(set(base) - mlp):
        b = base[key]
        for name, sd in (("ft1", ft1), ("ft2", ft2)):
            t = sd[key]
            if t.shape != b.shape or t.dtype != b.dtype:
                raise SystemExit(
                    f"{key}: {name} has shape/dtype {tuple(t.shape)}/{t.dtype}, "
                    f"base has {tuple(b.shape)}/{b.dtype}"
                )
            if not torch.equal(t, b):
                raise SystemExit(f"{key}: {name} differs from base outside the MLP tensors")

    # shapes/dtypes of the MLP tensors must match too, or the arithmetic is meaningless
    for key in sorted(mlp):
        b = base[key]
        for name, sd in (("ft1", ft1), ("ft2", ft2)):
            t = sd[key]
            if t.shape != b.shape or t.dtype != b.dtype:
                raise SystemExit(
                    f"{key}: {name} has shape/dtype {tuple(t.shape)}/{t.dtype}, "
                    f"base has {tuple(b.shape)}/{b.dtype}"
                )

    # --- step 2/3: merge, always against the unmodified base ---
    out: dict[str, torch.Tensor] = {}
    merged = 0
    for key, b in base.items():
        if key in mlp:
            b32 = b.to(torch.float32)
            tv1 = ft1[key].to(torch.float32) - b32
            tv2 = ft2[key].to(torch.float32) - b32
            out[key] = (b32 + LAMBDA * tv1 + LAMBDA * tv2).to(b.dtype).contiguous()
            merged += 1
        else:
            out[key] = b.clone().contiguous()

    if merged != 48:
        raise SystemExit(f"merged {merged} tensors, expected exactly 48")
    if len(out) != 160:
        raise SystemExit(f"output has {len(out)} tensors, expected exactly 160")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT))

    # --- post-write verification of the file that was actually produced ---
    back = load_file(OUT)
    if len(back) != 160:
        raise SystemExit(f"written file has {len(back)} tensors, expected 160")
    if set(back) != set(base):
        raise SystemExit("written file key set differs from base")
    changed = sum(1 for k in back if not torch.equal(back[k], base[k]))
    for key in set(base) - mlp:
        if not torch.equal(back[key], base[key]):
            raise SystemExit(f"{key}: non-MLP tensor was modified")
    print(f"wrote {OUT}: {len(back)} tensors, {merged} merged, {changed} differ from base")


if __name__ == "__main__":
    sys.exit(main())
