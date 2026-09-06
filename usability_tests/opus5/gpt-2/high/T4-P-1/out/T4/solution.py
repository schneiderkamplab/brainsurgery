"""T4: task-vector merge of two GPT-2 fine-tunes onto the base checkpoint.

out[X] = base[X] + lambda * (ft1[X] - base[X]) + lambda * (ft2[X] - base[X])
for the 48 MLP tensors; every other tensor is copied from the base verbatim.
Each task vector is taken against the *unmodified* base.
"""

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LAMBDA = 0.4
N_LAYERS = 12
N_TENSORS = 160
N_MLP = 48

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BASE = ROOT / "inputs" / "base" / "model.safetensors"
FT1 = ROOT / "inputs" / "ft1" / "model.safetensors"
FT2 = ROOT / "inputs" / "ft2" / "model.safetensors"
OUT = ROOT / "out" / "T4" / "model.safetensors"

MLP_NAMES = [
    f"h.{i}.mlp.{mod}.{kind}"
    for i in range(N_LAYERS)
    for mod in ("c_fc", "c_proj")
    for kind in ("weight", "bias")
]


def load(path):
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return {k: f.get_tensor(k) for k in f.keys()}


def main():
    base = load(BASE)
    ft1 = load(FT1)
    ft2 = load(FT2)

    # --- step 1: the three checkpoints must agree on names, and on every
    # tensor outside the MLP set, before anything is computed. ---
    if not (set(base) == set(ft1) == set(ft2)):
        only_base = sorted(set(base) - (set(ft1) & set(ft2)))
        extra = sorted((set(ft1) | set(ft2)) - set(base))
        raise SystemExit(
            f"tensor names differ across checkpoints: "
            f"missing from a fine-tune {only_base[:5]}, unexpected {extra[:5]}"
        )
    if len(base) != N_TENSORS:
        raise SystemExit(f"expected {N_TENSORS} tensors in the base, got {len(base)}")

    missing = [n for n in MLP_NAMES if n not in base]
    if missing:
        raise SystemExit(f"MLP tensors absent from the checkpoints: {missing}")
    if len(MLP_NAMES) != N_MLP:
        raise SystemExit(f"expected {N_MLP} MLP names, built {len(MLP_NAMES)}")

    mlp = set(MLP_NAMES)
    shared_differ = []
    for name in sorted(base):
        b, a1, a2 = base[name], ft1[name], ft2[name]
        if b.shape != a1.shape or b.shape != a2.shape:
            raise SystemExit(
                f"shape mismatch for {name}: base {tuple(b.shape)}, "
                f"ft1 {tuple(a1.shape)}, ft2 {tuple(a2.shape)}"
            )
        if b.dtype != a1.dtype or b.dtype != a2.dtype:
            raise SystemExit(
                f"dtype mismatch for {name}: base {b.dtype}, ft1 {a1.dtype}, ft2 {a2.dtype}"
            )
        if name in mlp:
            continue
        # bit-exact byte comparison; torch.equal would report NaN != NaN
        raw = b.contiguous().numpy().tobytes()
        if raw != a1.contiguous().numpy().tobytes() or raw != a2.contiguous().numpy().tobytes():
            shared_differ.append(name)
    if shared_differ:
        raise SystemExit(
            f"{len(shared_differ)} non-MLP tensor(s) differ between the checkpoints, "
            f"the frozen-backbone assumption does not hold: {shared_differ[:10]}"
        )

    # --- step 2: merge, always against the untouched base ---
    out = dict(base)
    merged = 0
    for name in MLP_NAMES:
        b = base[name].to(torch.float32)
        if b.dtype != torch.float32 or base[name].dtype != torch.float32:
            raise SystemExit(f"{name} is not float32: {base[name].dtype}")
        tv1 = ft1[name].to(torch.float32) - b
        tv2 = ft2[name].to(torch.float32) - b
        out[name] = (b + LAMBDA * tv1 + LAMBDA * tv2).contiguous()
        merged += 1

    if merged != N_MLP:
        raise SystemExit(f"merged {merged} tensors, expected exactly {N_MLP}")
    if len(out) != N_TENSORS:
        raise SystemExit(f"output has {len(out)} tensors, expected {N_TENSORS}")

    for name in out:
        if out[name].shape != base[name].shape or out[name].dtype != base[name].dtype:
            raise SystemExit(f"output tensor {name} changed shape or dtype")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous().clone() for k, v in out.items()}, str(OUT),
              metadata={"format": "pt"})

    with safe_open(str(OUT), framework="pt", device="cpu") as f:
        written = list(f.keys())
    if len(written) != N_TENSORS:
        raise SystemExit(f"wrote {len(written)} tensors to {OUT}, expected {N_TENSORS}")
    if set(written) != set(base):
        raise SystemExit("written tensor names do not match the base")

    print(f"verified {N_TENSORS - N_MLP} shared tensors identical across base/ft1/ft2")
    print(f"merged {merged} MLP tensors with lambda={LAMBDA}")
    print(f"wrote {len(written)} tensors to {OUT}")


if __name__ == "__main__":
    main()
