"""T4: task-vector merge of two fine-tunes (Pythia-1B), plain safetensors + torch."""
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "inputs/base/model.safetensors"
FT1 = ROOT / "inputs/ft1/model.safetensors"
FT2 = ROOT / "inputs/ft2/model.safetensors"
OUT = ROOT / "out/T4/model.safetensors"

LAMBDA = 0.4
N_LAYERS = 16
EXPECTED_TOTAL = 244
MLP_RE = re.compile(
    r"^gpt_neox\.layers\.(\d+)\.mlp\.(dense_h_to_4h|dense_4h_to_h)\.(weight|bias)$"
)


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    base, ft1, ft2 = load_file(BASE), load_file(FT1), load_file(FT2)

    # Step 1: same tensor names in all three.
    if not (base.keys() == ft1.keys() == ft2.keys()):
        fail(
            "tensor name sets differ: "
            f"base^ft1={sorted(set(base) ^ set(ft1))[:5]} "
            f"base^ft2={sorted(set(base) ^ set(ft2))[:5]}"
        )
    if len(base) != EXPECTED_TOTAL:
        fail(f"expected {EXPECTED_TOTAL} tensors, got {len(base)}")

    mlp_names = {k for k in base if MLP_RE.match(k) and int(MLP_RE.match(k).group(1)) < N_LAYERS}
    if len(mlp_names) != 64:
        fail(f"expected 64 MLP tensors, matched {len(mlp_names)}")

    # Step 1 (cont.): every non-MLP tensor is bit-identical across the three.
    for k in base:
        for name, other in (("ft1", ft1), ("ft2", ft2)):
            if base[k].shape != other[k].shape or base[k].dtype != other[k].dtype:
                fail(f"{k}: shape/dtype mismatch base vs {name}")
        if k in mlp_names:
            continue
        for name, other in (("ft1", ft1), ("ft2", ft2)):
            if not torch.equal(base[k], other[k]):
                fail(f"shared tensor {k} differs between base and {name}")
    print(f"verified: {len(base)} names match, {len(base) - len(mlp_names)} shared tensors identical")

    # Step 2/3: merge in float32 against the unmodified base; everything else from base.
    out: dict[str, torch.Tensor] = {}
    merged = 0
    for k, b in base.items():
        if k in mlp_names:
            b32 = b.float()
            m = b32 + LAMBDA * (ft1[k].float() - b32) + LAMBDA * (ft2[k].float() - b32)
            out[k] = m.to(b.dtype).contiguous()
            merged += 1
        else:
            out[k] = b.contiguous()

    if merged != 64:
        fail(f"merged {merged} tensors, expected 64")
    if len(out) != EXPECTED_TOTAL:
        fail(f"output has {len(out)} tensors, expected {EXPECTED_TOTAL}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT), metadata={"format": "pt"})

    # Post-write sanity: reload and recheck count.
    n = len(load_file(OUT))
    if n != EXPECTED_TOTAL:
        fail(f"written file has {n} tensors, expected {EXPECTED_TOTAL}")
    print(f"wrote {OUT} with {n} tensors ({merged} merged, lambda={LAMBDA})")


if __name__ == "__main__":
    main()
