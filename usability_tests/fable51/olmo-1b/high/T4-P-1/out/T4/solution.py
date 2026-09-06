"""T4: task-vector merge of two fine-tunes onto OLMo-1B-0724-hf (float32).

out[X] = base[X] + lam * (ft1[X] - base[X]) + lam * (ft2[X] - base[X])
for the 48 MLP tensors; every other tensor is copied from the base unchanged.
"""

import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "inputs" / "base"
FT1_FILE = ROOT / "inputs" / "ft1" / "model.safetensors"
FT2_FILE = ROOT / "inputs" / "ft2" / "model.safetensors"
OUT_FILE = ROOT / "out" / "T4" / "model.safetensors"

LAMBDA = 0.4
N_LAYERS = 16
MLP_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")
EXPECTED_MLP = 3 * N_LAYERS  # 48
EXPECTED_TOTAL = 114


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


class Checkpoint:
    """Lazy reader over one or more safetensors files."""

    def __init__(self, files: list[Path]):
        self.handles = [safe_open(str(f), framework="pt", device="cpu") for f in files]
        self.where: dict[str, int] = {}
        for i, h in enumerate(self.handles):
            for k in h.keys():
                if k in self.where:
                    fail(f"duplicate tensor {k!r} across shards {files}")
                self.where[k] = i

    def keys(self) -> set[str]:
        return set(self.where)

    def get(self, name: str) -> torch.Tensor:
        return self.handles[self.where[name]].get_tensor(name)


def open_base() -> Checkpoint:
    index = json.loads((BASE_DIR / "model.safetensors.index.json").read_text())
    shards = sorted({BASE_DIR / v for v in index["weight_map"].values()})
    return Checkpoint(shards)


def main() -> None:
    base = open_base()
    ft1 = Checkpoint([FT1_FILE])
    ft2 = Checkpoint([FT2_FILE])

    # ---- step 1: verify names and shared (non-MLP) tensors before touching anything
    kb, k1, k2 = base.keys(), ft1.keys(), ft2.keys()
    if not (kb == k1 == kb == k2):
        fail(
            "tensor name sets differ: "
            f"base-ft1={sorted(kb ^ k1)[:5]} base-ft2={sorted(kb ^ k2)[:5]}"
        )
    names = sorted(kb)
    if len(names) != EXPECTED_TOTAL:
        fail(f"expected {EXPECTED_TOTAL} tensors in base, found {len(names)}")

    mlp_names = [n for n in names if MLP_RE.match(n)]
    if len(mlp_names) != EXPECTED_MLP:
        fail(f"expected {EXPECTED_MLP} MLP tensors, matched {len(mlp_names)}: {mlp_names}")
    layers = {int(MLP_RE.match(n).group(1)) for n in mlp_names}
    if layers != set(range(N_LAYERS)):
        fail(f"MLP tensors cover layers {sorted(layers)}, expected 0..{N_LAYERS - 1}")
    mlp_set = set(mlp_names)

    shared = [n for n in names if n not in mlp_set]
    for n in shared:
        b, a1, a2 = base.get(n), ft1.get(n), ft2.get(n)
        for tag, t in (("ft1", a1), ("ft2", a2)):
            if t.shape != b.shape or t.dtype != b.dtype:
                fail(f"{n}: {tag} shape/dtype {tuple(t.shape)}/{t.dtype} "
                     f"!= base {tuple(b.shape)}/{b.dtype}")
            if not torch.equal(b, t):
                fail(f"{n}: {tag} differs from base but is outside the MLP set")
    print(f"verified {len(shared)} shared tensors identical across base/ft1/ft2")

    # ---- step 2/3: merge MLP tensors against the unmodified base, copy the rest
    out: dict[str, torch.Tensor] = {}
    merged = 0
    for n in names:
        b = base.get(n)
        if n in mlp_set:
            a1, a2 = ft1.get(n), ft2.get(n)
            for tag, t in (("ft1", a1), ("ft2", a2)):
                if t.shape != b.shape or t.dtype != b.dtype:
                    fail(f"{n}: {tag} shape/dtype mismatch vs base")
            if b.dtype != torch.float32:
                fail(f"{n}: expected float32, got {b.dtype}")
            # Both task vectors are taken against the same, unmodified base tensor `b`.
            merged_t = b + LAMBDA * (a1 - b) + LAMBDA * (a2 - b)
            out[n] = merged_t.to(torch.float32).contiguous()
            merged += 1
        else:
            out[n] = b.contiguous()

    if merged != EXPECTED_MLP:
        fail(f"merged {merged} tensors, expected {EXPECTED_MLP}")
    if len(out) != EXPECTED_TOTAL:
        fail(f"output has {len(out)} tensors, expected {EXPECTED_TOTAL}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT_FILE), metadata={"format": "pt"})

    # ---- post-write check on the file actually written
    with safe_open(str(OUT_FILE), framework="pt", device="cpu") as f:
        written = list(f.keys())
    if len(written) != EXPECTED_TOTAL or set(written) != set(names):
        fail(f"written file has {len(written)} tensors / wrong names")
    print(f"merged {merged} MLP tensors (lambda={LAMBDA}), wrote {len(written)} tensors to {OUT_FILE}")


if __name__ == "__main__":
    main()
