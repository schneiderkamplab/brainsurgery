"""T4: task-vector merge of two fine-tunes onto the OLMo-1B base (lambda = 0.4).

Plain safetensors + torch. Verifies the frozen-backbone precondition before
touching anything, merges the 48 MLP tensors against the unmodified base,
copies everything else from the base, and fails loudly on every required check.
"""
import json
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DIR = os.path.join(ROOT, "inputs", "base")
FT1 = os.path.join(ROOT, "inputs", "ft1", "model.safetensors")
FT2 = os.path.join(ROOT, "inputs", "ft2", "model.safetensors")
OUT = os.path.join(ROOT, "out", "T4", "model.safetensors")
LAMBDA = 0.4
MLP_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")
EXPECTED_MLP = 48
EXPECTED_TOTAL = 114


def fail(msg):
    raise SystemExit(f"ERROR: {msg}")


def load_base():
    with open(os.path.join(BASE_DIR, "model.safetensors.index.json")) as f:
        index = json.load(f)
    sd = {}
    for shard in sorted(set(index["weight_map"].values())):
        part = load_file(os.path.join(BASE_DIR, shard))
        if set(part) & set(sd):
            fail(f"duplicate keys across shards in {shard}")
        sd.update(part)
    if set(sd) != set(index["weight_map"]):
        fail("base shards do not match model.safetensors.index.json")
    return sd


def main():
    base, ft1, ft2 = load_base(), load_file(FT1), load_file(FT2)

    # Step 1: same names in all three, and every non-MLP tensor identical.
    names = set(base)
    if not (names == set(ft1) == set(ft2)):
        fail(
            "tensor name sets differ: "
            f"base^ft1={sorted(names ^ set(ft1))[:5]} base^ft2={sorted(names ^ set(ft2))[:5]}"
        )
    mlp = sorted(n for n in names if MLP_RE.match(n))
    if len(mlp) != EXPECTED_MLP:
        fail(f"expected {EXPECTED_MLP} MLP tensors, matched {len(mlp)}")
    layers = {int(MLP_RE.match(n).group(1)) for n in mlp}
    if layers != set(range(16)):
        fail(f"MLP layer indices unexpected: {sorted(layers)}")
    for n in sorted(names - set(mlp)):
        b, a1, a2 = base[n], ft1[n], ft2[n]
        for tag, t in (("ft1", a1), ("ft2", a2)):
            if t.shape != b.shape or t.dtype != b.dtype:
                fail(f"{n}: {tag} shape/dtype {tuple(t.shape)}/{t.dtype} != base {tuple(b.shape)}/{b.dtype}")
            if not torch.equal(b, t):
                fail(f"shared tensor {n} differs between base and {tag}")
    print(f"verified: {len(names)} names shared, {len(names) - len(mlp)} non-MLP tensors identical")

    # Step 2+3: merge MLP tensors against the unmodified base; copy the rest.
    out = {}
    merged = 0
    for n in sorted(names):
        b = base[n]
        if n in mlp:
            a1, a2 = ft1[n], ft2[n]
            if not (a1.shape == a2.shape == b.shape and a1.dtype == a2.dtype == b.dtype == torch.float32):
                fail(f"{n}: MLP tensor shape/dtype mismatch or not float32")
            out[n] = (b + LAMBDA * (a1 - b) + LAMBDA * (a2 - b)).contiguous()
            merged += 1
        else:
            out[n] = b.contiguous()

    # Required checks.
    if merged != EXPECTED_MLP:
        fail(f"merged {merged} tensors, expected {EXPECTED_MLP}")
    if len(out) != EXPECTED_TOTAL:
        fail(f"output has {len(out)} tensors, expected {EXPECTED_TOTAL}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    save_file(out, OUT, metadata={"format": "pt"})

    # Re-open and re-verify the written file.
    written = load_file(OUT)
    if len(written) != EXPECTED_TOTAL or set(written) != names:
        fail("written file does not contain exactly the 114 expected tensors")
    for n in names:
        if written[n].shape != base[n].shape or written[n].dtype != base[n].dtype:
            fail(f"written {n} shape/dtype changed")
        if n not in mlp and not torch.equal(written[n], base[n]):
            fail(f"written {n} should equal base bit-exactly")
    print(f"OK: merged {merged} MLP tensors, wrote {len(written)} tensors to {OUT}")


if __name__ == "__main__":
    sys.exit(main())
