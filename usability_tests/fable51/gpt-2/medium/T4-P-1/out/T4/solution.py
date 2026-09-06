"""T4: task-vector merge of two GPT-2 fine-tunes (lambda = 0.4) into base."""
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
LAMBDA = 0.4
N_LAYERS = 12
MLP_RE = re.compile(r"^h\.(\d+)\.mlp\.(c_fc|c_proj)\.(weight|bias)$")
EXPECTED_MLP = {
    f"h.{i}.mlp.{m}.{p}" for i in range(N_LAYERS) for m in ("c_fc", "c_proj") for p in ("weight", "bias")
}


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    base = load_file(ROOT / "inputs/base/model.safetensors")
    ft1 = load_file(ROOT / "inputs/ft1/model.safetensors")
    ft2 = load_file(ROOT / "inputs/ft2/model.safetensors")

    # Step 1: same names in all three, shared tensors identical.
    names = set(base)
    if not (names == set(ft1) == set(ft2)):
        fail(
            "tensor name sets differ: base-only=%s ft1-only=%s ft2-only=%s"
            % (sorted(names - set(ft1) - set(ft2)), sorted(set(ft1) - names), sorted(set(ft2) - names))
        )
    if len(names) != 160:
        fail(f"expected 160 tensors, found {len(names)}")

    mlp_names = {n for n in names if MLP_RE.match(n)}
    if mlp_names != EXPECTED_MLP:
        fail(f"MLP tensor set mismatch: {sorted(mlp_names ^ EXPECTED_MLP)}")

    bad = []
    for n in sorted(names - mlp_names):
        b, a, c = base[n], ft1[n], ft2[n]
        for tag, t in (("ft1", a), ("ft2", c)):
            if t.shape != b.shape or t.dtype != b.dtype or not torch.equal(t, b):
                bad.append(f"{n} differs in {tag}")
    if bad:
        fail("shared tensors are not identical across checkpoints:\n  " + "\n  ".join(bad))

    # Step 2/3: merge MLP tensors against the unmodified base; copy the rest.
    out = {}
    merged = 0
    for n in names:
        b = base[n]
        if n in mlp_names:
            for tag, t in (("ft1", ft1[n]), ("ft2", ft2[n])):
                if t.shape != b.shape or t.dtype != b.dtype:
                    fail(f"{n}: shape/dtype mismatch in {tag}: {tuple(t.shape)} {t.dtype}")
            if b.dtype != torch.float32:
                fail(f"{n}: expected float32, got {b.dtype}")
            out[n] = (b + LAMBDA * (ft1[n] - b) + LAMBDA * (ft2[n] - b)).to(torch.float32).contiguous()
            merged += 1
        else:
            out[n] = b.contiguous()

    if merged != 48:
        fail(f"expected to merge 48 tensors, merged {merged}")
    if len(out) != 160:
        fail(f"expected 160 output tensors, got {len(out)}")

    dst = ROOT / "out/T4/model.safetensors"
    dst.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(dst))

    check = load_file(dst)
    if len(check) != 160:
        fail(f"written file has {len(check)} tensors, expected 160")
    for n in names - mlp_names:
        if not torch.equal(check[n], base[n]):
            fail(f"{n}: unchanged tensor was altered on disk")
    print(f"OK: merged {merged} MLP tensors, wrote {len(check)} tensors to {dst}")


if __name__ == "__main__":
    main()
