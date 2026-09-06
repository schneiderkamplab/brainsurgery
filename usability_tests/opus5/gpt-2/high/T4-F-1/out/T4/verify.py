"""Independent verification of out/T4/model.safetensors (does not write the output).

1. Recomputes the merge with a different, algebraically equivalent expression
   and compares relative Frobenius error against the produced file.
2. Confirms the 112 non-MLP tensors are bit-identical to the base.
3. Negative test: runs solution.main() against synthetic checkpoints in which a
   non-MLP tensor differs, and asserts that it aborts.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import solution as S

LAM = S.LAMBDA
mlp = set(S.MLP_KEYS)

base = safe_open(S.IN / "base" / "model.safetensors", framework="pt")
ft1 = safe_open(S.IN / "ft1" / "model.safetensors", framework="pt")
ft2 = safe_open(S.IN / "ft2" / "model.safetensors", framework="pt")
got = safe_open(S.OUT, framework="pt")

keys = sorted(got.keys())
assert len(keys) == 160, len(keys)
assert keys == sorted(base.keys())
assert len(mlp & set(keys)) == 48

worst = 0.0
n_exact = 0
for k in keys:
    b, a, c, g = base.get_tensor(k), ft1.get_tensor(k), ft2.get_tensor(k), got.get_tensor(k)
    assert g.shape == b.shape and g.dtype == b.dtype, k
    if k in mlp:
        # equivalent form: (1 - 2*lambda) * base + lambda*ft1 + lambda*ft2
        ref = (1.0 - 2.0 * LAM) * b + LAM * a + LAM * c
        rel = (g - ref).norm().item() / max(ref.norm().item(), 1e-30)
        worst = max(worst, rel)
        # the merge must actually have changed the tensor
        assert not torch.equal(g, b), f"{k} unchanged by the merge"
    else:
        assert torch.equal(g, b), f"{k} not bit-identical to the base"
        n_exact += 1
assert n_exact == 112, n_exact
assert worst <= 1e-5, worst
print(f"values ok: 112 bit-exact, 48 merged, worst relative Frobenius error {worst:.3e}")

# --- negative test ------------------------------------------------------
names = keys
with tempfile.TemporaryDirectory(dir=str(S.OUT.parent)) as td:
    root = Path(td)
    for d in ("base", "ft1", "ft2"):
        (root / d).mkdir()
        sd = {n: torch.zeros(2, dtype=torch.float32) for n in names}
        if d == "ft2":
            sd["h.3.attn.c_attn.weight"] = torch.ones(2, dtype=torch.float32)  # tampered backbone
        save_file(sd, str(root / d / "model.safetensors"))
    old_in, old_out = S.IN, S.OUT
    S.IN, S.OUT = root, root / "out.safetensors"
    try:
        S.main()
    except S.CheckFailed as exc:
        print(f"negative test ok: aborted with -> {exc}")
    else:
        raise AssertionError("negative test FAILED: tampered backbone was not detected")
    finally:
        S.IN, S.OUT = old_in, old_out
