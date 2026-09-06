"""T4: task-vector merge of two OLMo-1B fine-tunes into the base.

Plain torch + safetensors script: the required checks (shared-tensor
verification, exactly 48 merged, exactly 114 output tensors) are assertions
here, so any violation aborts the run.
"""

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LAMBDA = 0.4
IN = Path("inputs")
OUT = Path("out/T4/model.safetensors")

MLP = {
    f"model.layers.{i}.mlp.{p}.weight"
    for i in range(16)
    for p in ("gate_proj", "up_proj", "down_proj")
}


class Ckpt:
    """Lazy reader over either a sharded directory or a single file."""

    def __init__(self, path: Path):
        path = Path(path)
        if path.is_dir():
            index = path / "model.safetensors.index.json"
            if index.exists():
                weight_map = json.loads(index.read_text())["weight_map"]
                self._loc = {k: path / v for k, v in weight_map.items()}
            else:
                files = sorted(path.glob("*.safetensors"))
                self._loc = {}
                for f in files:
                    with safe_open(f, framework="pt") as h:
                        for k in h.keys():
                            self._loc[k] = f
        else:
            with safe_open(path, framework="pt") as h:
                self._loc = {k: path for k in h.keys()}
        self._handles: dict[Path, object] = {}

    def keys(self) -> set[str]:
        return set(self._loc)

    def get(self, name: str) -> torch.Tensor:
        f = self._loc[name]
        if f not in self._handles:
            self._handles[f] = safe_open(f, framework="pt")
        return self._handles[f].get_tensor(name)


def main() -> int:
    base = Ckpt(IN / "base")
    ft1 = Ckpt(IN / "ft1/model.safetensors")
    ft2 = Ckpt(IN / "ft2/model.safetensors")

    # --- check 1: identical key sets ---------------------------------------
    kb, k1, k2 = base.keys(), ft1.keys(), ft2.keys()
    if kb != k1 or kb != k2:
        raise SystemExit(
            f"tensor name sets differ: ft1 only={sorted(k1 - kb)} base only={sorted(kb - k1)} "
            f"ft2 only={sorted(k2 - kb)} base only vs ft2={sorted(kb - k2)}"
        )
    if len(kb) != 114:
        raise SystemExit(f"expected 114 tensors in the base, found {len(kb)}")

    missing = MLP - kb
    if missing:
        raise SystemExit(f"expected MLP tensors are absent: {sorted(missing)}")
    if len(MLP) != 48:
        raise SystemExit(f"MLP name set is not 48 tensors: {len(MLP)}")

    # --- check 1b: everything outside the MLP set is identical in all three -
    out: dict[str, torch.Tensor] = {}
    merged = 0
    for name in sorted(kb):
        b = base.get(name)
        a = ft1.get(name)
        c = ft2.get(name)
        for other, tag in ((a, "ft1"), (c, "ft2")):
            if other.shape != b.shape or other.dtype != b.dtype:
                raise SystemExit(
                    f"{name}: {tag} has shape/dtype {tuple(other.shape)}/{other.dtype}, "
                    f"base has {tuple(b.shape)}/{b.dtype}"
                )
        if name in MLP:
            if b.dtype != torch.float32:
                raise SystemExit(f"{name}: expected float32, found {b.dtype}")
            # task vectors are both taken against the *unmodified* base
            tv1 = a.to(torch.float32) - b.to(torch.float32)
            tv2 = c.to(torch.float32) - b.to(torch.float32)
            out[name] = (b.to(torch.float32) + LAMBDA * tv1 + LAMBDA * tv2).contiguous()
            merged += 1
        else:
            if not torch.equal(b, a) or not torch.equal(b, c):
                raise SystemExit(
                    f"{name} is outside the MLP set but differs between the checkpoints; "
                    "the frozen-backbone assumption does not hold"
                )
            out[name] = b.clone().contiguous()

    # --- check 2: exactly 48 merged ----------------------------------------
    if merged != 48:
        raise SystemExit(f"merged {merged} tensors, expected 48")

    # --- check 3: exactly 114 output tensors -------------------------------
    if len(out) != 114:
        raise SystemExit(f"output has {len(out)} tensors, expected 114")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT))

    with safe_open(OUT, framework="pt") as h:
        written = set(h.keys())
    if written != kb:
        raise SystemExit("written file does not have the expected tensor names")
    if len(written) != 114:
        raise SystemExit(f"written file has {len(written)} tensors, expected 114")

    print(f"merged {merged} MLP tensors, copied {len(out) - merged}, wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
