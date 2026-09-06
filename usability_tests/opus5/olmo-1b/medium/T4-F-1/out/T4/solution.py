"""T4: task-vector merge of two fine-tunes of OLMo-1B-0724-hf.

out[X] = base[X] + lambda*(ft1[X]-base[X]) + lambda*(ft2[X]-base[X])
for the 48 MLP tensors; every other tensor is copied from the base unchanged.

Tooling: safetensors (0.5.3) for lazy per-tensor checkpoint I/O and torch
(2.14.0) for the float32 arithmetic. Tensors are read one name at a time via
safe_open so the three 5 GB checkpoints are never all resident.
"""

from __future__ import annotations

import json
import re
import sys
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # sandbox root
BASE_DIR = ROOT / "inputs" / "base"
FT1 = ROOT / "inputs" / "ft1" / "model.safetensors"
FT2 = ROOT / "inputs" / "ft2" / "model.safetensors"
OUT = HERE / "model.safetensors"

LAMBDA = 0.4
EXPECTED_TOTAL = 114
EXPECTED_MERGED = 48
MLP_RE = re.compile(r"^model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")


class CheckFailed(RuntimeError):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


class Shards:
    """Read-only view over one or more safetensors files, keyed by tensor name."""

    def __init__(self, stack: ExitStack, files: list[Path]) -> None:
        self.owner: dict[str, object] = {}
        for path in files:
            handle = stack.enter_context(safe_open(str(path), framework="pt", device="cpu"))
            for name in handle.keys():
                check(name not in self.owner, f"duplicate tensor {name!r} across shards")
                self.owner[name] = handle

    @property
    def names(self) -> set[str]:
        return set(self.owner)

    def get(self, name: str) -> torch.Tensor:
        return self.owner[name].get_tensor(name)


def base_shard_files() -> list[Path]:
    index = json.loads((BASE_DIR / "model.safetensors.index.json").read_text())
    files = sorted({BASE_DIR / f for f in index["weight_map"].values()})
    check(bool(files), "base index lists no shard files")
    return files


def main() -> int:
    with ExitStack() as stack:
        base = Shards(stack, base_shard_files())
        ft1 = Shards(stack, [FT1])
        ft2 = Shards(stack, [FT2])

        # --- step 1: same tensor names in all three checkpoints ---------------
        names = base.names
        check(
            names == ft1.names,
            f"ft1 key set differs from base: only-base={sorted(names - ft1.names)[:5]} "
            f"only-ft1={sorted(ft1.names - names)[:5]}",
        )
        check(
            names == ft2.names,
            f"ft2 key set differs from base: only-base={sorted(names - ft2.names)[:5]} "
            f"only-ft2={sorted(ft2.names - names)[:5]}",
        )
        check(
            len(names) == EXPECTED_TOTAL,
            f"expected {EXPECTED_TOTAL} tensors in the base, found {len(names)}",
        )

        mlp = sorted(n for n in names if MLP_RE.match(n))
        check(
            len(mlp) == EXPECTED_MERGED,
            f"expected {EXPECTED_MERGED} MLP tensors, matched {len(mlp)}: {mlp}",
        )
        mlp_set = set(mlp)

        # --- step 1 (cont.): everything outside the MLP set is identical ------
        merged: dict[str, torch.Tensor] = {}
        n_merged = 0
        for name in sorted(names):
            b = base.get(name)
            t1 = ft1.get(name)
            t2 = ft2.get(name)
            for tag, t in (("ft1", t1), ("ft2", t2)):
                check(
                    t.shape == b.shape and t.dtype == b.dtype,
                    f"{tag}[{name}] is {tuple(t.shape)}/{t.dtype}, "
                    f"base is {tuple(b.shape)}/{b.dtype}",
                )
            if name in mlp_set:
                check(b.dtype == torch.float32, f"{name}: expected float32, got {b.dtype}")
                # Both task vectors are taken against the *unmodified* base.
                out = b + LAMBDA * (t1 - b) + LAMBDA * (t2 - b)
                check(out.dtype == torch.float32, f"{name}: merged dtype {out.dtype}")
                merged[name] = out
                n_merged += 1
            else:
                check(
                    torch.equal(b, t1),
                    f"shared-tensor check failed: {name} differs between base and ft1",
                )
                check(
                    torch.equal(b, t2),
                    f"shared-tensor check failed: {name} differs between base and ft2",
                )
                merged[name] = b  # bit-exact copy of the base

        # --- required checks on the result -----------------------------------
        check(n_merged == EXPECTED_MERGED, f"merged {n_merged} tensors, expected {EXPECTED_MERGED}")
        check(
            len(merged) == EXPECTED_TOTAL,
            f"output has {len(merged)} tensors, expected {EXPECTED_TOTAL}",
        )

        OUT.parent.mkdir(parents=True, exist_ok=True)
        save_file({k: v.contiguous() for k, v in merged.items()}, str(OUT))

    # --- post-write verification of the written file -------------------------
    with safe_open(str(OUT), framework="pt", device="cpu") as f:
        written = set(f.keys())
    check(
        len(written) == EXPECTED_TOTAL,
        f"written file has {len(written)} tensors, expected {EXPECTED_TOTAL}",
    )
    check(written == names, "written key set differs from the base key set")

    print(f"OK: wrote {OUT} with {len(written)} tensors, {n_merged} merged (lambda={LAMBDA})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CheckFailed as exc:
        print(f"CHECK FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
