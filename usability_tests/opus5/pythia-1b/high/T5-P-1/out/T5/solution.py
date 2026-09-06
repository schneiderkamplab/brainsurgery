#!/usr/bin/env python
"""T5: merge a PEFT LoRA adapter into the Pythia-1B base weights and write a
sharded safetensors checkpoint.

Run:  python out/T5/solution.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # sandbox root
BASE = ROOT / "inputs" / "base" / "model.safetensors"
LORA = ROOT / "inputs" / "lora" / "adapter_model.safetensors"
LORA_CFG = ROOT / "inputs" / "lora" / "adapter_config.json"
OUT_DIR = ROOT / "out" / "T5"

SHARD_LIMIT = 512 * 1024 * 1024  # 536_870_912 bytes of tensor data per shard
EXPECTED_TENSORS = 244
EXPECTED_PAIRS = 16
PEFT_PREFIX = "base_model.model."
SENTINEL = "gpt_neox.layers.0.attention.query_key_value.weight"


class CheckFailed(RuntimeError):
    """A required check did not hold."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


def load_ordered(path: Path) -> dict[str, torch.Tensor]:
    """Load a safetensors file preserving the file's own tensor order."""
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return {name: f.get_tensor(name) for name in f.keys()}


def header_order(path: Path) -> list[str]:
    """Tensor names in the order they are laid out in the file (by data offset)."""
    with open(path, "rb") as fh:
        header_len = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(header_len))
    entries = [(v["data_offsets"][0], k) for k, v in header.items() if k != "__metadata__"]
    entries.sort()
    return [name for _, name in entries]


def nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def main() -> None:
    # ---------------------------------------------------------------- config
    cfg = json.loads(LORA_CFG.read_text())
    r = int(cfg["r"])
    alpha = float(cfg["lora_alpha"])
    fan_in_fan_out = bool(cfg["fan_in_fan_out"])
    check(r > 0, f"adapter r must be positive, got {r}")
    scale = alpha / r
    print(f"adapter: r={r} lora_alpha={alpha} scale={scale} fan_in_fan_out={fan_in_fan_out}")

    # ------------------------------------------------------------- load base
    order = header_order(BASE)
    base = load_ordered(BASE)
    check(
        len(base) == EXPECTED_TENSORS,
        f"base has {len(base)} tensors, expected {EXPECTED_TENSORS}",
    )
    check(sorted(order) == sorted(base), "header order does not cover the base tensor set")
    print(f"base: {len(base)} tensors, {sum(nbytes(t) for t in base.values())} bytes")

    # ------------------------------------------------------------ load lora
    lora = load_ordered(LORA)
    print(f"adapter: {len(lora)} tensors")

    # Pair up lora_A / lora_B by the module they adapt.
    pat = re.compile(r"^(?P<mod>.+)\.lora_(?P<ab>[AB])(?:\.default)?\.weight$")
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in lora.items():
        m = pat.match(name)
        check(m is not None, f"unrecognised adapter tensor name: {name}")
        module = m.group("mod")
        if module.startswith(PEFT_PREFIX):
            module = module[len(PEFT_PREFIX) :]
        slot = pairs.setdefault(module, {})
        check(m.group("ab") not in slot, f"duplicate lora_{m.group('ab')} for {module}")
        slot[m.group("ab")] = tensor

    for module, slot in pairs.items():
        check(set(slot) == {"A", "B"}, f"incomplete A/B pair for {module}: {sorted(slot)}")

    # REQUIRED CHECK: exactly 16 adapter pairs.
    check(
        len(pairs) == EXPECTED_PAIRS,
        f"found {len(pairs)} adapter pairs, expected {EXPECTED_PAIRS}",
    )

    # ---------------------------------------------------------------- merge
    merged = 0
    for module in sorted(pairs):
        A = pairs[module]["A"]  # [r, in]
        B = pairs[module]["B"]  # [out, r]
        target = f"{module}.weight"
        check(target in base, f"adapter targets {target}, which is not in the base checkpoint")
        W = base[target]

        check(A.ndim == 2 and B.ndim == 2, f"{module}: lora factors must be 2-D")
        check(A.shape[0] == r, f"{module}: lora_A first dim {A.shape[0]} != r={r}")
        check(B.shape[1] == r, f"{module}: lora_B second dim {B.shape[1]} != r={r}")

        delta = (B.to(torch.float32) @ A.to(torch.float32)) * scale  # [out, in]
        if fan_in_fan_out:
            # base stored as [in, out] (Conv1D-style); the product is [out, in].
            delta = delta.T
        check(
            tuple(delta.shape) == tuple(W.shape),
            f"{module}: delta {tuple(delta.shape)} does not match base {tuple(W.shape)}",
        )

        out_dtype = W.dtype
        new_W = (W.to(torch.float32) + delta).to(out_dtype).contiguous()
        check(new_W.shape == W.shape, f"{module}: merge changed the shape")
        check(new_W.dtype == out_dtype, f"{module}: merge changed the dtype")
        base[target] = new_W
        merged += 1

    check(merged == EXPECTED_PAIRS, f"merged {merged} modules, expected {EXPECTED_PAIRS}")
    print(f"merged {merged} modules with scale {scale}")

    # ------------------------------------------------------- required checks
    # No adapter or intermediate tensor in the output.
    leaked = [k for k in base if "lora_" in k]
    check(not leaked, f"adapter tensors leaked into the output: {leaked[:5]}")

    # The sentinel keeps its name, shape and dtype.
    check(SENTINEL in base, f"{SENTINEL} is missing from the output")
    check(
        tuple(base[SENTINEL].shape) == (6144, 2048),
        f"{SENTINEL} has shape {tuple(base[SENTINEL].shape)}, expected (6144, 2048)",
    )
    check(
        base[SENTINEL].dtype == torch.float16,
        f"{SENTINEL} has dtype {base[SENTINEL].dtype}, expected torch.float16",
    )

    # Exactly 244 tensors, with the same names as the base.
    check(
        len(base) == EXPECTED_TENSORS,
        f"output has {len(base)} tensors, expected {EXPECTED_TENSORS}",
    )
    check(sorted(base) == sorted(order), "output key set differs from the base key set")

    # ---------------------------------------------------------------- shard
    # Greedy packing in the checkpoint's own tensor order: fill a shard until
    # the next tensor would push it past the budget, then start a new one.  A
    # tensor bigger than the budget therefore ends up alone in its own shard.
    shards: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for name in order:
        size = nbytes(base[name])
        if current and current_size + size > SHARD_LIMIT:
            shards.append(current)
            current, current_size = [], 0
        current.append(name)
        current_size += size
    if current:
        shards.append(current)

    n = len(shards)
    check(n >= 1, "no shards were produced")
    filenames = [f"model-{i + 1:05d}-of-{n:05d}.safetensors" for i in range(n)]

    weight_map: dict[str, str] = {}
    total_size = 0
    for fname, names in zip(filenames, shards):
        shard_size = sum(nbytes(base[k]) for k in names)
        check(
            shard_size <= SHARD_LIMIT,
            f"{fname} holds {shard_size} bytes of tensor data, over the {SHARD_LIMIT} budget",
        )
        if len(names) > 1:
            check(
                max(nbytes(base[k]) for k in names) <= SHARD_LIMIT,
                f"{fname} mixes an oversized tensor with others",
            )
        total_size += shard_size
        for k in names:
            weight_map[k] = fname

    check(
        len(weight_map) == EXPECTED_TENSORS,
        f"weight_map covers {len(weight_map)} tensors, expected {EXPECTED_TENSORS}",
    )
    check(sorted(weight_map) == sorted(base), "weight_map does not cover exactly the output keys")

    # ---------------------------------------------------------------- write
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in list(OUT_DIR.glob("*.safetensors")) + list(OUT_DIR.glob("*.index.json")):
        stale.unlink()

    for fname, names in zip(filenames, shards):
        shard = {k: base[k].contiguous() for k in names}
        save_file(shard, str(OUT_DIR / fname), metadata={"format": "pt"})
        print(f"wrote {fname}: {len(names)} tensors, {sum(nbytes(t) for t in shard.values())} bytes")

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (OUT_DIR / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"wrote model.safetensors.index.json: {n} shards, total_size={total_size}")

    # --------------------------------------------------------- verify output
    seen: dict[str, str] = {}
    for fname in filenames:
        path = OUT_DIR / fname
        check(path.exists(), f"{fname} was not written")
        check(os.path.getsize(path) > 0, f"{fname} is empty")
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for k in f.keys():
                check(k not in seen, f"{k} appears in both {seen.get(k)} and {fname}")
                seen[k] = fname
                t = f.get_slice(k)
                check(
                    tuple(t.get_shape()) == tuple(base[k].shape),
                    f"{k}: shape changed on write",
                )
    check(
        len(seen) == EXPECTED_TENSORS,
        f"re-read {len(seen)} tensors from the shards, expected {EXPECTED_TENSORS}",
    )
    check(seen == weight_map, "the shards on disk disagree with the index weight_map")
    check(not [k for k in seen if "lora_" in k], "adapter tensors present in the written shards")
    print(f"OK: {len(seen)} tensors across {n} shards in {OUT_DIR}")


if __name__ == "__main__":
    main()
