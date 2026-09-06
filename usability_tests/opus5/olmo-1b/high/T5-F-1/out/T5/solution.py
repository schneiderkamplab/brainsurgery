"""T5: fold a PEFT LoRA adapter into OLMo-1B base weights and write a sharded
safetensors checkpoint.

Tools: safetensors (streaming load/save) + torch (matmul), plus the adapter's
own adapter_config.json for r / lora_alpha / fan_in_fan_out. The merge is done
on the checkpoint files, so the model is never instantiated.
"""

from __future__ import annotations

import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
SANDBOX = HERE.parent.parent
BASE_DIR = SANDBOX / "inputs" / "base"
LORA_DIR = SANDBOX / "inputs" / "lora"
OUT_DIR = SANDBOX / "out" / "T5"

SHARD_BUDGET = 512 * 1024 * 1024  # 536,870,912 bytes of tensor data per shard
# Tensors the task requires to be stored alone in their own shard.
SOLO = {"model.embed_tokens.weight", "lm_head.weight"}
INDEX_NAME = "model.safetensors.index.json"

EXPECTED_PAIRS = 32
EXPECTED_TENSORS = 114
PROBE = "model.layers.0.self_attn.q_proj.weight"
PROBE_SHAPE = (2048, 2048)

ST_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}

# base_model.model.<base name minus ".weight">.lora_{A,B}.weight
ADAPTER_RE = re.compile(r"^base_model\.model\.(?P<mod>.+)\.lora_(?P<side>[AB])\.weight$")


def fail(msg: str) -> None:
    raise SystemExit(f"CHECK FAILED: {msg}")


def nbytes_of(shape, dtype: torch.dtype) -> int:
    n = torch.empty(0, dtype=dtype).element_size()
    for d in shape:
        n *= d
    return n


def load_adapter() -> tuple[dict[str, dict[str, torch.Tensor]], float]:
    cfg = json.loads((LORA_DIR / "adapter_config.json").read_text())
    r, alpha = cfg["r"], cfg["lora_alpha"]
    if not r:
        fail("adapter_config.json has r = 0")
    if cfg.get("fan_in_fan_out", False):
        # Base weights here are nn.Linear [out, in]; a fan_in_fan_out adapter
        # would need (B @ A).T instead.  Refuse rather than silently transpose.
        fail("adapter_config.json sets fan_in_fan_out = true; this script only handles false")
    scale = alpha / r

    pairs: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(LORA_DIR / "adapter_model.safetensors", framework="pt") as f:
        for key in f.keys():
            m = ADAPTER_RE.match(key)
            if m is None:
                fail(f"unrecognised adapter tensor name: {key}")
            target = f"{m.group('mod')}.weight"
            pairs.setdefault(target, {})[m.group("side")] = f.get_tensor(key)

    for target, sides in pairs.items():
        if set(sides) != {"A", "B"}:
            fail(f"incomplete LoRA pair for {target}: have {sorted(sides)}")
    return pairs, scale


def plan_shards(order: list[str], sizes: dict[str, int]) -> list[list[str]]:
    shards: list[list[str]] = []
    cur: list[str] = []
    cur_bytes = 0
    for name in order:
        n = sizes[name]
        if name in SOLO or n > SHARD_BUDGET:
            if cur:
                shards.append(cur)
            shards.append([name])
            cur, cur_bytes = [], 0
            continue
        if cur and cur_bytes + n > SHARD_BUDGET:
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append(name)
        cur_bytes += n
    if cur:
        shards.append(cur)
    return shards


def clear_previous_output() -> None:
    """Remove checkpoint files from a previous run, but keep authored files
    (this script, run.sh, REPORT.md) that also live in out/T5."""
    if not OUT_DIR.exists():
        OUT_DIR.mkdir(parents=True)
        return
    for path in OUT_DIR.iterdir():
        if path.name == INDEX_NAME or (
            path.name.startswith("model-") and path.name.endswith(".safetensors")
        ):
            path.unlink()


def verify_output(
    filenames: list[str],
    base_names: list[str],
    handles: dict,
    weight_map: dict[str, str],
    pairs: dict[str, dict[str, torch.Tensor]],
    scale: float,
) -> None:
    """Re-read what was written and re-run the required checks against disk."""
    index = json.loads((OUT_DIR / INDEX_NAME).read_text())
    out_map: dict[str, str] = index["weight_map"]

    on_disk = sorted(
        p.name
        for p in OUT_DIR.iterdir()
        if p.name.startswith("model-") and p.name.endswith(".safetensors")
    )
    if on_disk != sorted(filenames):
        fail(f"unexpected shard files in the output: {on_disk} vs {sorted(filenames)}")

    seen: dict[str, str] = {}
    for fname in filenames:
        with safe_open(OUT_DIR / fname, framework="pt") as f:
            names = list(f.keys())
            total = 0
            for name in names:
                sl = f.get_slice(name)
                total += nbytes_of(sl.get_shape(), ST_DTYPES[sl.get_dtype()])
                seen[name] = fname
            if total > SHARD_BUDGET and len(names) > 1:
                fail(f"{fname} on disk holds {total} bytes, over the {SHARD_BUDGET} budget")

    if len(seen) != EXPECTED_TENSORS:
        fail(f"output holds {len(seen)} tensors, expected {EXPECTED_TENSORS}")
    if sorted(seen) != base_names:
        fail("output key set differs from the base key set")
    leaked = sorted(n for n in seen if "lora_" in n)
    if leaked:
        fail(f"adapter tensor names present in the output: {leaked}")
    if seen != out_map:
        fail("index weight_map does not match the tensors actually present in the shards")

    with safe_open(OUT_DIR / seen[PROBE], framework="pt") as f:
        probe = f.get_tensor(PROBE)
    if tuple(probe.shape) != PROBE_SHAPE:
        fail(f"{PROBE} has shape {tuple(probe.shape)} in the output, expected {PROBE_SHAPE}")
    if probe.dtype is not torch.float32:
        fail(f"{PROBE} has dtype {probe.dtype} in the output, expected float32")

    # Spot-check every merged weight and a sample of the untouched ones.
    for name in base_names:
        base = handles[weight_map[name]].get_tensor(name)
        if name in pairs:
            want = base.to(torch.float32) + scale * (pairs[name]["B"] @ pairs[name]["A"])
            with safe_open(OUT_DIR / seen[name], framework="pt") as f:
                got = f.get_tensor(name)
            err = torch.linalg.norm(got - want) / torch.linalg.norm(want)
            if not (err <= 1e-6):
                fail(f"{name}: relative Frobenius error {err:.3e} against the expected merge")
            if torch.equal(got, base):
                fail(f"{name}: output is bit-identical to the base, the delta was not applied")
        elif name in SOLO or name.endswith(
            (".0.mlp.down_proj.weight", ".0.self_attn.k_proj.weight")
        ):
            with safe_open(OUT_DIR / seen[name], framework="pt") as f:
                got = f.get_tensor(name)
            if not torch.equal(got, base):
                fail(f"{name}: unchanged tensor differs from the base")


def main() -> None:
    base_index = json.loads((BASE_DIR / INDEX_NAME).read_text())
    weight_map: dict[str, str] = base_index["weight_map"]
    base_names = sorted(weight_map)

    pairs, scale = load_adapter()

    # --- check: exactly 32 adapter pairs, all naming a real base tensor ---
    if len(pairs) != EXPECTED_PAIRS:
        fail(f"expected {EXPECTED_PAIRS} adapter pairs, found {len(pairs)}")
    missing = sorted(t for t in pairs if t not in weight_map)
    if missing:
        fail(f"adapter targets absent from the base checkpoint: {missing}")

    handles = {
        fname: safe_open(BASE_DIR / fname, framework="pt")
        for fname in sorted(set(weight_map.values()))
    }

    sizes: dict[str, int] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    for name in base_names:
        sl = handles[weight_map[name]].get_slice(name)
        dtype = ST_DTYPES.get(sl.get_dtype())
        if dtype is None:
            fail(f"{name} has unsupported dtype {sl.get_dtype()}")
        shapes[name] = tuple(sl.get_shape())
        sizes[name] = nbytes_of(shapes[name], dtype)

    shards = plan_shards(base_names, sizes)
    n_shards = len(shards)
    filenames = [f"model-{i + 1:05d}-of-{n_shards:05d}.safetensors" for i in range(n_shards)]

    # --- checks on the planned output, before anything is written ---
    planned = [name for shard in shards for name in shard]
    if len(planned) != EXPECTED_TENSORS:
        fail(f"output would hold {len(planned)} tensors, expected {EXPECTED_TENSORS}")
    if sorted(planned) != base_names:
        fail("planned output key set differs from the base key set")
    leaked = [n for n in planned if "lora_" in n]
    if leaked:
        fail(f"adapter tensor names leaked into the output: {leaked}")
    if shapes.get(PROBE) != PROBE_SHAPE:
        fail(f"{PROBE} has shape {shapes.get(PROBE)}, expected {PROBE_SHAPE}")
    for fname, shard in zip(filenames, shards):
        total = sum(sizes[n] for n in shard)
        if total > SHARD_BUDGET and len(shard) > 1:
            fail(f"{fname} holds {total} bytes of tensor data, over the {SHARD_BUDGET} budget")

    clear_previous_output()

    merged: set[str] = set()
    index_map: OrderedDict[str, str] = OrderedDict()
    total_size = 0
    for fname, shard in zip(filenames, shards):
        tensors: dict[str, torch.Tensor] = {}
        for name in shard:
            t = handles[weight_map[name]].get_tensor(name)
            if name in pairs:
                if t.dtype is not torch.float32:
                    fail(f"{name} is {t.dtype}, expected float32 for the merge")
                a = pairs[name]["A"].to(torch.float32)
                b = pairs[name]["B"].to(torch.float32)
                delta = scale * (b @ a)
                if delta.shape != t.shape:
                    fail(f"{name}: delta {tuple(delta.shape)} vs base {tuple(t.shape)}")
                t = t.to(torch.float32) + delta
                merged.add(name)
            tensors[name] = t.contiguous()
            index_map[name] = fname
            total_size += sizes[name]
        save_file(tensors, str(OUT_DIR / fname), metadata={"format": "pt"})
        del tensors

    if len(merged) != EXPECTED_PAIRS:
        fail(f"merged {len(merged)} weights, expected {EXPECTED_PAIRS}")

    (OUT_DIR / INDEX_NAME).write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": index_map}, indent=2)
        + "\n"
    )

    verify_output(filenames, base_names, handles, weight_map, pairs, scale)

    print(f"merged {len(merged)} LoRA pairs at scale {scale:g}")
    print(f"wrote {len(planned)} tensors into {n_shards} shards under {OUT_DIR}")
    for fname, shard in zip(filenames, shards):
        print(f"  {fname}: {len(shard):3d} tensors, {sum(sizes[n] for n in shard)} bytes")


if __name__ == "__main__":
    sys.exit(main())
