"""T3: mixed-precision export with sharding for OLMo-1B-0724-hf.

Casts the 112 attention/MLP projection matrices to bfloat16, keeps everything
else (embeddings, lm_head) in float32, and writes a sharded safetensors
checkpoint with a 256 MiB shard budget plus an index file.
"""

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

IN_DIR = Path("inputs/base")
OUT_DIR = Path("out/T3")
SHARD_LIMIT = 256 * 1024 * 1024  # 268,435,456 bytes of tensor data

# Exactly the projection matrices named in the task: per-layer q/k/v/o and
# gate/up/down. Anchored so it cannot reach embed_tokens or lm_head.
PROJ_RE = re.compile(
    r"^model\.layers\.\d+\.(?:self_attn\.[qkvo]_proj|mlp\.(?:gate|up|down)_proj)\.weight$"
)


def load_state_dict() -> dict[str, torch.Tensor]:
    index = json.loads((IN_DIR / "model.safetensors.index.json").read_text())
    shards: dict[str, list[str]] = {}
    for name, shard in index["weight_map"].items():
        shards.setdefault(shard, []).append(name)
    state: dict[str, torch.Tensor] = {}
    for shard, names in shards.items():
        with safe_open(IN_DIR / shard, framework="pt") as f:
            for name in names:
                state[name] = f.get_tensor(name)
    return state


def plan_shards(names: list[str], sizes: dict[str, int]) -> list[list[str]]:
    """Greedy packing in sorted key order; a tensor over the budget goes alone."""
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name in names:
        size = sizes[name]
        if size > SHARD_LIMIT:
            if current:
                shards.append(current)
                current, current_bytes = [], 0
            shards.append([name])
            continue
        if current and current_bytes + size > SHARD_LIMIT:
            shards.append(current)
            current, current_bytes = [], 0
        current.append(name)
        current_bytes += size
    if current:
        shards.append(current)
    return shards


def main() -> None:
    state = load_state_dict()
    if len(state) != 114:
        raise SystemExit(f"expected 114 input tensors, got {len(state)}")

    out: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if PROJ_RE.match(name):
            out[name] = tensor.to(torch.bfloat16)
        else:
            if tensor.dtype != torch.float32:
                out[name] = tensor.to(torch.float32)
            else:
                out[name] = tensor.clone()

    # --- Required checks: fail loudly before writing anything. ---
    bf16 = [n for n, t in out.items() if t.dtype == torch.bfloat16]
    if len(bf16) != 112:
        raise SystemExit(f"expected 112 bfloat16 tensors, got {len(bf16)}")
    if out["model.layers.0.self_attn.q_proj.weight"].dtype != torch.bfloat16:
        raise SystemExit("model.layers.0.self_attn.q_proj.weight is not bfloat16")
    if out["model.embed_tokens.weight"].dtype != torch.float32:
        raise SystemExit("model.embed_tokens.weight is not float32")
    if len(out) != 114:
        raise SystemExit(f"expected 114 output tensors, got {len(out)}")
    if set(out) != set(state):
        raise SystemExit("tensor names changed")
    non_bf16 = {n: t.dtype for n, t in out.items() if t.dtype != torch.bfloat16}
    if any(d != torch.float32 for d in non_bf16.values()):
        raise SystemExit(f"non-projection tensors are not all float32: {non_bf16}")

    sizes = {n: t.numel() * t.element_size() for n, t in out.items()}
    names = sorted(out)
    shards = plan_shards(names, sizes)
    total = len(shards)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in OUT_DIR.glob("*.safetensors"):
        stale.unlink()

    weight_map: dict[str, str] = {}
    for i, shard_names in enumerate(shards, start=1):
        filename = f"model-{i:05d}-of-{total:05d}.safetensors"
        shard_bytes = sum(sizes[n] for n in shard_names)
        if shard_bytes > SHARD_LIMIT and len(shard_names) > 1:
            raise SystemExit(f"{filename} exceeds the shard budget: {shard_bytes} bytes")
        save_file(
            {n: out[n].contiguous() for n in shard_names},
            OUT_DIR / filename,
            metadata={"format": "pt"},
        )
        for n in shard_names:
            weight_map[n] = filename

    if len(weight_map) != 114:
        raise SystemExit(f"weight_map covers {len(weight_map)} tensors, expected 114")

    (OUT_DIR / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": sum(sizes.values())}, "weight_map": weight_map},
            indent=2,
        )
        + "\n"
    )

    print(f"wrote {total} shards, {len(weight_map)} tensors, {sum(sizes.values())} bytes")
    for i, shard_names in enumerate(shards, start=1):
        print(f"  shard {i}: {len(shard_names)} tensors, {sum(sizes[n] for n in shard_names)} bytes")


if __name__ == "__main__":
    main()
