"""T2: structured attention-head pruning for Pythia-1B.

Removes head 5 from every layer at the checkpoint level:
  - fused query_key_value weight/bias: drop the head's 768-row block
    (GPT-NeoX interleaved layout, head h owns rows 768*h .. 768*h+767)
  - attention.dense weight: drop the head's 256-wide column block
"""

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
SANDBOX = HERE.parent.parent
SRC = SANDBOX / "inputs" / "base" / "model.safetensors"
DST = HERE / "model.safetensors"

NUM_LAYERS = 16
NUM_HEADS = 8
HEAD_DIM = 256
QKV_BLOCK = 3 * HEAD_DIM  # 768 rows per head in the fused projection
PRUNE_HEAD = 5


def drop_block(t: torch.Tensor, dim: int, start: int, size: int) -> torch.Tensor:
    keep = torch.cat(
        [t.narrow(dim, 0, start), t.narrow(dim, start + size, t.shape[dim] - start - size)],
        dim=dim,
    )
    return keep.contiguous()


def main() -> None:
    with safe_open(SRC, framework="pt") as f:
        metadata = f.metadata()
        state = {k: f.get_tensor(k) for k in f.keys()}

    n_in = len(state)
    print(f"loaded {n_in} tensors from {SRC}")

    qkv_start = QKV_BLOCK * PRUNE_HEAD  # 3840
    dense_start = HEAD_DIM * PRUNE_HEAD  # 1280

    for i in range(NUM_LAYERS):
        p = f"gpt_neox.layers.{i}.attention."
        for name, dim, start, size, want in (
            (p + "query_key_value.weight", 0, qkv_start, QKV_BLOCK, (5376, 2048)),
            (p + "query_key_value.bias", 0, qkv_start, QKV_BLOCK, (5376,)),
            (p + "dense.weight", 1, dense_start, HEAD_DIM, (2048, 1792)),
        ):
            if name not in state:
                raise KeyError(f"missing expected tensor {name}")
            pruned = drop_block(state[name], dim, start, size)
            if tuple(pruned.shape) != want:
                raise ValueError(f"{name}: got {tuple(pruned.shape)}, expected {want}")
            if pruned.dtype != state[name].dtype:
                raise TypeError(f"{name}: dtype changed to {pruned.dtype}")
            state[name] = pruned

    # Required checks: fail loudly before writing.
    checks = {
        "gpt_neox.layers.0.attention.query_key_value.weight": (5376, 2048),
        "gpt_neox.layers.0.attention.query_key_value.bias": (5376,),
        "gpt_neox.layers.0.attention.dense.weight": (2048, 1792),
    }
    for name, want in checks.items():
        got = tuple(state[name].shape)
        if got != want:
            raise AssertionError(f"check failed: {name} has shape {got}, expected {want}")
        print(f"check ok: {name} {got}")

    if len(state) != 244:
        raise AssertionError(f"check failed: output has {len(state)} tensors, expected 244")
    if len(state) != n_in:
        raise AssertionError(f"tensor count changed: {n_in} -> {len(state)}")
    print(f"check ok: 244 tensors")

    DST.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, DST, metadata=metadata)
    print(f"wrote {DST}")


if __name__ == "__main__":
    main()
