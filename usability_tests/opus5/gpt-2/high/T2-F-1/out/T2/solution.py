"""T2: remove attention head 5 from every layer of GPT-2 (124M).

Route: plain torch + safetensors slicing.

Why not transformers.prune_heads: this checkpoint has flat keys (`h.0....`,
no `transformer.` prefix) and carries the `h.<i>.attn.bias` causal-mask
buffers. A GPT2Model.prune_heads + save_pretrained round-trip would re-prefix
the keys and drop the non-persistent mask buffers, which violates requirements
4 and 5 (names unchanged, exactly 160 tensors). Direct slicing is bit-exact
and touches only the three head-bearing tensors per layer.

GPT-2 uses Conv1D layout [in, out]:
  c_attn.weight [768, 2304] = [q | k | v] along dim 1; head h of segment s
      occupies columns s*768 + h*64 .. +63          -> slice dim 1
  c_attn.bias   [2304]      same layout             -> slice dim 0
  c_proj.weight [768, 768]  heads are row blocks    -> slice dim 0
"""

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = Path("inputs/base/model.safetensors")
DST = Path("out/T2/model.safetensors")

N_LAYERS = 12
N_HEADS = 12
HEAD_DIM = 64
HIDDEN = N_HEADS * HEAD_DIM  # 768
PRUNE_HEAD = 5


def keep_in_segment() -> list[int]:
    """Column indices kept inside one 768-wide segment (head 5 dropped)."""
    return [
        i
        for h in range(N_HEADS)
        if h != PRUNE_HEAD
        for i in range(h * HEAD_DIM, (h + 1) * HEAD_DIM)
    ]


def main() -> None:
    seg = keep_in_segment()
    # q | k | v: same pattern repeated at offsets 0, 768, 1536, in that order.
    keep_qkv = torch.tensor(
        [s * HIDDEN + i for s in range(3) for i in seg], dtype=torch.long
    )
    keep_out = torch.tensor(seg, dtype=torch.long)

    assert len(seg) == 704, len(seg)
    assert keep_qkv.numel() == 2112, keep_qkv.numel()

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(SRC, framework="pt") as f:
        metadata = f.metadata()
        names = list(f.keys())
        if len(names) != 160:
            raise SystemExit(f"input has {len(names)} tensors, expected 160")
        for name in names:
            t = f.get_tensor(name)
            parts = name.split(".")
            is_layer_attn = (
                len(parts) >= 4 and parts[0] == "h" and parts[2] == "attn"
            )
            suffix = ".".join(parts[3:]) if is_layer_attn else ""
            if is_layer_attn and suffix == "c_attn.weight":
                if tuple(t.shape) != (HIDDEN, 3 * HIDDEN):
                    raise SystemExit(f"{name}: unexpected shape {tuple(t.shape)}")
                t = t.index_select(1, keep_qkv)
            elif is_layer_attn and suffix == "c_attn.bias":
                if tuple(t.shape) != (3 * HIDDEN,):
                    raise SystemExit(f"{name}: unexpected shape {tuple(t.shape)}")
                t = t.index_select(0, keep_qkv)
            elif is_layer_attn and suffix == "c_proj.weight":
                if tuple(t.shape) != (HIDDEN, HIDDEN):
                    raise SystemExit(f"{name}: unexpected shape {tuple(t.shape)}")
                t = t.index_select(0, keep_out)
            tensors[name] = t.contiguous()

    # --- Required checks: fail loudly before writing ---------------------
    expected = {
        "h.0.attn.c_attn.weight": (768, 2112),
        "h.0.attn.c_attn.bias": (2112,),
        "h.0.attn.c_proj.weight": (704, 768),
    }
    for name, shape in expected.items():
        got = tuple(tensors[name].shape)
        if got != shape:
            raise SystemExit(f"check failed: {name} has shape {got}, expected {shape}")
    if len(tensors) != 160:
        raise SystemExit(f"check failed: output has {len(tensors)} tensors, expected 160")

    # Same checks on every layer, not just layer 0.
    for i in range(N_LAYERS):
        for suffix, shape in (
            ("c_attn.weight", (768, 2112)),
            ("c_attn.bias", (2112,)),
            ("c_proj.weight", (704, 768)),
        ):
            name = f"h.{i}.attn.{suffix}"
            got = tuple(tensors[name].shape)
            if got != shape:
                raise SystemExit(f"check failed: {name} has shape {got}, expected {shape}")

    DST.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(DST), metadata=metadata)
    print(f"wrote {DST} with {len(tensors)} tensors")


if __name__ == "__main__":
    main()
